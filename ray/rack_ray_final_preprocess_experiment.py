import os
import ray
import torch
from torchvision import transforms
import time
from PIL import Image
import torch.nn.functional as F
import itertools
import pandas as pd
import s3fs


cwd = os.getcwd()

ds_name = '2024-01-03_11h10m14s'

param_grid_preprocess = {
    'dataset_size': [1772],
    'parallelism_read': [-1, 1, 25, 75, 200],                                      # -1 is the default value
    'num_cpus': [1, 2, 3, 4, 10, 25, 50],                                                      # 0 is the default value
    'batch_size_map_batches': ['default', None, 8, 64, 128, 256, 512],    # 'default' is the default value
    'preserve_order': [False]                                                     # False is the default value
}

# Create all possible parameter combinations
configurations = list(itertools.product(*param_grid_preprocess.values()))


def preprocess(image_batch):
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.grid_sample(x.unsqueeze(0), torch.nn.functional.affine_grid(
            torch.eye(2, 3, dtype=torch.float32).unsqueeze(0), [1, 3, 224, 224], True), mode='bilinear',
                                                  padding_mode='reflection', align_corners=True)),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    preprocessed_images = []
    for img in image_batch["image"]:
        img_pil = Image.fromarray(img)
        preprocessed_img = composed_transforms(img_pil)
        preprocessed_images.append(preprocessed_img.numpy())
    return {"image": preprocessed_images}


def save_metrics_to_csv(metrics, file_name, job_id):
    try:
        # Try to load existing CSV file
        existing_df = pd.read_csv(file_name)

        # Concatenate new metrics with existing DataFrame
        new_df = pd.concat([existing_df, pd.DataFrame(metrics)])

        # Save the updated DataFrame to the CSV file
        new_df.to_csv(file_name, index=False)

        print("Updated metrics have been saved to", file_name, "for Job with id", job_id)
    except FileNotFoundError:
        # If the file doesn't exist, create a new one with the metrics
        df = pd.DataFrame(metrics)
        df.to_csv(file_name, index=False)
        print("A new file", file_name, "has been created with metrics for Job with id", job_id)


def evaluate(params):
    dataset_size, parallelism_read, num_cpus, batch_size_map_batches, preserve_order = params

    if ray.is_initialized:
        ray.shutdown()

    ray.init(num_cpus=50)

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True
    ctx.execution_options.preserve_order = preserve_order

    job_id = ray.get_runtime_context().get_job_id()

    start_time = time.time()

    fs = s3fs.S3FileSystem(anon=False, key='lab144', secret='astl1a4b4', client_kwargs={'endpoint_url': 'http://192.168.5.24:9000'})
    ds = ray.data.read_images("s3://pepe-bucket/2024-01-03_11h10m14s/", filesystem=fs, mode="RGB", file_extensions=None, parallelism=parallelism_read, ray_remote_args={"num_cpus": num_cpus})

    start_time_without_metadata_fetching = time.time()

    ds = ds.map_batches(preprocess, batch_format="numpy", batch_size=batch_size_map_batches, num_cpus=num_cpus)

    for _ in ds.iter_batches(batch_size=None):
        pass

    end_time = time.time()

    num_records = ds.count()
    total_time = end_time - start_time
    throughput = num_records / total_time
    total_time_without_metadata_fetching = end_time - start_time_without_metadata_fetching
    throughput_without_metadata_fetching = num_records / total_time_without_metadata_fetching

    metrics = {
        "Job ID": [job_id],

        "Dataset": [ds_name],

        "Preserve order": [preserve_order],

        "Parallelism": [parallelism_read],
        "Num cpus": [num_cpus],
        "Batch size map_batches()": [batch_size_map_batches],

        "Dataset size (bytes)": [ds.size_bytes()],
        "Num records dataset": [ds.count()],
        "Num blocks dataset": [ds.num_blocks()],

        "Total time": [total_time],
        "Throughput (img/sec)": [throughput],
        "Total time w/o metadata fetching": [total_time_without_metadata_fetching],
        "Throughput w/o metadata fetching (img/sec)": [throughput_without_metadata_fetching]
    }

    save_metrics_to_csv(metrics, "rack_metrics_torchscript_embl_preprocess.csv", job_id)

    return throughput


best_throughput = -1
best_params = None
for config in configurations:
    throughput = evaluate(config)
    if throughput > best_throughput:
        best_throughput = throughput
        best_params = config

print("Best throughput:", best_throughput)
print("Best parameters:", best_params)