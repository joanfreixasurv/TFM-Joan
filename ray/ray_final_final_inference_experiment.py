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

ray.init()

ds_name = '2024-02-26_22h34m46s'

param_grid_inference = {
    'dataset_size': [10000],
    'parallelism_read': [-1, 50, 100, 200],
    'num_cpus_preprocess': [5.5],
    'num_cpus_inference': [5.5],
    'batch_size_preprocess': [8, 16, 32, 64],
    'batch_size_inference': [8, 16, 32, 64],
    'concurrency': [1, 2, 3, 4, 5],
    'preserve_order': [False],
    'repetitions': [1]
}

# Create all possible parameter combinations
configurations = list(itertools.product(*param_grid_inference.values()))

local_file_path = "rack_metrics_torchscript_embl_inference.csv"
s3_file_path = "pepe-bucket/joan_rack_metrics_torchscript_embl_inference.csv"
fs = s3fs.S3FileSystem(anon=False, key='lab144', secret='astl1a4b4',
                       client_kwargs={'endpoint_url': 'http://192.168.5.24:9000'})

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


class Actor:
    def __init__(self):
        self.model = torch.jit.load(f"/bin/model.pt",torch.device('cpu'))

    def __call__(self, batch):
        inputs = torch.as_tensor(batch["image"], device="cpu")
        with torch.no_grad():
            output_batch = self.model.forward(inputs)
            predictions_batch = torch.softmax(output_batch, dim=1)
            pred_probs = predictions_batch.numpy()
            preds = pred_probs.argmax(axis=1)
            labels = []
            probabilities = []
            for i in range(len(pred_probs)):
                probabilities.append(pred_probs[i][0])
                if preds[i] == 0:
                    labels.append('off')
                else:
                    labels.append('on')
            results = [{'prob': float(prob), 'label': label} for prob, label in zip(probabilities, labels)]
            return {"class": results}


def save_metrics_to_csv(metrics, file_name, job_id):
    try:
        # Try to load existing CSV file
        if fs.exists(s3_file_path):
            with fs.open(s3_file_path, 'rb') as f:
                existing_df = pd.read_csv(f)
        else:
            existing_df = pd.DataFrame()

        # Concatenate new metrics with existing DataFrame
        new_df = pd.concat([existing_df, pd.DataFrame(metrics)])

        # Save the updated DataFrame to a local CSV file
        new_df.to_csv(file_name, index=False)
        print("Updated metrics have been saved to", file_name, "for Job with id", job_id)

        # Upload the file to S3
        with fs.open(s3_file_path, 'wb') as s3_file:
            with open(file_name, 'rb') as local_file:
                s3_file.write(local_file.read())
            print("File uploaded successfully to", s3_file_path)

    except Exception as e:
        print("Failed to update metrics due to:", e)


def evaluate(params):
    dataset_size, parallelism_read, num_cpus_preprocess, num_cpus_inference, batch_size_preprocess, batch_size_inference, concurrency, preserve_order, repetitions = params

    #if ray.is_initialized:
     #   ray.shutdown()


    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True
    ctx.execution_options.preserve_order = preserve_order

    job_id = ray.get_runtime_context().get_job_id()

    start_time = time.time()

    fs = s3fs.S3FileSystem(anon=False, key='lab144', secret='astl1a4b4', client_kwargs={'endpoint_url': 'http://192.168.5.24:9000'})
    ds = ray.data.read_images("s3://pepe-bucket/2024-02-26_22h34m46s/", filesystem=fs, mode="RGB", file_extensions=None, parallelism=parallelism_read, ray_remote_args={"num_cpus": num_cpus_preprocess})

    start_time_without_metadata_fetching = time.time()

    ds = ds.map_batches(preprocess, batch_format="numpy", batch_size=batch_size_preprocess, num_cpus=num_cpus_preprocess)
    ds = ds.map_batches(Actor, batch_size=batch_size_inference, num_gpus=0, batch_format="numpy", concurrency=concurrency, num_cpus=num_cpus_inference)

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
        "Num cpus preprocess": [num_cpus_preprocess],
        "Num cpus inference": [num_cpus_inference],
        "Batch size preprocess": [batch_size_preprocess],
        "Batch size inference": [batch_size_inference],
        "Concurrency": [concurrency],
        "Repetitions": [repetitions],

        "Dataset size (bytes)": [ds.size_bytes()],
        "Num records dataset": [ds.count()],
        "Num blocks dataset": [ds.num_blocks()],

        "Total time": [total_time],
        "Throughput (img/sec)": [throughput],
        "Total time w/o metadata fetching": [total_time_without_metadata_fetching],
        "Throughput w/o metadata fetching (img/sec)": [throughput_without_metadata_fetching]
    }

    save_metrics_to_csv(metrics, "rack_metrics_torchscript_embl_inference.csv", job_id)



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