import os
import ray
import torch
import time
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import psutil
import csv

DATASET_SIZE = 400

# Get cwd
cwd = os.getcwd()

image_directory = f"{cwd}/EMBLModelExample/datasets_images/2024-01-08_11h01m57s/"
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]
image_files = image_files[:DATASET_SIZE]

# Define parameter combinations
batch_sizes = [8, 16, 32]
concurrencies = [2, 4, 8]
num_cpus_values = [1, 2, 8]

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


# Create a CSV file to store the results
with open('preprocessing_results.csv', mode='w', newline='') as csv_file:
    fieldnames = ['Batch Size', 'Concurrency', 'Num CPUs', 'Execution Time (s)', 'Throughput (images/sec)', 'CPU Before', 'CPU After']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for batch_size in batch_sizes:
        for concurrency in concurrencies:
            for num_cpus in num_cpus_values:
                start_time = time.time()
                ds = ray.data.read_images(image_files, mode="RGB", file_extensions=None)

                # Before map_batches
                cpu_before = psutil.cpu_percent(interval=None)

                ds = ds.map_batches(preprocess, batch_format="numpy", num_cpus=num_cpus, concurrency=concurrency, batch_size=batch_size)

                # After map_batches
                cpu_after = psutil.cpu_percent(interval=None)

                for _ in ds.iter_batches(batch_size=batch_size):
                    pass

                end_time = time.time()

                # Calculate metrics
                execution_time = end_time - start_time
                throughput = DATASET_SIZE / execution_time
                cpu_usage_before = cpu_before
                cpu_usage_after = cpu_after

                # Write results to CSV file
                writer.writerow({'Batch Size': batch_size, 'Concurrency': concurrency, 'Num CPUs': num_cpus, 
                                 'Execution Time (s)': execution_time, 'Throughput (images/sec)': throughput, 
                                 'CPU Before': cpu_usage_before, 'CPU After': cpu_usage_after})
