import os
import ray
import torch
import time
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import psutil
import csv

DATASET_SIZES = [100, 200, 400, 1000, 2000, 5000, 10000]  # Example: Small, medium, large
BATCH_SIZES = [1, 2, 4, 8]  # Example: Small, medium, large

# Get current working directory
cwd = os.getcwd()

# Load image files
image_directory = f"{cwd}/EMBLModelExample/datasets_images/2024-01-08_11h01m57s/"
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]

# Define preprocess function
def preprocess(image_batch):
    # Define transformations
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.grid_sample(x.unsqueeze(0), torch.nn.functional.affine_grid(
            torch.eye(2, 3, dtype=torch.float32).unsqueeze(0), [1, 3, 224, 224], True), mode='bilinear',
                                                  padding_mode='reflection', align_corners=True)),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Apply transformations to each image in the batch
    preprocessed_images = [composed_transforms(Image.fromarray(img)) for img in image_batch["image"]]
    return {"image": preprocessed_images}

# Initialize Ray
ray.init()

# Open CSV file for writing results
with open('experiment_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset Size', 'Batch Size', 'Execution Time (s)', 'Throughput (images/sec)', 'CPU Before', 'CPU After'])

    # Iterate over dataset sizes and batch sizes
    for dataset_size in DATASET_SIZES:
        for batch_size in BATCH_SIZES:
            start_time = time.time()

            # Load dataset with specified size
            image_files_subset = image_files[:dataset_size]
            ds = ray.data.read_images(image_files_subset, mode="RGB", file_extensions=None)

            # Before map_batches
            cpu_before = psutil.cpu_percent(interval=None)

            # Apply preprocess function with specified batch size
            ds = ds.map_batches(preprocess, batch_format="numpy", batch_size=batch_size, num_cpus=1)

            # After map_batches
            cpu_after = psutil.cpu_percent(interval=None)

            # Iterate over the dataset to trigger preprocessing
            for _ in ds.iter_batches(batch_size=None):
                pass

            # Record metrics
            end_time = time.time()
            total_time = end_time - start_time
            throughput = dataset_size / total_time  # images/second

            # Write results to CSV
            writer.writerow([dataset_size, batch_size, total_time, throughput, cpu_before, cpu_after])

            print(f"Experiment: Dataset Size={dataset_size}, Batch Size={batch_size}")
            print(f"Total Execution Time for preprocess: {total_time} seconds")
            print(f"Throughput: {throughput} images/second")
            print(f"CPU Usage Before map_batches: {cpu_before}")
            print(f"CPU Usage After map_batches: {cpu_after}")
            print("-" * 50)