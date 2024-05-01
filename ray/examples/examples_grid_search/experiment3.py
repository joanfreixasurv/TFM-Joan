import io
import os
import ray
import torch
import time
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import psutil
from ray.data.block import BlockMetadata  # Import BlockMetadata

DATASET_SIZE = 400
BATCH_SIZE = 2

# Get cwd
cwd = os.getcwd()

image_directory = f"{cwd}/EMBLModelExample/datasets_images/2024-01-08_11h01m57s/"
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]
image_files = image_files[:DATASET_SIZE]

start_time = time.time()
ds = ray.data.read_images(image_files, mode="RGB", file_extensions=None)
print("Number of images loaded:", ds.count())

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
        self.model = torch.jit.load("EMBLModelExample/commons/torchscript_model_2.1.2.pt", torch.device('cpu'))

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
                if (preds[i] == 0):
                    labels.append('off')
                else:
                    labels.append('on')
            results = [{'prob': float(prob), 'label': label} for prob, label in zip(probabilities, labels)]
            return {"class": results}

start_time_without_metadata_fetching = time.time()

# Before map_batches
cpu_before = psutil.cpu_percent(interval=None)

# Perform map_batches operation
ds = ds.map_batches(preprocess, batch_format="numpy", num_cpus=8)

# After map_batches
cpu_after = psutil.cpu_percent(interval=None)

# Iterate over batches to access block metadata
for batch in ds.iter_batches(batch_size=None):
    # Access block metadata for each batch
    for block_metadata in batch.block_metadata():
        # Print block metadata
        print("Block Metadata:")
        print("Number of rows:", block_metadata.num_rows)
        print("Size in bytes:", block_metadata.size_bytes)
        print("Schema:", block_metadata.schema)
        print("Input files:", block_metadata.input_files)
        print("Execution stats:", block_metadata.exec_stats)
        print("-" * 50)

# Print CPU usage
print("CPU Usage Before map_batches:", cpu_before)
print("CPU Usage After map_batches:", cpu_after)

end_time = time.time()

print("Total Execution Time for preprocess:", end_time - start_time_without_metadata_fetching)

print("Total time: ", end_time-start_time)
print("Throughput (img/sec): ", (16232)/(end_time-start_time))
print("Total time w/o metadata fetching (img/sec) : ", (end_time-start_time_without_metadata_fetching))
print("Throughput w/o metadata fetching (img/sec) ", (16232)/(end_time-start_time_without_metadata_fetching))
