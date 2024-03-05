from typing import Dict
import numpy as np
import torch
import torch.nn as nn

import ray

# Step 1: Create a Ray Dataset from in-memory Numpy arrays.
# You can also create a Ray Dataset from many other sources and file
# formats.
ds = ray.data.from_numpy(np.ones((1, 100)))

# Step 2: Define a Predictor class for inference.
# Use a class to initialize the model just once in `__init__`
# and re-use it for inference across multiple batches.
class TorchPredictor:
    def __init__(self):
        # Load a dummy neural network.
        # Set `self.model` to your pre-trained PyTorch model.
        self.model = nn.Sequential(
            nn.Linear(in_features=100, out_features=1),
            nn.Sigmoid(),
        )
        self.model.eval()

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        tensor = torch.as_tensor(batch["data"], dtype=torch.float32)
        with torch.inference_mode():
            # Get the predictions from the input batch.
            return {"output": self.model(tensor).numpy()}

# Step 2: Map the Predictor over the Dataset to get predictions.
# Use 2 parallel actors for inference. Each actor predicts on a
# different partition of data.
predictions = ds.map_batches(TorchPredictor, concurrency=2)
# Step 3: Show one prediction output.
predictions.show(limit=1)