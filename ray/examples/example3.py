from typing import Dict
import numpy as np
import torch
import torch.nn as nn

import ray

#Create a Ray Dataset from in-memory Numpy arrays.
ds = ray.data.from_numpy(np.ones((1, 100)))

# Define a Predictor class for inference.
class TorchPredictor:
    def __init__(self):
        # Load a dummy neural network.
        # Set `self.model` to the pre-trained PyTorch model.
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

# Map the Predictor over the Dataset to get predictions. Use 2 parallel actors for inference. Each actor predicts on a different partition of data.
predictions = ds.map_batches(TorchPredictor, concurrency=2)
# Show one prediction output.
predictions.show(limit=1)