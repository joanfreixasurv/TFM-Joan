import ray.data as ray_data
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from torchvision import transforms
import io
import torch
import ray

# MinIO configuration
minio_endpoint = 'localhost:9000'
access_key = 'minioadmin'
secret_key = 'minioadmin'
bucket_name = 'images'

# Ray initialization
ray.init()

def preprocess_image(image_data):
    # Convert image_data to a PIL Image
    img = Image.open(io.BytesIO(image_data))
    
    # Apply the required preprocessing steps
    transform = transforms.Compose([
        transforms.ToPILImage(),
    ])
    
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor

def main():
    # Define the image data set using read_images
    image_dataset = ray.data.read_images(f"{bucket_name}/*")


    # Use concurrent processing to preprocess images in parallel
    with ProcessPoolExecutor() as executor:
        input_tensors = list(executor.map(preprocess_image, image_dataset))

    # Convert the list of input tensors to a torch tensor
    input_batch = torch.cat(input_tensors, dim=0)

    print("Preprocessed images shape:", input_batch.shape)

if __name__ == "__main__":
    main()

