from minio import Minio
from minio.error import S3Error
import os

minio_endpoint = 'localhost:9000'
access_key = 'minioadmin'
secret_key = 'minioadmin'

# Initialize MinIO client
minio_client = Minio(minio_endpoint, access_key=access_key, secret_key=secret_key, secure=False)

bucket_name = 'imagesfinal'

local_image_directory = '/home/joan/TFM/EMBLModelExample/datasets_images/2024-01-08_11h01m57s'


try:
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    # Loop through the local image directory and upload each image to the bucket
    for filename in os.listdir(local_image_directory):
        local_file_path = os.path.join(local_image_directory, filename)
        object_name = filename

        # Upload the file to the bucket
        minio_client.fput_object(bucket_name, object_name, local_file_path)

        print(f"Uploaded {filename} to {bucket_name}/{object_name}")

except S3Error as err:
    print(f"Error: {err}")
