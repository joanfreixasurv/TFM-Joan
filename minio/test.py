import ray
import s3fs

# Replace these with your Minio configuration
minio_endpoint = 'https://localhost:9000'
minio_access_key = 'minioadmin'
minio_secret_key = 'minioadmin'
minio_bucket = 'images'


# Construct Minio URL
minio_path = s3fs.S3FileSystem(anon=False, key='minioadmin', secret= 'minioadmin', client_kwargs={'endpoint_url': 'http://localhost:9000'})


# Read images from Minio bucket
ds = ray.data.read_images("s3://images", filesystem=minio_path)
ds.schema()
