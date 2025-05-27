import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BASE_PATH = os.getenv("S3_BASE_PATH", "uploads")

def upload_file_to_s3(local_path: str, s3_key: str) -> str:
    s3_path = f"{S3_BASE_PATH}/{s3_key}"
    s3.upload_file(local_path, BUCKET_NAME, s3_path)
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_path}"