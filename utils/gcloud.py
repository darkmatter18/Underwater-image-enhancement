from google.cloud import storage
from google.cloud.storage import Bucket


def setup_cloud_bucket(bucket_name: str) -> Bucket:
    """Setup Google Cloud Bucket

    :param bucket_name: The Root of the Data-storage
    :return: Bucket
    """
    print(f"Using Bucket: {bucket_name} for storing and Loading Data")
    c = storage.Client()
    b = c.get_bucket(bucket_name)
    assert b.exists(), f"Bucket {bucket_name} doesn't exist. Try different one"
    return b



class GCloud:
    def __init__(self, bucket_name: str):
        """Setup Google Cloud Bucket
        :param bucket_name: The Root of the Data-storage
        :return: Bucket
        """
        print(f"Using Bucket: {bucket_name} for storing and Loading Data")
        c = storage.Client()
        b = c.get_bucket(bucket_name)
        assert b.exists(), f"Bucket {bucket_name} doesn't exist. Try different one"
        self.bucket = b

    def load_object_from_gcloud(self, cloud_full_path: str, local_full_path: str):
        cloud_relative_path = "/".join(cloud_full_path.split("/")[3:])
        self.bucket.get_blob(cloud_relative_path).download_to_filename(local_full_path)
