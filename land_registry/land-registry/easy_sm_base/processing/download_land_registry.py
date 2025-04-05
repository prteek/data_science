#%%

import boto3
import requests

client = boto3.client('s3')

#%%

def main(bucket, key, chunk_size_in_MB, url):
    """ This is function collects a series of separate functions initiating the upload.
    Downloads the file at the url provided in chunks, and uploads to <key> in <bucket>.
    Parameters
    ----------
    bucket: string
        Bucket name.
    key: string
        File name. Subdirectories within buckets are generated by prepending 'subdir_name/' to
        the file name.
    chunk_size_in_MB: int
        Size of megabyte chunks to stream the file and upload it in.
    url: string
        The url for the file download.
    Returns
    -------
    location: string
        Location of uploaded file
    """
    upload_id = create_multipart_upload(bucket, key)
    parts = download_and_upload(url, upload_id, key, bucket, chunk_size_in_MB)
    location = complete_multipart_upload(key, bucket, upload_id, parts)

    return location


def create_multipart_upload(bucket, key):
    """ This is function starts the multipart upload to the bucket with the relevant key.
    Parameters
    ----------
    bucket: string
        Bucket name.
    key: string
        File name. Subdirectories within buckets are generated by prepending 'subdir_name/' to
        the file name.
    Returns
    -------
    upload_id: string
        ID for the initiated multipart upload.
    """
    # For more information about the multipart upload response, see
    # https://docs.aws.amazon.com/AmazonS3/latest/API/API_CreateMultipartUpload.html
    response = client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
    )
    upload_id = response['UploadId']
    return upload_id


def download_and_upload(url, upload_id, key, bucket, chunk_size_in_MB):
    """ This is function initiates the download of the data and subsequent upload to S3.
    The data is streamed from the source in discrete chunks. Each of these chunks is uploaded
    using client.upload_part(...) which works in conjunction with client.create_multipart_upload(...).
    Parameters
    ----------
    url: string
        The url for the file download.
    upload_id: string
        ID for the initiated multipart upload.
    key: string
        File name. Subdirectories within buckets are generated by prepending 'subdir_name/' to
        the file name.
    bucket: string
        Bucket name.
    chunk_size_in_MB: int
        Size of megabyte chunks to stream the file and upload it in.
    Returns
    -------
    parts: list
        List containing the part number and associated entity tag for each uploaded object.
    """
    parts = []
    # The download URL is accessed and data is streamed from there.
    with requests.get(url, stream=True) as r:
        # Download & upload chunks
        for part_number, chunk in enumerate(r.iter_content(chunk_size=chunk_size_in_MB * 1024 * 1024)):
            response = client.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number + 1,
                Body=chunk,
            )
            parts.append({
                "ETag": response['ETag'],
                "PartNumber": part_number + 1,
            })

    return parts


def complete_multipart_upload(key, bucket, upload_id, parts):
    """ This is function completes the multipart upload.
    Parameters
    ----------
    key: string
        File name. Subdirectories within buckets are generated by prepending 'subdir_name/' to
        the file name.
    bucket: string
        Bucket name.
    upload_id: string
        ID for the initiated multipart upload.
    parts: list
        List containing the part number and associated entity tag for each uploaded object.
    Returns
    -------
    location: string
        Location of uploaded file
    """
    # complete multipart upload
    print(f"Completed uploaded, closing multipart")
    response = client.complete_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )
    location = response['Location']
    eTag = response['ETag']

    return location

#%%

if __name__ == '__main__':
    #%%
    # lr_url: link for the land registry data complete dataset download.
    lr_url = 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv'
    # chunk_size_in_MB: size of chunks to download in (minimum 5MB, or completion will fail)
    chunk_size_in_MB = 5

    key = 'raw_land_registry_transactions/' + lr_url.split('/')[-1]

    bucket = 'ds-dev-bkt'
    location = main(
        bucket=bucket,
        key=key,
        chunk_size_in_MB=chunk_size_in_MB,
        url=lr_url
    )
