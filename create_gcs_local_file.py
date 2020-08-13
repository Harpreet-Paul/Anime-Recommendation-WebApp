#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import storage


# In[ ]:


def create_gcs_local_file(bucket_name, bucket_path, local_file_name):
    
    bucket_name = bucket_name
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(bucket_path)
    blob.download_to_filename(local_file_name)   

