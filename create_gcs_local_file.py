#!/usr/bin/env python
# coding: utf-8

########## This module contains the 'create_gcs_local_file' that is called in ars_app.py to load the necessary objects from google cloud storage to generate recommendations. These objects are the similarity matrix, array of anime iids and rids, array of show biases and the anime list dataframe. #############################################################################################


from google.cloud import storage

def create_gcs_local_file(bucket_name, bucket_path, local_file_name):
    
    bucket_name = bucket_name
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(bucket_path)
    blob.download_to_filename(local_file_name)   

