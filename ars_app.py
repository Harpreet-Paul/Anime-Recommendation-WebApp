#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import flask
import make_recommendations
import create_gcs_local_file
from joblib import load, dump


# In[2]:


k = 20


# In[3]:


global_mean = 7.328450933497362   


# In[4]:


bucket_name = "anime-recommendation-system"


# In[5]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/ids_biases_array.pkl", "local_ids_biases_array.pkl")
ids_biases_array = pickle.load(open("local_ids_biases_array.pkl", "rb"))


# In[6]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/joblib_similarity_matrix", "local_joblib_similarity_matrix")
similarity_matrix = load("local_joblib_similarity_matrix")


# In[7]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/joblib_iid_rid_array", "local_joblib_iid_rid_array")
iid_rid_array = load("local_joblib_iid_rid_array")


# In[8]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/anime_list/AnimeList.csv", "local_anime_list.csv")
anime_list_df = pd.read_csv("local_anime_list.csv", usecols = [0,1])


# In[ ]:


app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])

def main():
    
    if flask.request.method == 'GET':
        return(flask.render_template('main'))
    
    if flask.request.method == 'POST':
        
        username = flask.request.form['username']
        num_of_recommendations = int(flask.request.form['number of recommendations'])
        
        recommendations = make_recommendations.make_recommendations(username, ids_biases_array, similarity_matrix, global_mean, iid_rid_array, k, num_of_recommendations, anime_list_df)
        
    return flask.render_template('main',
                                     original_input={'Username':username,
                                                     'Number of Recommendations':num_of_recommendations},
                                     result=recommendations.to_html(escape = False),
                                     )

if __name__ == '__main__':
    app.run()


# In[ ]:






