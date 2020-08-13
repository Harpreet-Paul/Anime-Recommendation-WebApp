#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from joblib import load
import pickle
import pandas as pd
import flask
import make_recommendations
import create_gcs_local_file


# In[ ]:


bucket_name = "anime-recommendation-system"


# In[ ]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/ids_biases_array.pkl", "local_ids_biases_array.pkl")
ids_biases_array = pickle.load(open("local_ids_biases_array.pkl", "rb"))


# In[ ]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/joblib_prediction_algo", "local_joblib_prediction_algo")
prediction_algo = load("local_joblib_prediction_algo")


# In[ ]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/joblib_similarity_matrix", "local_joblib_similarity_matrix")
similarity_matrix = load("local_joblib_similarity_matrix")


# In[ ]:


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
        
        recommendations = make_recommendations.make_recommendations(username, ids_biases_array, similarity_matrix, prediction_algo, num_of_recommendations, anime_list_df)
        
    return flask.render_template('main',
                                     original_input={'Username':username,
                                                     'Number of Recommendations':num_of_recommendations},
                                     result=recommendations.to_html(escape = False),
                                     )

if __name__ == '__main__':
    app.run()

