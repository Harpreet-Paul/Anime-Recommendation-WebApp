#!/usr/bin/env python
# coding: utf-8

# ################# This is where the Flask web app is defined. This .py file will be run on the google cloud virtual machine. ############################################################################################################

# In[ ]:


import pickle
import pandas as pd
import flask
import make_recommendations
import create_gcs_local_file
from joblib import load, dump


# In[ ]:


k = 20


# In[ ]:


global_mean = 7.328450933497362   


# In[ ]:


bucket_name = "anime-recommendation-system"


# In[ ]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/ids_biases_array.pkl", "local_ids_biases_array.pkl")
ids_biases_array = pickle.load(open("local_ids_biases_array.pkl", "rb"))


# In[ ]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/joblib_similarity_matrix", "local_joblib_similarity_matrix")
similarity_matrix = load("local_joblib_similarity_matrix")


# In[ ]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/pkl-objects/joblib_iid_rid_array", "local_joblib_iid_rid_array")
iid_rid_array = load("local_joblib_iid_rid_array")


# In[ ]:


create_gcs_local_file.create_gcs_local_file(bucket_name, "saved-objects/anime_list/AnimeList.csv", "local_anime_list.csv")
anime_list_df = pd.read_csv("local_anime_list.csv", usecols = [0,1])


# In[ ]:


## Instantiating a flask object with the 'template_folder' parameter defining where to retrieve the HTML templates that will be rendered at the web endpoint. 

app = flask.Flask(__name__, template_folder='templates')

## Defining where the app will be hosted and with which HTML methods it can be accessed. 

@app.route('/', methods=['GET', 'POST'])

## Defining the function that will be triggered at the web endpoint. 

def main():
    
    ## Rendering an HTML template with HTML forms to input MAL username and desired number of recommendations when the user is calling the webpage (thus making a 'get' request). 
    
    if flask.request.method == 'GET':
        return(flask.render_template('ars_app_home_page'))
    
    ## When the user enters their MAL username and desired number of recommendations into the HTML form, these parameters are passed to the 'make_recommendations' function from the 'make_recommendations' module in order to generate the dataframe of anime recommendations. 
    
    if flask.request.method == 'POST':
        
        username = flask.request.form['username']
        num_of_recommendations = int(flask.request.form['number of recommendations'])
        
        recommendations = make_recommendations.make_recommendations(username, ids_biases_array, similarity_matrix, global_mean, iid_rid_array, k, num_of_recommendations, anime_list_df)
        
        ## The recommendations dataframe is converted into HTML code and passed into the HTML template to be rendered to display the recommendation results. 
        
    return flask.render_template('ars_app_recommendations_page', result=recommendations.to_html(escape = False, index=False, border=0),)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

