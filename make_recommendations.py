#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas as pd
import numpy as np
from heapq import nlargest


# In[ ]:


def get_user_ratings(user):
    
    user_ratings = []
    x=1
    
    while x >= 1:
        
        user_ratings_page_x = requests.get('https://api.jikan.moe/v3/user/' + user + '/animelist/all/' + str(x)).json()['anime']
        user_ratings.extend(user_ratings_page_x)
        
        if len(user_ratings_page_x) < 300:
            break
        
        x += 1
        
        time.sleep(2)
    
    return user_ratings


# In[ ]:


def get_user_ratings_df(user_ratings):
    
    user_ratings_df = pd.DataFrame([[rating['mal_id'], rating['title'], rating['score'], rating['watching_status']] for rating in user_ratings], columns = ['anime_id', 'title', 'my_score', 'my_status'])
    
    user_ratings_df = user_ratings_df[(user_ratings_df.my_status != 6)]
    user_ratings_df = user_ratings_df[((user_ratings_df.my_score != 0) | ((user_ratings_df.my_score == 0) & (user_ratings_df.my_status == 4)))]
    user_ratings_df.replace({'my_score': {0:1}}, inplace=True)
    
    return user_ratings_df


# In[ ]:


def get_user_bias(global_mean, user_ratings_df, ids_biases_array):
    
    bu = 0
    global_mean = global_mean
    n_epochs = 20
    reg = 0.02
    lr = 0.005
    
    for dummy in range(n_epochs):
        
        for row in user_ratings_df.iterrows():
            
            try:
                
                err = row[1]['my_score'] - (global_mean + bu + ids_biases_array[ids_biases_array[:, 0] == row[1]['anime_id']][0][2])
                bu += lr * (err - reg * bu)
            
            except IndexError:
                
                continue
    
    return bu


# In[ ]:


def get_baseline_prediction(rid, global_mean, bu, ids_biases_array):
    
    global_mean = global_mean 
    baseline_prediction = global_mean + bu + ids_biases_array[ids_biases_array[:, 0] == rid][0][2]
    
    return baseline_prediction


# In[ ]:


def get_unseen_anime_rids(ids_biases_array, user_ratings_df):
    
    all_anime_rids = ids_biases_array[:, 0]
    seen_anime_rids = user_ratings_df.anime_id.values
    unseen_anime_rids = np.setdiff1d(all_anime_rids, seen_anime_rids)
    
    return unseen_anime_rids


# In[ ]:


def get_seen_anime_iids(user_ratings_df, iid_rid_array):
    
    seen_anime_rids = user_ratings_df.anime_id.values
    
    mask = np.isin(iid_rid_array[:,1], seen_anime_rids)
    
    seen_anime_iids = iid_rid_array[mask][:,0]
    
    return seen_anime_iids
    


# In[ ]:


def filter_ids_biases_similarities_by_seen_anime_iids(ids_biases_array, similarity_matrix, seen_anime_iids):
    
    ids_biases_array_seen_anime = ids_biases_array[seen_anime_iids, :]
    similarity_matrix_seen_anime = similarity_matrix[seen_anime_iids, :]
    
    return ids_biases_array_seen_anime, similarity_matrix_seen_anime


# In[ ]:


def get_top_k_seen_similar_anime(ids_biases_array_seen_anime, similarity_matrix_seen_anime, rid, iid_rid_array, user_ratings_df, k):
    
    iid = iid_rid_array[iid_rid_array[:,1] == rid][0,0]
    mask = similarity_matrix_seen_anime[:, iid] > 0

    similarity_matrix_seen_similar_anime =  similarity_matrix_seen_anime[mask]
    ids_biases_array_seen_similar_anime = ids_biases_array_seen_anime[mask]
    rid_similarity_seen_similar_anime = np.column_stack((ids_biases_array_seen_similar_anime[:, 0], similarity_matrix_seen_similar_anime[:, iid]))
   
    top_k_seen_similar_anime = pd.DataFrame(nlargest(k, rid_similarity_seen_similar_anime, key = lambda x: x[1]), columns = ['anime_id', 'similarity'])
    top_k_seen_similar_anime =  top_k_seen_similar_anime.merge(user_ratings_df[['anime_id', 'my_score']], on = 'anime_id')
    
    return top_k_seen_similar_anime


# In[ ]:


def get_rating_prediction(top_k_seen_similar_anime, baseline_prediction, top_k_seen_similar_anime_baseline_ratings):
    
    if len(top_k_seen_similar_anime) == 0:
        
        predicted_rating = baseline_prediction
            
    else:
        
        top_k_seen_similar_anime_residuals = top_k_seen_similar_anime.my_score.values - top_k_seen_similar_anime_baseline_ratings
            
        sum_of_similarity_residual_products = np.dot(top_k_seen_similar_anime_residuals, top_k_seen_similar_anime.similarity.values)
            
        sum_of_similarities = np.sum(top_k_seen_similar_anime.similarity.values)
            
        predicted_rating = baseline_prediction + (sum_of_similarity_residual_products/sum_of_similarities)
        
    return predicted_rating
        
        
        


# In[ ]:


def get_unseen_anime_predicted_ratings(ids_biases_array, user_ratings_df, global_mean, iid_rid_array, similarity_matrix, k):
    
    unseen_anime_rids = get_unseen_anime_rids(ids_biases_array, user_ratings_df)
    
    bu = get_user_bias(global_mean, user_ratings_df, ids_biases_array)
    
    seen_anime_iids = get_seen_anime_iids(user_ratings_df, iid_rid_array)
    
    ids_biases_array_seen_anime, similarity_matrix_seen_anime = filter_ids_biases_similarities_by_seen_anime_iids(ids_biases_array, similarity_matrix, seen_anime_iids)
    
    predicted_ratings_unseen_anime = []
    
    for unseen_anime_rid in unseen_anime_rids:
        
        baseline_prediction = get_baseline_prediction(unseen_anime_rid, global_mean, bu, ids_biases_array)
        
        top_k_seen_similar_anime = get_top_k_seen_similar_anime(ids_biases_array_seen_anime, similarity_matrix_seen_anime, unseen_anime_rid, iid_rid_array, user_ratings_df, k)
        
        top_k_seen_similar_anime_baseline_ratings = top_k_seen_similar_anime.anime_id.apply(get_baseline_prediction, args = (global_mean, bu, ids_biases_array))
        
        rating_prediction = get_rating_prediction(top_k_seen_similar_anime, baseline_prediction, top_k_seen_similar_anime_baseline_ratings)
        
        predicted_ratings_unseen_anime.append((unseen_anime_rid, rating_prediction))
    
    return predicted_ratings_unseen_anime


# In[ ]:


def get_top_n_recommendations(predicted_ratings_unseen_anime, num_of_recommendations):
    
    top_n_recommendations = nlargest(num_of_recommendations, predicted_ratings_unseen_anime, key = lambda x: x[1])
    
    return top_n_recommendations 


# In[ ]:


def get_recommendation_titles(top_n_recommendations, anime_list_df):
    
    recommendations_df = pd.DataFrame(top_n_recommendations, columns = ['anime_id', 'predicted rating'])
    recommendation_titles = recommendations_df.merge(anime_list_df, on='anime_id')
    recommendation_titles = recommendation_titles[['anime_id','title', 'predicted rating']]
    
    recommendation_titles['title'] = recommendation_titles.apply(lambda x: '<a href="https://myanimelist.net/anime/' + str(x['anime_id']) + '/">' + str(x['title']) + '</a>'.format(x['title']), axis = 1)
    
    return recommendation_titles


# In[ ]:


def make_recommendations(user, ids_biases_array, similarity_matrix, global_mean, iid_rid_array, k, num_of_recommendations, anime_list_df):
    
    user_ratings = get_user_ratings(user)
    
    user_ratings_df = get_user_ratings_df(user_ratings)
    
    predicted_ratings_unseen_anime = get_unseen_anime_predicted_ratings(ids_biases_array, user_ratings_df, global_mean, iid_rid_array, similarity_matrix, k)
    
    top_n_recommendations = get_top_n_recommendations(predicted_ratings_unseen_anime, num_of_recommendations)
    
    recommendation_titles = get_recommendation_titles(top_n_recommendations, anime_list_df)
    
    return recommendation_titles



