#!/usr/bin/env python
# coding: utf-8

############## This module defines all the logic needed to take a user's MAL username and desired number of recommendations as input and generate a dataframe of recommendations. This module culminates in the 'make_recommendations' function which is the main function called within the Flask app. ######################################################################################


import requests
import pandas as pd
import numpy as np
from heapq import nlargest


## Retrieve a user's anime list which contains their anime ratings from their profile at MyAnimeList.net and store these ratings in a list. 

def get_user_ratings(user):
    
    user_ratings = []
    x=1
    
    ## Only one page of the user's anime list can be retrieved at a time. x represents page numbers, starting from 1, and API calls are made to the corresponding page (based on the current value of x) of the user's anime list endpoint until all pages have been retrieved. 
    
    while x >= 1:
        
        ## Make an API call to a particular page of the user's anime list endpoint.
        
        user_ratings_page_x = requests.get('https://api.jikan.moe/v3/user/' + user + '/animelist/all/' + str(x)).json()['anime']
        user_ratings.extend(user_ratings_page_x)
        
        ## The 'while' loop that keeps the API calls to particular pages keeps running until a page with less than 300 entries is retrieved as that marks the last page of the user's anime list. 
        
        if len(user_ratings_page_x) < 300:
            break
        
        x += 1
        
        ## A 2 second delay is integrated into the loop to space out the API calls so as to not get rate-limited. 
        
        time.sleep(2)
    
    return user_ratings


## Constructing a dataframe out of the list of user ratings.

def get_user_ratings_df(user_ratings):
    
    user_ratings_df = pd.DataFrame([[rating['mal_id'], rating['title'], rating['score'], rating['watching_status']] for rating in user_ratings], columns = ['anime_id', 'title', 'my_score', 'my_status'])
    
    ## Rows (anime rating entries) with a 'my_status' value of 6 are dropped because these represent anime the user has yet to watch. 
    
    user_ratings_df = user_ratings_df[(user_ratings_df.my_status != 6)]
    
    ## Only rows with non-zero anime ratings are kept as zero ratings represent unscored anime by the user. The only exceptions are entries with a zero rating and a 'my_status' value of 4 as these are anime that were dropped by the user and hence assigning them a zero rating is reasonable. 
    
    user_ratings_df = user_ratings_df[((user_ratings_df.my_score != 0) | ((user_ratings_df.my_score == 0) & (user_ratings_df.my_status == 4)))]
    
    ## All the zero ratings (ones assigned to anime that were dropped by the user) are converted to ratings of 1 since this is the lowest score a MyAnimeList.net user is allowed to assign as an anime rating. 
    
    user_ratings_df.replace({'my_score': {0:1}}, inplace=True)
    
    return user_ratings_df


## Calculates user bias (deviation from the global average rating that accounts for whether the user is a tough critic or easy rater) for a given user.

def get_user_bias(global_mean, user_ratings_df, ids_biases_array):
    
    ## Initializing value of user bias to 0
    
    bu = 0
    
    ## global average anime rating 
    
    global_mean = global_mean
    
    ## number of iterations for gradient descent procedure
    
    n_epochs = 20
    
    ## regularization constant 
    
    reg = 0.02
    
    ## learning rate
    
    lr = 0.005
    
    for dummy in range(n_epochs):
        
        for row in user_ratings_df.iterrows():
            
            try:
                
                ## Error is calculated by taking a user's raw rating for an anime and subtracting from it the global mean, current value of the user bias and the corresponding item bias (deviation from the global average rating that accounts for whether an anime is inherently above or below average in quality)for that anime from the ids_biases_array.
                
                err = row[1]['my_score'] - (global_mean + bu + ids_biases_array[ids_biases_array[:, 0] == row[1]['anime_id']][0][2])
                
                ## The value of the user bias is updated based on the learning rate, regularization constant and the error value of the current iteration. 
                
                bu += lr * (err - reg * bu)
            
            ## A try-except block is implemented so that user ratings for anime not in the training set during algorithm generation are skipped over. Similarity measures for these anime have not been calculated so they cannot be used as a basis for generating recommendations. 
            
            except IndexError:
                
                continue
    
    return bu


## Calculates a baseline rating prediction for a given user on a given anime.

def get_baseline_prediction(rid, global_mean, bu, ids_biases_array):
    
    global_mean = global_mean 
    
    ## The baseline prediction is the sum of the global average rating, the user bias and the item/show bias.
    
    baseline_prediction = global_mean + bu + ids_biases_array[ids_biases_array[:, 0] == rid][0][2]
    baseline_prediction = round(baseline_prediction, 0)
    
    return baseline_prediction


## Function that returns an array of all the rids (raw ids) corresponding to anime that the user has yet to rate.

def get_unseen_anime_rids(ids_biases_array, user_ratings_df):
    
    all_anime_rids = ids_biases_array[:, 0]
    seen_anime_rids = user_ratings_df.anime_id.values
    
    ## 'np.setdiff1d' takes the set difference between the all_anime_rids array and the seen_anime_rids array.
    
    unseen_anime_rids = np.setdiff1d(all_anime_rids, seen_anime_rids)
    
    return unseen_anime_rids


## Function that returns an array of the iids (inner ids) corresponding to all the anime the user has already rated.

def get_seen_anime_iids(user_ratings_df, iid_rid_array):
    
    seen_anime_rids = user_ratings_df.anime_id.values
    
    ## Creating a boolean mask where each index of iid_rid_array is 'True' if the rid at that index is found in seen_anime_rids.
    
    mask = np.isin(iid_rid_array[:,1], seen_anime_rids)
    
    ## Applying the boolean mask to the iid_rid_array so that the remaining subset is only the iids of those anime which the user has rated.
    
    seen_anime_iids = iid_rid_array[mask][:,0]
    
    return seen_anime_iids
    

## Function that uses the array of seen_anime_iids to perform an index masking of the ids_biases_array and similarity_matrix. Returns the ids_biases_array and similarity_matrix subsetted such that they only contain ids and biases and similarity measures for anime the user has rated.

def filter_ids_biases_similarities_by_seen_anime_iids(ids_biases_array, similarity_matrix, seen_anime_iids):
    
    ids_biases_array_seen_anime = ids_biases_array[seen_anime_iids, :]
    similarity_matrix_seen_anime = similarity_matrix[seen_anime_iids, :]
    
    return ids_biases_array_seen_anime, similarity_matrix_seen_anime


## Function that returns a dataframe of the top K most similar anime that a user has rated to an anime for which a rating prediction is being made.

def get_top_k_seen_similar_anime(ids_biases_array_seen_anime, similarity_matrix_seen_anime, rid, iid_rid_array, user_ratings_df, k):
    
    ## Getting the inner id of the anime for which a rating prediction is being made.
    
    iid = iid_rid_array[iid_rid_array[:,1] == rid][0,0]
    
    ## Creating a boolean mask where each index of the similarity matrix of anime the user has rated is 'True' if the anime at that index has a similarity of greater than 0 with the anime for which a rating prediction is being made.
    
    mask = similarity_matrix_seen_anime[:, iid] > 0

    similarity_matrix_seen_similar_anime =  similarity_matrix_seen_anime[mask]
    ids_biases_array_seen_similar_anime = ids_biases_array_seen_anime[mask]
    
    ## Creating a 2d-array of the raw ids and similarity measures corresponding to all anime the user has rated that have a greater than 0 similarity with the anime for which a rating prediction is being made.
    
    rid_similarity_seen_similar_anime = np.column_stack((ids_biases_array_seen_similar_anime[:, 0], similarity_matrix_seen_similar_anime[:, iid]))
    
    ## The top K anime with the highest similarity to the anime for which a rating prediction is being made are sorted into a dataframe.
   
    top_k_seen_similar_anime = pd.DataFrame(nlargest(k, rid_similarity_seen_similar_anime, key = lambda x: x[1]), columns = ['anime_id', 'similarity'])
    
    ## User raw ratings for the top K most similar anime are added to the dataframe generated in the last step.
    
    top_k_seen_similar_anime =  top_k_seen_similar_anime.merge(user_ratings_df[['anime_id', 'my_score']], on = 'anime_id')
    
    return top_k_seen_similar_anime


##  Function that returns a predicted rating for a given anime that the user has not yet seen.

def get_rating_prediction(top_k_seen_similar_anime, baseline_prediction, top_k_seen_similar_anime_baseline_ratings):
    
     ## If the user has not rated any anime with a positive similarity to the anime for which a rating is being predicted, then the rating prediction is simply the baseline rating prediction for that anime.
    
    if len(top_k_seen_similar_anime) == 0:
        
        predicted_rating = baseline_prediction
            
    else:
        
        ## Subtracting the baseline rating from the raw rating for each of the K nearest neighbours to the anime for which a rating prediction is being made. The differences are called the residuals. 
        
        top_k_seen_similar_anime_residuals = top_k_seen_similar_anime.my_score.values - top_k_seen_similar_anime_baseline_ratings
            
        ## Taking the dot product between the residuals from the previous step and the corresponding similarities between a nearest neighbour and the anime for which a rating prediction is being made.     
            
        sum_of_similarity_residual_products = np.dot(top_k_seen_similar_anime_residuals, top_k_seen_similar_anime.similarity.values)
            
        ## Getting the sum of similarities for each of the K nearest neighbours.     
            
        sum_of_similarities = np.sum(top_k_seen_similar_anime.similarity.values)
        
            
        predicted_rating = baseline_prediction + (sum_of_similarity_residual_products/sum_of_similarities)
        
    return predicted_rating
        
        
## Function that iterates through each anime the user has not rated and generates a predicted rating for those anime. This function relies on all the smaller helper functions defined thus far. 
 
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


## Function which returns the top N highest predicted ratings (top N best recommendations) for anime the user has not yet seen based on the number of recommendations desired by the user. 

def get_top_n_recommendations(predicted_ratings_unseen_anime, num_of_recommendations):
    
    top_n_recommendations = nlargest(num_of_recommendations, predicted_ratings_unseen_anime, key = lambda x: x[1])
    
    return top_n_recommendations 


## This function stores the top N recommendations in a dataframe.

def get_recommendation_titles(top_n_recommendations, anime_list_df):
    
    recommendations_df = pd.DataFrame(top_n_recommendations, columns = ['anime_id', 'predicted rating'])
    
    ## 'recommendations_df' is merged with the 'anime_list_df' to add a column, 'title', that stores the names of the anime corresponding to the iids of the top N recommendations. 
    
    recommendation_titles = recommendations_df.merge(anime_list_df, on='anime_id')
    recommendation_titles = recommendation_titles[['anime_id', 'title']]
    
    ## Converting each anime title to a hyperlink that links to the corresponding MyAnimeList.net information page for that anime. 
    
    recommendation_titles['title'] = recommendation_titles.apply(lambda x: '<a href="https://myanimelist.net/anime/' + str(x['anime_id']) + '/">' + str(x['title']) + '</a>'.format(x['title']), axis = 1)
    
    recommendation_titles = recommendation_titles[['title']]
    recommendation_titles.rename(columns={"title": "Anime Title"}, inplace = True)
    
    ## Adding a column that tells the recommendation rank from 1 to N = number of recommendations desired for each anime. 
    
    recommendation_titles['Rank'] = np.arange(start=1, stop=(len(recommendation_titles.index) + 1))
    recommendation_titles = recommendation_titles[['Rank', 'Anime Title']]
    
    return recommendation_titles


## This is the main function of the program within which all of the smaller functions previously defined are wrapped. Returns a dataframe of the top N recommended anime. 

def make_recommendations(user, ids_biases_array, similarity_matrix, global_mean, iid_rid_array, k, num_of_recommendations, anime_list_df):
    
    user_ratings = get_user_ratings(user)
    
    user_ratings_df = get_user_ratings_df(user_ratings)
    
    predicted_ratings_unseen_anime = get_unseen_anime_predicted_ratings(ids_biases_array, user_ratings_df, global_mean, iid_rid_array, similarity_matrix, k)
    
    top_n_recommendations = get_top_n_recommendations(predicted_ratings_unseen_anime, num_of_recommendations)
    
    recommendation_titles = get_recommendation_titles(top_n_recommendations, anime_list_df)
    
    return recommendation_titles

