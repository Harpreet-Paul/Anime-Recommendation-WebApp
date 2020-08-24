# Hakken: An Anime Recommender System

## Background

Anime refers to hand-drawn and computer generated animation originating from Japan. Anime is characterized by it's unique art style, expressive characters and fantastical themes. Since it's inception in 1920, anime has grown to become internationally popular and has found it's way into popular North American streaming services such as Netflix. 

## What is Hakken?

Hakken is a machine-learning based anime recommendation system. It takes user's ratings of anime they've previously seen and uses them as a basis for generating personalized recommendations.

## Motivation

If Netflix, which already implements a machine-learning based recommendation system, hosts anime on it's platform, why build Hakken? 

The selection of anime available on Netflix is limited. Avid anime fans use Crunchyroll, a subscription based anime streaming service, or KissAnime, a free anime streaming website to access a wider array of anime. However, neither Crunchyroll, nor Kissanime serve up personalized anime recommendations to their users. Thus, the motivation behind creating Hakken was to give users automated, personalized recommendations from a much larger pool of anime.  

## How it Works

### Problem/Solution Framework

The task of generating anime recommendations can be framed as follows: 

A model or algorithm should predict the ratings that a user $u$ would give to all the items $i$ that that user has yet to see. These predicted ratings should then be sorted from highest to lowest and the unseen anime corresponding to the top N highest predicted ratings should be served as recommendations. 

### The Data

MyAnimeList.net is an anime community and database where users create profiles that feature a list of their anime ratings on a scale of 1-10. Kaggle user azathoth42 scraped together a dataset of ~45 million user ratings from public MyAnimeList profiles which was used for algorithm training and testing in this project.

### The Algorithm

Hakken uses the Item-Based KNN-with-Baseline collaborative filtering algorithm to predict the ratings that a user $u$ would give to all unseen items $i$ and generate recommendations. There are two main components to this algorithm: the baseline rating model and the item-based k-nearest-neighbours algorithm. 

#### Baseline Rating

The baseline rating model is a simple model for predicting user ratings on items that assumes that any given user's rating on an item is a function of 3 terms: the gobal average rating across all items, the user bias and the item bias. User bias can be thought of as how much above or below the global average a user tends to rate items on average and the item bias can be thought of as how much above or below the global average an item tends to get rated on average. More simply, the user bias is a term meant to capture whether or not a user is a tough critic or easy rater and the item bias is a term mean to capture whether an item is inherently above or below average in quality. 

The baseline rating models takes the following form:

$b_{ui} = \mu + b_u + b_i$

where $b_{ui}$ is the baseline rating for a user $u$ on an item $i$,  $\mu$ is the global average item rating, $b_u$ is the user bias and $b_i$ is the item bias. 

$b_u$ and $b_i$ are chosen for each user and each item so as to minimize the regularized square error cost function: 

$\sum_{r_{ui} \in R_{train}} \left(r_{ui} - (\mu + b_u + b_i)\right)^2 +
\lambda \left(b_u^2 + b_i^2 \right)$

where $r_{ui}$ is the real/known rating by a user $u$ on an item $i$, $R_{train}$ is the set of all real/known ratings on which the baseline rating model is trained and $\lambda$ is a regularization constant. 

#### Item-Based K-Nearest-Neighbours 

Item-Based K-Nearest-Neighbours is a collaborative filtering algorithm where the predicted rating for a user $u$ on an item $i$ is:

$\hat{r}_{ui} = \frac{\sum\limits_{j \in N^k_u(i)} \text{sim}(i, j) \cdot r_{uj}}{\sum\limits_{j \in N^k_u(i)} \text{sim}(i, j)}$

where ${sim}(i, j)$ is the degree of similarity between an item $i$ and another item $j$, $r_{uj}$ is the rating that user $u$ gave to the similiar item $j$ and $N^k_u(i)$ is the set of $k$ items that are most similar to show $i$ that user $u$ has rated.

The basic idea is as follows:

Compare all the ratings for an item $i$ with all the ratings for another item $j$ and then compute some measure of similarity between the two. Similarity between the ratings for the two items is used as a proxy for the degree of similarity between the items themsevles. This step is repeated for all pairs of items.

Then, find the k most similar items to the unseen item $i$ that user $u$ has also rated. The user's ratings on these similar items become the basis for predicting the rating on $i$. The predicted rating is an average of the ratings by $u$ on the k similar items $j$, weighted by how similar each of those k items are to the item $i$. The greater the similarity between $j$ and $i$, the more the rating on $j$ contributes to the predicted rating for $i$. Intuitively, if $u$ gave a low rating to a very similar item $j$, then the predicted rating on $i$ should go down. 

Item-Based KNN-with-Baseline is a modified version of the above algorithm that takes the form: 

$\hat{r}_{ui} = b_{ui} + \frac{ \sum\limits_{j \in N^k_u(i)}
\text{sim}(i, j) \cdot (r_{uj} - b_{uj})} {\sum\limits_{j \in
N^k_u(i)} \text{sim}(i, j)}$

where $b_{ui}$ is the baseline rating for a user $u$ on an item $i$ and b_{uj} is the baseline rating for the user $u$ on the similar item $j$. 

In this version of the algorithm, the similarity between items is the shrunk pearson-baseline correlation coefficient which is calculated as:

$\begin{align}\begin{aligned}\text{pearson_baseline_shrunk_sim}(i, j) &= \frac{|I_{ij}| - 1}
{|I_{ij}| - 1 + \text{shrinkage}} \cdot \hat{\rho}_{ij}\\\end{aligned}\end{align}$

where $|I_{ij}|$ is the number of instances where a user rated both items $i$ and $j$, "shrinkage" is a pre-determined shrinkage factor and $\hat{\rho}_{ij}$ is the pearson-baseline correlation coefficient, which is calculated as:





1. where i got data from and the structure of the data
2. models I used and evaluation metrics to assess model performance 
3. high level overview of the backend (API call, storing model objects)

## Future Improvements 
