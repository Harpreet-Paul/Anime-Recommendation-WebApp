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

Hakken uses the **Item-Based KNN-with-Baseline** collaborative filtering algorithm to predict the ratings that a user $u$ would give to all unseen items $i$ and generate recommendations. To explain how this algorithm works, we first introduce the baseline rating model and the item-based k-nearest-neighbours algorithm and then show how the Item-Based KNN-with-Baseline algorithm is synthesized from these two approaches. 

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

where $r_{uj}$ is the rating that user $u$ gave to the similiar item $j$, $N^k_u(i)$ is the set of $k$ items that are most similar to show $i$ that user $u$ has rated and ${sim}(i, j)$ is the degree of similarity between an item $i$ and another item $j$ and is computed using the pearson correlation coefficient:

$\text{pearson_sim}(i, j) = \frac{ \sum\limits_{u \in U_{ij}}
(r_{ui} -  \mu_i) \cdot (r_{uj} - \mu_{j})} {\sqrt{\sum\limits_{u
\in U_{ij}} (r_{ui} -  \mu_i)^2} \cdot \sqrt{\sum\limits_{u \in
U_{ij}} (r_{uj} -  \mu_{j})^2} }$

where $\mu_i$ and $\mu_{j}$ are the average ratings for items $i$ and $j$ respectively and $U_{ij}$ is the set of all users that have rated both items $i$ and $j$. 

The algorithm works as follows:

Compute the pearson correlation coefficient as a measure of similarity between the set of ratings for item $i$ and the set of ratings for item $j$, only taking into account instances where a user rated both items. Similarity between the ratings for the two items is used as a proxy for the degree of similarity between the items themsevles. This step is repeated for all pairs of items.

Then, find the k most similar items to the unseen item $i$ that user $u$ has also rated. The user's ratings on these similar items become the basis for predicting the rating on $i$. The predicted rating is an average of the ratings by $u$ on the k similar items $j$, weighted by how similar each of those k items are to the item $i$. The greater the similarity between $j$ and $i$, the more the rating on $j$ contributes to the predicted rating for $i$. Intuitively, if $u$ gave a low rating to a very similar item $j$, then the predicted rating on $i$ should go down. 

#### Item-Based KNN-with-Baseline

Collaborative-filtering methods try to capture the effects of user-item interactions, ie. user preferences for certain kinds of items over others, on user ratings, but such effects are small compared to the influence of item and user biases. In other words, the majority of a user's rating on an item is determined by how inherently tough of a rater that user is and how inherently above or below average in quality an item is, independent of a user's preference or aversion for that kind of item. It is only the extent to which a user's raw rating differs from the predicted baseline rating -- which encapsulates the effects of user and item bias on predicted rating -- that captures how much of a preference or aversion a user has to a particular kind of item. 

For example, if a user rates the anime "Death Note" a 9/10, but "Death Note" has a global average rating of 9/10, meaning it's universally liked, and the user's average rating is a 9/10, meaning the user gives high ratings easily, it wouldn't make sense to say this user specifically has a preference for "Death Note"-like shows -- cerebal, philisophical, thrilling. Now, if a user gave "Steins Gate" a 9/10, when "Steins Gate" only has a global average rating of 7/10 and the user's average rating is a 7/10, then the fact that the user gave "Stein's Gate" a rating above and beyond that which would be predicted from the user and item bias alone is indicative of a preference for "Steins Gate"-like shows -- romantic and sci-fi based.  

The goal then, is to feed into the KNN algorithm only the isolated part of the signal that truly represents user preference/aversion for certain items. To that end, the Item-Based KNN-with-Baseline algorithm adjusts the rating data by centering each rating by a user $u$ on an item $i$, $r_{ui}$, on the baseline rating for that user-item pair, $b_{ui}$. The residuals, $r_{ui} -  b_{ui}$, are then operated on almost identically to how the raw ratings are operated on in the standard Item-Based KNN algorithm. 

The output of this procedure is represented by the following term in the final rating prediction equation:

$\frac{ \sum\limits_{j \in N^k_u(i)}
\text{sim}(i, j) \cdot (r_{uj} - b_{uj})} {\sum\limits_{j \in
N^k_u(i)} \text{sim}(i, j)}$

where $b_{uj}$ is the baseline rating for the user $u$ on the similar item $j$. 

The difference is that item similarity is determined by the shrunken pearson correlation coefficient in order to prevent the item similarites from overfitting the data. 

The shrunk pearson-baseline correlation coefficient is calculated as:

$\begin{align}\begin{aligned}\text{pearson_baseline_shrunk_sim}(i, j) &= \frac{|U_{ij}| - 1}
{|U_{ij}| - 1 + \text{shrinkage}} \cdot \hat{\rho}_{ij}\\\end{aligned}\end{align}$

where $|U_{ij}|$ is the number of instances where a user rated both items $i$ and $j$, "shrinkage" is a pre-determined shrinkage factor and $\hat{\rho}_{ij}$ is the pearson-baseline correlation coefficient, which is calculated as:

$\text{pearson_baseline_sim}(i, j) = \hat{\rho}_{ij} = \frac{
    \sum\limits_{u \in U_{ij}} (r_{ui} -  b_{ui}) \cdot (r_{uj} -
    b_{uj})} {\sqrt{\sum\limits_{u \in U_{ij}} (r_{ui} -  b_{ui})^2}
    \cdot \sqrt{\sum\limits_{u \in U_{ij}} (r_{uj} -  b_{uj})^2}}$

The extent of the shrinkage becomes greater when $|U_{ij}|$ is smaller, which is desirable because we are less confident in the computed similarity between items $i$ and $j$ mapping onto reality when only a few users have rated both items. 

Because the $\frac{ \sum\limits_{j \in N^k_u(i)}
\text{sim}(i, j) \cdot (r_{uj} - b_{uj})} {\sum\limits_{j \in
N^k_u(i)} \text{sim}(i, j)}$ term represents only the component of the predicted rating determined by user-item interactions, it must be added to the baseline rating, the component of the predicted rating dictated user and item bias effects, in order to synthesize the full predicted rating, $\hat{r}_{ui}$.  

Putting everything together, the predicted rating takes the form: 

$\hat{r}_{ui} = b_{ui} + \frac{ \sum\limits_{j \in N^k_u(i)}
\text{sim}(i, j) \cdot (r_{uj} - b_{uj})} {\sum\limits_{j \in
N^k_u(i)} \text{sim}(i, j)}$

$k$, the number of nearest neighbours, and "shrinkage", the pearson correlation coefficient shrinkage factor, are hyperparameters that are fine-tuned through a 3-fold RandomizedSearchCV procedure. 

### Making Real-Time Recommendations

When a user interfaces with Hakken, they are asked to submit their MyAnimeList username. Using the Jikan Unofficial MyAnimeList API, an API call for the user's anime list is made to the anime list endpoint for that user on MyAnimeList.net. 

The retrieved anime list is processed and then fed into the Item-Based KNN-with-Baseline algorithm. A rating prediction is made for each unseen anime based on how the user rated similar anime in their anime list. These predicted ratings are then sorted from highest to lowest and the unseen anime corresponding to the top N (# of recommendations requested) highest predicted ratings are served as recommendations. 

The Item-Based KNN-with-Baseline algorithm is able to generate recommendations with a ~2 minute wait time. It can generate recommendations at this speed (slow as it may seem to the consumer!) because the item-item similarities, which make up the biggest chunk of the computation time, and the item biases, are pre-computed and stored in numpy arrays which are loaded into the program at start. The user bias computation and nearest neighbours aggregation are performed on the spot because they obviously cannot be pre-computed. 

## Shortcomings and Future Improvements 

There are 3 main levels at which Hakken could be improved: recommendations, UI and efficiency.

* **Recommendations**
    * Implicit Feedback
        * Item ratings convey explicit feedback on user preferences, but there is also implicit feedback in ratings data which can be harnessed to make recommendations. The implicit feedback is the knowledge of which items the users chose to rate in the first place, regardless of they actually rated (high/low) them. This is because users don't select anime to watch at random; they pick ones they think they will like and avoid ones they think they won't. Therefore, having rated certain items implicitly conveys a preference for similar items and not having rated others implicitly conveys an aversion for items similar to those. There are formulations of the KNN algorithm that incorporate implicit feedback data, which, if implemented, would likely yield better recommedations.  
    * Up-to-Date Anime Catalogue
        * Currently, Hakken can only recommend and incorporate the ratings of anime which aired during or before the summer of 2018 because the dataset from which the item-item similarity matrix was genarated was scraped from MAL at that time. Scraping current anime lists from users and building the item-item similarity matrix from them would allow Hakken to serve newer anime as recommendations. 

* **User-Interface**
    * Anime Picture and Synopsis
        * Adding an anime picture and synopsis alongside recommendation titles would improve the user experience as users would be able to determine which recommendations make sense for them without having to click each title link and be redirected to the corresponding MAL anime info page to find out more about them. 

* **Efficiency**
    * Faster Execution Time
        * Hakken can serve recommendations in approximately 2 minutes, but this time could be significantly reduced if the Hakken program was written in C++ instead of Python. 

