# Hakken: An Anime Recommender System

* **Access the app here:** [Hakken](http://Hakken.ca)

## Background

Anime refers to hand-drawn and computer-generated animation originating from Japan. Anime is characterized by its' unique art style, expressive characters and fantastical themes. Since its' inception in 1920, anime has grown to become internationally popular and has found its' way into popular North American streaming services such as Netflix. 

## What is Hakken?

Hakken is a machine learning based anime recommendation system. It takes user's ratings of anime(s) they have previously seen and uses these ratings as a basis for generating personalized recommendations.

## Motivation

If Netflix already implements a machine learning based recommendation system and hosts animes on it's platform, why build Hakken? 

The selection of animes available on Netflix is very limited. Hence, avid anime fans use anime streaming services such as Crunchyroll, which provide more selection. Crunchyroll, however, does not have a personalized anime recommendation feature for their users. Thus, the motivation behind creating Hakken was to give anime fans automated and personalized recommendations from a much larger pool of animes than netflix can offer.  

## How it Works

### Problem/Solution Framework

The task of generating anime recommendations can be framed as follows: 

A model or algorithm should predict the ratings that a user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> would give to all the items <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> that the user has yet to see. These predicted ratings should then be sorted from highest to lowest and the unseen anime corresponding to the top N highest predicted ratings should be served as recommendations. 

### The Data

MyAnimeList.net is an anime community and database where users create profiles that feature a list of their anime ratings on a scale of 1-10. Kaggle user azathoth42 scraped together a dataset of ~45 million user ratings from public MyAnimeList profiles which was used for algorithm training and testing for Hakken.

### The Algorithm

Hakken uses the **Item-Based KNN-with-Baseline** collaborative filtering algorithm to predict the ratings that a user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> would give to all unseen items <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and generate recommendations. To explain how this algorithm works, we first introduce the baseline rating model and the item-based k-nearest-neighbours algorithm and then show how the Item-Based KNN-with-Baseline algorithm is synthesized from these two approaches. 

#### Baseline Rating

The baseline rating model is a simple model for predicting user ratings on items that assumes a user rating is a function of 3 terms: the global average rating across all items, the user bias and the item bias. User bias can be thought of as how much above or below the global average a user tends to rate items on average. Item bias can be thought of as how much above or below the global average an item tends to get rated on average. More simply, the user bias is a term meant to capture whether or not a user is a tough critic or easy-to-please rater. The item bias is a term mean to capture whether an item is inherently above or below average in quality. 

The baseline rating models takes the following form:

<p align="center"><img src="/tex/6ccff0a634302c00755da2ec7a8c94a1.svg?invert_in_darkmode&sanitize=true" align=middle width=119.65924079999999pt height=14.611878599999999pt/></p>

where <img src="/tex/a5eb431ad7aa0c93eae373def59f0e6f.svg?invert_in_darkmode&sanitize=true" align=middle width=19.477847399999987pt height=22.831056599999986pt/> is the baseline rating for a user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> on an item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>, <img src="/tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> is the global average item rating, <img src="/tex/f7115ca4804f002d341b17739b30a8a2.svg?invert_in_darkmode&sanitize=true" align=middle width=14.826947849999991pt height=22.831056599999986pt/> is the user bias and <img src="/tex/d3aa71141bc89a24937c86ec1d350a7c.svg?invert_in_darkmode&sanitize=true" align=middle width=11.705695649999988pt height=22.831056599999986pt/> is the item bias. 

<img src="/tex/f7115ca4804f002d341b17739b30a8a2.svg?invert_in_darkmode&sanitize=true" align=middle width=14.826947849999991pt height=22.831056599999986pt/> and <img src="/tex/d3aa71141bc89a24937c86ec1d350a7c.svg?invert_in_darkmode&sanitize=true" align=middle width=11.705695649999988pt height=22.831056599999986pt/> are chosen for each user and each item so as to minimize the regularized square error cost function: 

<p align="center"><img src="/tex/0bbbc985e19e603525467c62635ee212.svg?invert_in_darkmode&sanitize=true" align=middle width=318.63313845pt height=38.54816295pt/></p>

where <img src="/tex/165967386760509d1ff8fa2c775d4594.svg?invert_in_darkmode&sanitize=true" align=middle width=19.83937064999999pt height=14.15524440000002pt/> is the known rating by a user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> on an item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>, <img src="/tex/7efae7cfec0e6f3cd651382600af9603.svg?invert_in_darkmode&sanitize=true" align=middle width=43.81199624999999pt height=22.465723500000017pt/> is the set of all known ratings on which the baseline rating model is trained and <img src="/tex/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode&sanitize=true" align=middle width=9.58908224999999pt height=22.831056599999986pt/> is a regularization constant. 

#### Item-Based K-Nearest-Neighbours 

Item-Based K-Nearest-Neighbours is a collaborative filtering algorithm where the predicted rating for a user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> on an item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> is:

<p align="center"><img src="/tex/91e081752df2f735f85bd4a206bd301b.svg?invert_in_darkmode&sanitize=true" align=middle width=188.211936pt height=69.9243171pt/></p>

where <img src="/tex/4582c6a9d7f25e1c451b042f41906860.svg?invert_in_darkmode&sanitize=true" align=middle width=21.292981049999987pt height=14.15524440000002pt/> is the rating the user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> gave to the similiar item <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>, <img src="/tex/941eb0b434f6dc62fd63232c64b38fc3.svg?invert_in_darkmode&sanitize=true" align=middle width=41.536571999999985pt height=27.91243950000002pt/> is the set of <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> items that are most similar to show <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> the user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> has rated and <img src="/tex/fb2ac5ec63e77cef6130f20adbe67847.svg?invert_in_darkmode&sanitize=true" align=middle width=61.26676709999999pt height=24.65753399999998pt/> is the degree of similarity between an item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and another item <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> and is computed using the pearson correlation coefficient:

<p align="center"><img src="/tex/2e1dbf451cb924c7a8446d2e1d64d939.svg?invert_in_darkmode&sanitize=true" align=middle width=458.4001108499999pt height=76.6395531pt/></p>

where <img src="/tex/ce9c41bf6906ffd46ac330f09cacc47f.svg?invert_in_darkmode&sanitize=true" align=middle width=14.555823149999991pt height=14.15524440000002pt/> and <img src="/tex/0fc47ef768631c2c983a54d3bcdb8bf8.svg?invert_in_darkmode&sanitize=true" align=middle width=16.00943354999999pt height=14.15524440000002pt/> are the average ratings for items <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> respectively and <img src="/tex/91d79fc058413bff469d160f74946464.svg?invert_in_darkmode&sanitize=true" align=middle width=21.979146749999988pt height=22.465723500000017pt/> is the set of all users that have rated both items <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>. 

The algorithm works as follows:

Compute the pearson correlation coefficient as a measure of similarity between the set of ratings for item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and the set of ratings for item <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>, only taking into account instances where a user rated both items. Similarity between the ratings for the two items is used as a proxy for the degree of similarity between the items themsevles. This step is repeated for all pairs of items.

Then, find the k most similar items to the unseen item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> that user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> has also rated. The user's ratings on these similar items become the basis for predicting the rating on <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. The predicted rating is an average of the ratings by <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> on the k similar items <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>, weighted by how similar each of those k items are to the item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. The greater the similarity between <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> and <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>, the more the rating on <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> contributes to the predicted rating for <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>. Intuitively, if <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> gave a low rating to a very similar item <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>, then the predicted rating on <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> should go down. 

#### Item-Based KNN-with-Baseline

The KNN algorithm attempts to capture the effects of user-item interactions, i.e. user preferences for certain kinds of items over others, on user ratings. However, such effects are small compared to the influence of item and user biases. In other words, the majority of a user's rating on an item is determined by how critical of a rater that user is and how inherently above or below average in quality an item is, independent of a user's preference or aversion for that kind of item. It is only the extent to which a user's raw rating differs from the predicted baseline rating -- which encapsulates the effects of user and item bias -- that captures the user’s preference or aversion to a particular kind of item.

For example, suppose a user rates the anime, "Death Note",  a 9/10. If "Death Note" has a global average rating of 9/10, meaning it is universally liked, and the user's average rating is a 9/10, meaning the user gives high ratings easily, then it would not make sense to conclude that this user has a preference for "Death Note"-like shows.

Now, suppose a user rates the anime, "Stein's Gate", a 9/10. "Stein's Gate" has a global average rating of 7/10 and the user's average rating is a 7/10. Since the user rated this anime higher than what would be predicted from the user and item bias effects alone, we can conclude that the user has a preference for "Stein’s Gate"-like shows.

Thus, the goal is to feed  only the isolated part of the signal from the ratings data that truly represents user preference/aversion for certain items into the KNN algorithm. To that end, the Item-Based KNN-with-Baseline algorithm adjusts the ratings data by centering each rating by a user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> on an item <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>, <img src="/tex/165967386760509d1ff8fa2c775d4594.svg?invert_in_darkmode&sanitize=true" align=middle width=19.83937064999999pt height=14.15524440000002pt/>, on the baseline rating for that user-item pair, <img src="/tex/a5eb431ad7aa0c93eae373def59f0e6f.svg?invert_in_darkmode&sanitize=true" align=middle width=19.477847399999987pt height=22.831056599999986pt/>. The residuals, <img src="/tex/8cb256752b01dd3259f7c3ac8e245e87.svg?invert_in_darkmode&sanitize=true" align=middle width=60.230304749999995pt height=22.831056599999986pt/>, are then operated on almost identically to how the raw ratings are operated on in the standard Item-Based KNN algorithm:

<p align="center"><img src="/tex/1c3046589d45641ecee8dc9b6e8feb8f.svg?invert_in_darkmode&sanitize=true" align=middle width=198.29048744999997pt height=69.9243171pt/></p>

where <img src="/tex/cd19b2db8bfa1167b7138f0613456916.svg?invert_in_darkmode&sanitize=true" align=middle width=20.93145779999999pt height=22.831056599999986pt/> is the baseline rating for the user <img src="/tex/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode&sanitize=true" align=middle width=9.41027339999999pt height=14.15524440000002pt/> on the similar item <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>. 

The difference is that item similarity is determined by the shrunken pearson correlation coefficient in order to prevent the item similarities from overfitting the data.

The shrunk pearson-baseline correlation coefficient is calculated as:

<p align="center"><img src="/tex/21e40aa2e0111ab22633535a8c9ac7c9.svg?invert_in_darkmode&sanitize=true" align=middle width=483.3011469pt height=39.428498999999995pt/></p>

where <img src="/tex/521843328cd5f41a766f9f03fc7744b6.svg?invert_in_darkmode&sanitize=true" align=middle width=31.933468049999988pt height=24.65753399999998pt/> is the number of instances where a user rated both items <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>, "shrinkage" is a pre-determined shrinkage factor and <img src="/tex/43c0cdf6815c84296bfebd32c3d48434.svg?invert_in_darkmode&sanitize=true" align=middle width=19.25429219999999pt height=22.831056599999986pt/> is the pearson-baseline correlation coefficient, which is calculated as:

<p align="center"><img src="/tex/65810517d0ae26adce11a1b90f6d2d5b.svg?invert_in_darkmode&sanitize=true" align=middle width=572.4298255499999pt height=76.6395531pt/></p>

The extent of the shrinkage becomes greater when <img src="/tex/521843328cd5f41a766f9f03fc7744b6.svg?invert_in_darkmode&sanitize=true" align=middle width=31.933468049999988pt height=24.65753399999998pt/> is smaller, which is desirable because we are less confident in the computed similarity between items <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/> and <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> mapping onto reality when only a few users have rated both items. 

The first term in the predicted rating:

<p align="center"><img src="/tex/1c3046589d45641ecee8dc9b6e8feb8f.svg?invert_in_darkmode&sanitize=true" align=middle width=198.29048744999997pt height=69.9243171pt/></p> 

represents only the component of the predicted rating determined by user preferences. Therefore, it must be added to the baseline rating, the component of the predicted rating determined by user and item bias effects, in order to synthesize the full predicted rating, <img src="/tex/3581a5a4f5e44e560d6e9b1937f0c8e5.svg?invert_in_darkmode&sanitize=true" align=middle width=19.83937064999999pt height=22.831056599999986pt/>. 

Combining everything, the predicted rating takes the following form:

<p align="center"><img src="/tex/282a402a12db6368b4d42cda8af7d82b.svg?invert_in_darkmode&sanitize=true" align=middle width=283.232895pt height=69.9243171pt/></p>

<img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>, the number of nearest neighbours and "shrinkage", the pearson correlation coefficient shrinkage factor, are hyperparameters that are fine-tuned through a 3-fold RandomizedSearchCV procedure.

### Making Real-Time Recommendations

When using Hakken, users are asked to submit their MyAnimeList Username. Hakken uses the Jikan Unofficial MyAnimeList API to make an API call for the user's anime list from MyAnimeList.net.

The retrieved anime list is processed and fed into the Item-Based KNN-with-Baseline algorithm. A rating prediction is made for each unseen anime based on how the user rated similar anime in their anime list. These predicted ratings are then sorted from highest to lowest and the unseen anime corresponding to the top N (# of recommendations requested) highest predicted ratings are served as recommendations to the user.

The Item-Based KNN-with-Baseline algorithm is able to generate recommendations within  ~2 minutes. It generates recommendations at this speed because the item-item similarities, which have the longest computation time, are pre-computed and stored in numpy arrays which are loaded into memory when the program is first started. The user bias computation and nearest neighbours aggregation, which make up most of the 2 minute computation time, are performed on the spot because they cannot be pre-computed.


## Shortcomings and Future Improvements 

There are 3 main categories in which Hakken could be improved: recommendations, UI and efficiency.

* **Recommendations**
    * Implicit Feedback
        * Item ratings convey both explicit (the ratings themselves) and implicit feedback on user preferences.The implicit feedback is the knowledge of which items the users chose to rate in the first place, regardless of whether they were rated highly or lowly.  This is because users do not select anime to watch at random; they pick anime they believe they will enjoy and avoid ones they believe they won't enjoy. Therefore, having rated certain items implicitly conveys a preference for similar items and not having rated others implicitly conveys an aversion for similar items. There are formulations of the KNN algorithm that incorporate implicit feedback data, which if implemented, could yield better recommendations.
    * Up-to-Date Anime Catalogue
        * Currently, Hakken can only recommend and incorporate the ratings of animes that aired during or before the summer of 2018. This is because the dataset from which the item-item similarity matrix was generated  was scraped from MAL at that time. Scraping current anime lists from users and building the item-item similarity matrix from them would allow Hakken to recommend newer anime. 

* **User-Interface**
    * Anime Picture and Synopsis
        * Adding an anime picture and synopsis alongside recommendation titles would improve the user experience as they would be able to determine which recommendations are more appealing without having to click the link to learn more about them  on MAL.  

* **Efficiency**
    * Faster Execution Time
        * Hakken can serve recommendations in approximately 2 minutes. This time could be reduced, however, if Hakken was written in C++ instead of Python. 

## Contents
* Flask app in `ars_app.py`
* App functions in `make_recommendations.py`
* `create_gcs_local_file.py` is a script that loads model objects from google cloud storage
* HTML templates in `templates`
* CSS, Javascript and Image files in `static`
* Dependencies in `requirements.txt`

