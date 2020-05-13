#!/usr/bin/env python
# coding: utf-8

# Aleksandra Bednarczuk

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm


# In[2]:


get_ipython().run_line_magic('run', '01_MovieRec_EDA.ipynb')


# ## 1. Rating matrix

# In[3]:


UserItem = data.pivot(index = 'user_id', columns = 'movie_id', values = 'rating').fillna(0)


# In[4]:


UserItem


# In[5]:


UserItem_sparse = csr_matrix(UserItem.values.transpose()) # transpose - because I want to find neighbours for items, not users


# In[6]:


movie_to_ind = {
    movie: i for i, movie in 
    enumerate(list(movies.set_index('movie_id').loc[UserItem.columns].title))
}
# movie_to_ind = {}
# for index, row in movies.iterrows():
#     movie_to_ind[row['title']] = row['movie_id']


# ## 2.  Recommend 10 movies based on a favourite movie title

# * Based on: https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea and https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_KNN.ipynb 
# * K-nearest neighbours with cosine distance as a metric
# * Recommending 10 similar movies (10 nearest neighbours based on cosine similarity)

# In[7]:


model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(UserItem_sparse)


# In[11]:


def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 50:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]  

def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (ind, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[ind], dist))    


# In[12]:


my_favorite = 'Dumbo'

make_recommendation(
    model_knn=model_knn,
    data=UserItem_sparse,
    fav_movie=my_favorite,
    mapper=movie_to_ind,
    n_recommendations=10)


# In[ ]:




