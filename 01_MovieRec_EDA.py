#!/usr/bin/env python
# coding: utf-8

# Aleksandra Bednarczuk

# References:
# * *Recommenders System* course materials by professor Santiago Segui (https://github.com/ssegui/recsysMaster2020)
# * Badrul Sarwar, George Karypis, Joseph Konstan, John Riedl, *Item-based Collaborative Filtering Recommendation Algorithms*, 2001 (https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.167.7612)
# * Paolo Cremonesi, Yehuda Koren, Roberto Turrin, *Performance of Recommender Algorithms on Top-N Recommendation Tasks*, 2010 (https://dl.acm.org/doi/pdf/10.1145/1864708.1864721)
# * https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
# * https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_KNN.ipynb

# # Movie recommender system

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


# ## 1. Loading data

# In[2]:


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/users.dat', sep='::', names=u_cols)

m_cols = ['movie_id', 'title', 'genre']
movies = pd.read_csv('data/movies.dat', sep='::', names=m_cols, usecols=range(3), encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('data/ratings.dat', sep='::', names=r_cols, usecols=range(3))


# In[3]:


users.head()


# In[4]:


movies.head()


# In[5]:


ratings.head()


# ## 2. Exploratory data analysis

# * As the CF algorithm takes rating matrix (users x items matrix) as an input, in the next steps I create the input DataFrame from the *ratings* DataFrame only. 
# 
# * The *movies* DataFrame will be used to acquire movies' titles for recommendation system.

# In[6]:


print('Shape of the original ratings dataset: {}'.format(ratings.shape))


# In[7]:


print('There are {} unique users and {} unique movies in this data set'.format(ratings.user_id.nunique(), 
                                                                               ratings.movie_id.nunique()))


# In[8]:


print('Movies were rated from {} to {}'.format(ratings.rating.min(), ratings.rating.max()))


# In[9]:


ratings.info()


# In[10]:


ratings.rating.describe()


# * While building the recommendation algorithm, it is important to address the long-tail problem - the situation when the dataset contains big portion of data with low user-interaction. Below two types of this problem are investigated - movies with small number of ratings and users with low level of activity (small number of rated movies).

# * Most of the movies in the dataset have a very low number of ratings - only a small number of all items (refered to as "popular items") are frequently rated by users. To avoid the bias created by the items with very small number of ratings, the analysis focuses only on the more popular items - the dataset is limited to the movies which were rated by at least 100 users.

# In[11]:


movies_ratings_count = pd.DataFrame(ratings.groupby('movie_id').size(), columns=['count'])
ax = movies_ratings_count.sort_values('count', ascending=False).reset_index(drop=True).plot(figsize=(8, 5), title='Rating frequency of the movies', fontsize=10)
ax.set_xlabel("movies")
ax.set_ylabel("number of ratings")
plt.xlim(-100,3800)
plt.show()


# In[12]:


movies_ratings_count['count'].quantile(np.arange(1, 0.0, -0.05))


# In[13]:


popular_movies = list(set(movies_ratings_count.query('count >= 100').index))
data = ratings.loc[ratings.movie_id.isin(popular_movies)]
print('Number of movies in the original ratings data: ', ratings.movie_id.nunique())
print('Shape of the original ratings data: ', ratings.shape)
print('--')
print('Number of movies after dropping unpopular movies: ', data.movie_id.nunique())
print('Shape of the dataset after dropping unpopular movies: ', data.shape)


# * Some of the users rate only small number of movies - they either do not watch many movies or do not rate the movies they watched. To lower the impact of ratings by inactive users, the dataset is limited to the users who rated at least 50 movies (70% of users).

# In[14]:


users_ratings_count = pd.DataFrame(ratings.groupby('user_id').size(), columns=['count'])
ax = users_ratings_count.sort_values('count', ascending=False).reset_index(drop=True).plot(figsize=(8, 5), title='Rating frequency of the users', fontsize=10)
ax.set_xlabel("users")
ax.set_ylabel("number of ratings")
plt.xlim(-100,6100)
plt.show()


# In[15]:


users_ratings_count['count'].quantile(np.arange(1, 0.0, -0.05))


# In[16]:


active_users = list(set(users_ratings_count.query('count >= 50').index))
data = data.loc[data.user_id.isin(active_users)]
print('Number of users in the original ratings data: ', ratings.user_id.nunique())
print('Shape of the original ratings data: ', ratings.shape)
print('--')
print('Number of users after dropping inactive users: ', data.user_id.nunique())
print('Shape of the dataset after dropping unpopular movies and inactive users: ', data.shape)


# In[17]:


data


# In[ ]:




