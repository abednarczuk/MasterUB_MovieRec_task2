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


# In[2]:


get_ipython().run_line_magic('run', '01_MovieRec_EDA.ipynb')


# ## 1. Item-Based Recommender System

# * Adjusted cosine similarity: https://stackoverflow.com/questions/48941825/adjusted-cosine-similarity-in-python

# In[3]:


ind_to_movie = {}
for index, row in movies.iterrows():
    ind_to_movie[row['movie_id']] = row['title']


# In[4]:


pd.DataFrame(data.groupby('user_id').rating.mean()).rename(columns={'rating': 'user_mean'})


# In[5]:


## Divide the data in two sets: training and test
def assign_to_set(df):
    sampled_ids = np.random.choice(df.index,
                                   size=np.int64(np.ceil(df.index.size * 0.2)),
                                   replace=False)
    df.loc[sampled_ids, 'for_testing'] = True
    return df

data['for_testing'] = False
grouped = data.groupby('user_id', group_keys=False).apply(assign_to_set)
data_train = data[grouped.for_testing == False]
data_test = data[grouped.for_testing == True]
print(data_train.shape)
print(data_test.shape)
print(data_train.index & data_test.index)


# In[6]:


dataSmall = data[data['user_id']<100] # get only data from 100 users
print(dataSmall.shape)

dataSmall.loc[:,'for_testing'] = False
grouped = dataSmall.groupby('user_id', group_keys=False).apply(assign_to_set)
dataSmall_train = dataSmall[grouped.for_testing == False]
dataSmall_test = dataSmall[grouped.for_testing == True]

print(dataSmall_train.shape )
print(dataSmall_test.shape )

print('Number of users:', dataSmall.user_id.nunique() )
print('Number of movies:',dataSmall.movie_id.nunique() )


# In[7]:


from scipy.spatial import distance

def SimAdjCos(data, movie1, movie2, min_common_items=1):
    users1 = data[data['movie_id'] == movie1]
    users2 = data[data['movie_id'] == movie2]
    df = pd.merge(users1, users2, on='user_id')
    #df2 = pd.merge(df, pd.DataFrame(data.groupby('user_id').rating.mean()).rename(columns={'rating': 'user_mean'}), on='user_id')
    
    if len(df)<2:
        return 0    
    if(len(df)<min_common_items):
        return 0   
    
    sim = distance.cosine(df['rating_x'], df['rating_y'])
    if(np.isnan(sim)):
        return 0
    return sim

def SimPearson(DataFrame,movie1,movie2,min_common_items=1):
    # GET MOVIES OF USER1
    users1=DataFrame[DataFrame['movie_id'] ==movie1 ]
    # GET MOVIES OF USER2
    users2=DataFrame[DataFrame['movie_id'] ==movie2 ]
    
    # FIND SHARED FILMS
    rep=pd.merge(users1 ,users2,on='user_id',)
    if len(rep)<2:
        return 0    
    if(len(rep)<min_common_items):
        return 0    
    res=pearsonr(rep['rating_x'],rep['rating_y'])[0]
    if(np.isnan(res)):
        return 0
    return res


# In[8]:


class ItemCF:
   
    def __init__(self, data, similarity = SimAdjCos):
        self.data = data
        self.sim_method=similarity
        self.user_item = data.pivot(index = 'user_id', columns = 'movie_id', values = 'rating').fillna(0)
        self.sim_matrix = pd.DataFrame(np.sum([0]),columns=data_train.movie_id.unique(), index=data_train.movie_id.unique())
        
#     def learn(self):
#         avg_user_rating = data.pivot(index = 'user_id', columns = 'movie_id', values = 'rating').mean(axis=1)
#         item_mean_subtracted = self.user_item - avg_user_rating[:, None]
#         self.sim_matrix = 1 - squareform(pdist(item_mean_subtracted.T, 'cosine'))
        
#         self.sim_ind = {}
#         for i, m in zip(list(self.user_item.columns), range(len(list(self.user_item.columns)))):
#             self.sim_ind[i] = m
#         return print(self.sim_matrix)

    def learn(self):
        all_movies=set(self.data['movie_id'])
        self.sim_matrix = {}
        for movie1 in all_movies:
            self.sim_matrix.setdefault(movie1, {})
            a=data_train[data_train['movie_id']==movie1][['user_id']]
            data_reduced=pd.merge(data_train,a,on='user_id')
            for movie2 in all_movies:
                print(movie1, movie2)
                if movie1==movie2: continue
                self.sim_matrix.setdefault(movie2, {})
                if(movie1 in self.sim_matrix[movie2]):continue # since is a simetric matrix
                sim=self.sim_method(data_reduced,movie1,movie2)
                if(sim<0):
                    self.sim_matrix[movie1][movie2]=0
                    self.sim_matrix[movie2][movie1]=0
                else:
                    self.sim_matrix[movie1][movie2]=sim
                    self.sim_matrix[movie2][movie1]=sim
            
    def estimated_rating(self, user_id, movie_id):
        rating_num = 0.0
        rating_den = 0.0
        if self.user_item[movie_id][user_id] != 0:
            return print('User {0} has already seen "{1}" (rating: {2})'.format(user_id, ind_to_movie[movie_id], self.user_item[movie_id][user_id]))
        else:
            user_movies = list(self.data[self.data['user_id'] == user_id].movie_id)
        for movie in user_movies:
            if movie_id==movie: continue 
            rating_num += self.sim_matrix[self.sim_ind[movie], self.sim_ind[movie_id]] * self.user_item[movie][user_id]
            rating_den += self.sim_matrix[self.sim_ind[movie], self.sim_ind[movie_id]]
        if rating_den==0: 
            if self.data.rating[self.data['movie_id']==movie_id].mean()>0:
                # return the mean movie rating if there is no similar for the computation
                return self.data.rating[self.data['movie_id']==movie_id].mean()
            else:
                # else return mean user rating 
                return self.data.rating[self.data['user_id']==user_id].mean()
        return print('Estimated rating for "{0}" is {1}'.format(ind_to_movie[movie_id], round(rating_num/rating_den,2))) 

    def estimate(self, user_id, movie_id):
        rating_num = 0.0
        rating_den = 0.0
        user_movies = list(self.data[self.data['user_id'] == user_id].movie_id)
        for movie in user_movies:
            if movie_id==movie: continue 
            rating_num += self.sim_matrix[self.sim_ind[movie], self.sim_ind[movie_id]] * self.user_item[movie][user_id]
            rating_den += self.sim_matrix[self.sim_ind[movie], self.sim_ind[movie_id]]
        if rating_den==0: 
            if self.data.rating[self.data['movie_id']==movie_id].mean()>0:
                # return the mean movie rating if there is no similar for the computation
                return self.data.rating[self.data['movie_id']==movie_id].mean()
            else:
                # else return mean user rating 
                return self.data.rating[self.data['user_id']==user_id].mean()
        return round(rating_num/rating_den,2)


# In[11]:


check = ItemCF(dataSmall_train)


# In[ ]:


check.learn()


# In[ ]:


check.estimated_rating(4,1)


# In[ ]:





# ## 2. Evaluate

# In[ ]:


def compute_rmse(y_pred, y_true):
    """ Compute Root Mean Squared Error. """
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))


# In[ ]:


def evaluate(estimate_f,data_train,data_test):
    """ RMSE-based predictive performance evaluation with pandas. """
    ids_to_estimate = zip(data_test.user_id, data_test.movie_id)
    estimated = np.array([estimate_f(u,i) if u in data_train.user_id else 3 for (u,i) in ids_to_estimate ])
    real = data_test.rating.values
    return compute_rmse(estimated, real)


# In[ ]:


print('RMSE for Collaborative Recomender: %s' % evaluate(check.estimate,dataSmall_train,dataSmall_test))


# In[ ]:




