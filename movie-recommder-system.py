#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np
import pandas as pd


# In[128]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# In[129]:


movies.head()


# In[130]:


credits.head()


# In[131]:


movies = movies.merge(credits,on='title')


# In[132]:


movies.head(1)


# In[133]:


#genres
#id
#keywords
#title
#overview
#cast
#crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[134]:


movies.head()


# In[135]:


movies.isnull().sum()


# In[136]:


movies.dropna(inplace=True)


# In[137]:


movies.iloc[0].genres


# In[138]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','Fantasy','SciFi']


# In[139]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[140]:


import ast 
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[141]:


movies['genres'] = movies['genres'].apply(convert)


# In[142]:


movies.head()


# In[143]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[144]:


movies.head()


# In[145]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[146]:


movies['cast'] = movies['cast'].apply(convert3)


# In[147]:


movies.head() 


# In[148]:


movies['crew'][0]


# In[149]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[150]:


movies['crew'].apply(fetch_director)


# In[151]:


movies['overview'][0]


# In[152]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[153]:


movies.head()


# In[154]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "")for i in x])


# In[188]:


movies.head()


# In[189]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[190]:


movies.head()


# In[191]:


new_df = movies[['movie_id','title','tags']]


# In[192]:


new_df['tags']  = new_df['tags'].apply(lambda x:" ".join(x))


# In[193]:


new_df.head()


# In[194]:


new_df['tags'][0]


# In[195]:


new_df['tags']  = new_df['tags'].apply(lambda x:x.lower())


# In[196]:


new_df.head()


# In[197]:


import nltk


# In[198]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[199]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
        


# In[200]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[201]:


new_df [ 'tags'][1]


# In[ ]:





# In[204]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[205]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[206]:


vectors


# In[207]:


vectors[0]


# In[208]:


cv.get_feature_names()


# In[210]:


from sklearn.metrics.pairwise import cosine_similarity


# In[211]:


similarity = cosine_similarity(vectors)


# In[220]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[245]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(i)


# In[247]:


recommend('Batman Begins')


# In[243]:


recommend('Avatar')


# In[239]:


import pickle


# In[240]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[229]:


new_df['title'].values


# In[233]:


pickle.dump(new_df.to_dict(),open('movie.dict.pkl','wb'))


# In[234]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




