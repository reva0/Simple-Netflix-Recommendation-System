#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot


# In[3]:


df = pd.read_csv("netflix_titles.csv", encoding = 'utf-8')


# In[4]:


df.shape


# In[34]:


df.head()


# In[5]:


df.isna().sum()


# In[ ]:


# since a lot of columns in director column are NaN, I am dropping that column


# In[6]:


df = df.drop(columns = ['description','date_added','rating','show_id','director'])


# In[7]:


df.shape


# In[49]:


df.head()


# In[50]:


df['type'].value_counts()


# In[17]:


df['release_year'].value_counts()


# In[ ]:


# Sorting by release year and dropping the year from 1925 to 2005 as the count is small and it has NaN values


# In[37]:


sorted_df = df.sort_values('release_year')


# In[38]:


df = sorted_df.iloc[574:]


# In[39]:


df.head()


# In[59]:


## add new features in the dataset

df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)
df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)
df.head()


# In[40]:


df.isna().sum()


# In[41]:


df.shape


# In[47]:


col = "type"
grouped = df[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6a749b", "#a688de"]))
layout = go.Layout(title="", height=400, legend=dict(x=0.1, y=1.1))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)


# In[50]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df['release_year'].hist(bins=5)


# In[ ]:


## TOP ACTORS - FROM MOVIES LIST


# In[60]:


from collections import Counter
def country_trace(country, flag = "movie"):
    df["from_us"] = df['country'].fillna("").apply(lambda x : 1 if country.lower() in x.lower() else 0)
    small = df[df["from_us"] == 1]
    if flag == "movie":
        small = small[small["duration"] != ""]
    else:
        small = small[small["season_count"] != ""]
    cast = ", ".join(small['cast'].fillna("")).split(", ")
    tags = Counter(cast).most_common(25)
    tags = [_ for _ in tags if "" != _[0]]

    labels, values = [_[0]+"  " for _ in tags], [_[1] for _ in tags]
    trace = go.Bar(y=labels[::-1], x=values[::-1], orientation="h", name="", marker=dict(color="#a678de"))
    return trace

from plotly.subplots import make_subplots
traces = []
titles = ["United States", "","India","", "United Kingdom", "Canada","", "Spain","", "Japan"]
for title in titles:
    if title != "":
        traces.append(country_trace(title))

fig = make_subplots(rows=2, cols=5, subplot_titles=titles)
fig.add_trace(traces[0], 1,1)
fig.add_trace(traces[1], 1,3)
fig.add_trace(traces[2], 1,5)
fig.add_trace(traces[3], 2,1)
fig.add_trace(traces[4], 2,3)
fig.add_trace(traces[5], 2,5)

fig.update_layout(height=1200, showlegend=False)
fig.show()


# In[ ]:


## TOP ACTORS - FROM TV LIST


# In[70]:


traces = []
titles = ["United States","", "United Kingdom", "", "Canada"]
for title in titles:
    if title != "":
        traces.append(country_trace(title, flag="tv_shows"))

fig = make_subplots(rows=2, cols=3, subplot_titles=titles)
fig.add_trace(traces[0], 1,1)
fig.add_trace(traces[1], 1,3)
fig.add_trace(traces[2], 2,2)
# fig.add_trace(traces[3], 2,3)

fig.update_layout(height=900, showlegend=False)
fig.show()                                


# In[ ]:





# In[ ]:


## nothing to predict (no target variable), use clustering and groupby genre / cast


# In[76]:


df.isna().sum()


# In[95]:


new_df = df[['title','cast','listed_in','country']]
new_df.head()

new_df.dropna(inplace=True)

blanks = []  # start with an empty list

col=['title','cast','listed_in','country']
for i,col in new_df.iterrows():  # iterate over the DataFrame
    if type(col)==str:            # avoid NaN values
        if col.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

new_df.drop(blanks, inplace=True)


# In[96]:


new_df.head()


# In[80]:


new_df.shape


# In[97]:


new_df['cast'] = new_df['cast'].map(lambda x: x.split(',')[:3])

# putting the genres in a list of words
new_df['listed_in'] = new_df['listed_in'].map(lambda x: x.lower().split(','))

# new_df['director'] = new_df['director'].map(lambda x: x.split(' '))

# merging together first and last name for each actor and director, so it's considered as one word 
# and there is no mix up between people sharing a first name
for index, row in new_df.iterrows():
    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]
#     row['director'] = ''.join(row['director']).lower()


# In[98]:


new_df.set_index('title', inplace = True)
new_df.head()


# In[ ]:





# In[86]:


# generating the cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[99]:


# Joining words

new_df['words'] = ''
column = new_df.columns
for index, row in new_df.iterrows():
    words = ''
    for col in columns:
            words = words + ', '.join(row[col])+ ', '
#         else:
#             words = words + row[col]+ ' '
    row['words'] = words
    
new_df.drop(columns = [col for col in new_df.columns if col!= 'words'], inplace = True)


# In[100]:


new_df.head()


# In[112]:


# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(new_df['words'])

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(new_df.index)
indices[:5]


# In[102]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim


# In[ ]:





# In[103]:


def recommendations(Title, cosine_sim = cosine_sim):
    
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == Title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(new_df.index)[i])
        
    return recommended_movies


# In[108]:


recommendations("Atlantics")


# In[110]:


recommendations("Good People")


# In[107]:


recommendations("The Zoya Factor")

