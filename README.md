# Simple-Netflix-Recommendation-System

This dataset consists of tv shows and movies available on Netflix from 1925 - 2020. The dataset has over 6000 rows and 13 columns containing details about the movies and tv shows such as the title, director, and cast of the shows / movies, the release year, duration etc. 

As the first step, I have loaded the dataset, cleanedd it up, added some new features and dropped older movies from 1925 - 2004. Simple exploratory analysis shows the top actors of the movies / TV shows in top countries like USA, UK, Canada, Japan, India.

Bag of words are created with features that correlate the most. Using cosine similarity and CountVectorization, the final recommendation function is built and output is displayed.
