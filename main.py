'''
*** Author: Alexander Halpern ***

Purpose: 
----------------
Generate a graphical depiction of the Netflix Movie and TV Show in order to determine trends in the data and visualize which movie and TV-show categories are popular.


Methodology:
----------------
Start by taking the description for each movie and TV show in the Netflix catalog and generate a numerical representation of each description 
by using sentence embeddings. By representing each catalog entry with a vector, we can then clustering catalog entries based on their descriptions
and then use dimensionality reduction so that we can plot movies in 2D space. 

This means that when we look at the graph of movies, similar movies are closer together in space, and different movies are far apart.

'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import plotly.express as px
from sklearn.cluster import KMeans
import json

# Initialize the Sentence Embedding model that we chose for this purpose (MiniLM-L6)
model = SentenceTransformer('all-MiniLM-L6-v2')

# The movie data is stored in a CSV file called updated_movie_data.csv
# load it into a pandas dataframe
df = pd.read_csv('updated_movie_data.csv')

# We are now going to use sentence-transformers to create embeddings for each movie overview/description
# so that we can better represention of movies in the netflix catalog

# The embeddings will be stored in a numpy array called embeddings.npy
# if embeddings.npy already exists, we can load it
if os.path.exists('embeddings.npy'):
    embeddings = np.load('embeddings.npy')
else:
    # Generate the embeddings if they don't exist already
    embeddings = model.encode(df['overview'], show_progress_bar=True)
    np.save('embeddings.npy', embeddings)

# After generating the embeddings we will use K-Means clustering to cluster the movies into 50 clusters
# to generate 50 different groups of movies based on their descriptions
num_clusters = 50
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_numbers = kmeans.fit_predict(embeddings)

# turn cluster_numbers into strings because it is easier to work with in the dataframe later on
cluster_numbers = [str(x) for x in cluster_numbers]


# These all-MiniLM-L6-v2 embeddings models produce vectors that have 384 dimensions
# If we want to be able to visualize the clusters, we need to reduce the dimensionality of the embeddings
# I have performed dimensionality reduction to both 2D and 3D, but I have found that 2D is easier to see
# We will use t-SNE algorithmn to reduce the dimensionality of the embeddings
# I experimented wtih PCA, but t-SNE produced better results
if os.path.exists('reduced_embeddings.npy'):
    reduced_embeddings = np.load('reduced_embeddings.npy')
else:
    # Let's reduce the dimensionality of the embeddings to 2D
    n_components = 2
    # Create t-SNE object and fit-transform the Netflix embeddings
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    np.save('reduced_embeddings.npy', reduced_embeddings)

# Create a dataframe that contains the reduced embeddings as well as relevant information about each movie from the original dataframe
# We are going to plot the embeddings in 2D space with the first dimension on the x-axis and the second dimension on the y-axis
df = pd.DataFrame(
    {
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster_numbers': cluster_numbers,
        'movie_titles': df['title'],
        'overview': df['overview'],
        "production_countries": df['production_countries'],
        "release_date": df['release_date']
    }
)

# Now we can generate unique colors for each cluster so that each dot on the plot will be colored according to its cluster
color_palette = px.colors.qualitative.Plotly
category_colors = {
    cat: color_palette[i % len(color_palette)]
    for i, cat in enumerate(sorted(set(cluster_numbers)))
}


# There are a lot of clusters, and I only want to see relatively large clusters, so I can examine large trends in the data
# I will filter out clusters with less than 50 movies

# Start out by counting the number of movies in each cluster
cluster_sizes = df['cluster_numbers'].value_counts()

# Only select clusters with 50 or more movies
selected_clusters = cluster_sizes[cluster_sizes >= 50].index
selected_df = df[df['cluster_numbers'].isin(selected_clusters)]

# Now for each cluster I want to set a maximum radius from the center of the cluster
# for the points that I want to keep. Points that are too far from the center of the cluster will be filtered out
selected_df['centroid'] = selected_df['cluster_numbers'].astype(
    str) + '_centroid'

# Calculate cluster centroids
centroids = selected_df.groupby('centroid')[['x', 'y']].mean()

# Merge centroids based on centroid column
selected_df = pd.merge(selected_df, centroids, left_on='centroid',
                       right_index=True, suffixes=('', '_centroid'))

# I set a rather arbitrary/experimental distance threshold of 15 in order to filter out points that are very far from the cluster center
distance_threshold = 15

# Use the distance formula to calculate the distance between each point and its cluster centroid
selected_df['distance_to_centroid'] = ((selected_df['x'] - selected_df['x_centroid'])**2 +
                                       (selected_df['y'] - selected_df['y_centroid'])**2)**0.5

# Now we can filter out points that are beyond the distance threshold
selected_df = selected_df[selected_df['distance_to_centroid']
                          <= distance_threshold]

# Do some final formatting of the data before we plot it

# Plotly uses HTML to format the hover text, so we will add <br> to have line breaks in the hover text
# We will add one after every 10 words
selected_df['overview'] = selected_df['overview'].apply(
    lambda x: '<br>'.join([' '.join(x.split()[i:i+10]) for i in range(0, len(x.split()), 10)]))

# Another part of the project was automatically generating cluster labels by passing
# all of the movie descriptions to a large language model and telling it to come up with a category
# header. A teammate did this and saved the cluster number to category header mapping in a JSON file
# This is how I grouped the movie descriptions by cluster number before he passed them to the LLM:
'''
movie_descriptions_per_cluster = df.groupby(['cluster_numbers'], as_index=False).agg(
     {
         'overview': ' '.join
     }
)
movie_descriptions_per_cluster.to_csv('movie_descriptions_per_cluster.csv', index=False)
'''

cluster_descriptions = json.load(open('cluster_number_description.json', 'r'))

# map cluster_labels to cluster_descriptions
selected_df['cluster_labels'] = selected_df['cluster_numbers'].map(
    cluster_descriptions)


# Finally we can plot all of the data with a plotly scatter plot!
fig = px.scatter(
    selected_df,
    x='x',
    y='y',
    color='cluster_numbers',
    hover_name='movie_titles',
    color_discrete_map=category_colors,
    labels={
        'cluster_labels': 'Cluster'
    },
    custom_data=['overview', 'movie_titles', 'cluster_labels']
)

# Set the hovertemplate to include the data we want
fig.update_traces(hovertemplate="<br>".join([
    "<span style='text-decoration:underline;'><b>Cluster Label:</b> %{customdata[2]}</span><br>",
    "<b>Title:</b> %{customdata[1]}<br>",
    "<b>Overview:</b> %{customdata[0]}<br>",
    "<extra></extra>"
]))

fig.update_layout(
    title='Netflix Movie Catalog Clustering (Similar Movies are Closer Together)')
fig.show()
fig.write_html("cluster_movies.html")
