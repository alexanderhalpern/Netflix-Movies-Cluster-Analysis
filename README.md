# Purpose: 
Generate a graphical depiction of the Netflix Movie and TV Show in order to determine trends in the data and visualize which movie and TV-show categories are becoming increasingly popular.


# Methodology:
Start by taking the description for each movie and TV show in the Netflix catalog and generate a numerical representation of each description 
by using sentence embeddings. By representing each catalog entry with a vector of numbers, we can perform a cluster analysis on the catalog entries
so that we can generate self-identified groupings of movies. After clustering, we can use dimensionality reduction to plot each movie description in 2D space. 

# Explore the Results
When we look at the following cluster graph of the Netflix movies and TV Shows, similar movies will be closer together in space, and different movies will be far apart.
Movies with the same color are part of the same self-identified grouping.

[Interact with the Cluster Graph here!](https://alexanderhalpern.github.io/Netflix-Movie-Cluster-HTML/)
[![Cluster Analysis](https://i.imgur.com/5EMLB5F.png)](https://alexanderhalpern.github.io/Netflix-Movie-Cluster-HTML/)

[Watch the video of how the Netflix Catalog has evolved over time here!](https://alexanderhalpern.github.io/Netflix-Movie-Bar-Graph/)
[![Bar Graph Video](https://i.imgur.com/yPV246n.png)](https://alexanderhalpern.github.io/Netflix-Movie-Bar-Graph/)
