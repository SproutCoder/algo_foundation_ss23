"""
This Python script performs k-means clustering on a given dataset and save the intermediate and 
final cluster assignments and centroids to a file. Also, saves a plot of the clustered 
dataset to a file.

Usage:
1. Set the value of k to the desired number of clusters. (k=3 here)
2. Define the dataset S. (given on the exercise sheet)
3. Run the script to cluster the data, save the results to "kmeans_results.txt", and generate 
and save a plot of the clustered dataset to "kmeans_plot.png".

Dependencies:
- NumPy
- Matplotlib
"""


import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') #damit ich den plot in WSL 2 angezeigt bekomme
import matplotlib.pyplot as plt

# Set the number of clusters
k = 3

# our data set S
S = np.array([[2, 12], [3, 11], [3, 8], [5, 4], [7, 5], [7, 3], [10, 8], [13, 8]])

# Set the initial centroids to A, B, and C
centroids = np.array([S[0], S[1], S[2]])

# Initialize an array to store the cluster assignments for each data point
cluster_assignments = np.zeros(S.shape[0])

# Initialize a variable to keep track of whether the centroids have changed
centroids_changed = True
counter = 0
print("k-means algo initilized")

# Repeat until the centroids no longer change
while centroids_changed:
    counter += 1
    # Assign each data point to the nearest centroid
    for i in range(S.shape[0]):
        distances = np.linalg.norm(S[i] - centroids, axis=1)
        cluster_assignments[i] = np.argmin(distances)

    # Update the centroids by taking the mean of all data points assigned to each cluster
    new_centroids = np.array([S[cluster_assignments == j].mean(axis=0) for j in range(k)])

    # Check if the centroids have changed
    if np.all(centroids == new_centroids):
        centroids_changed = False
    else:
        centroids = new_centroids

    #save results in kmeans_results.txt
    with open("kmeans_results.txt", "w", encoding="utf-8") as f:
        f.write(f"{counter}. Cluster assignments (intermediate): {cluster_assignments}\n")
        f.write(f"Centroids (intermediate): {centroids}\n\n")



# Create plot to test wether it worked
colors = ['r', 'g', 'b'] # Assign colors to each cluster

# Plot the data points and color-code them according to their assigned clusters
for i in range(k):
    plt.scatter(S[cluster_assignments == i, 0], S[cluster_assignments == i, 1], c=colors[i])
# Plot the centroids as black crosses
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='k') 

# Set the axis labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('k-means clustering')

# Saving the plot
FILENAME = 'kmeans_plot.png' # Set the filename to save the plot
plt.savefig(FILENAME) # Save the plot to the specified filename
print("plot saved to 'kmeans_plot.png'")
print('all done (showing plot)')

# Show the plot
plt.show()