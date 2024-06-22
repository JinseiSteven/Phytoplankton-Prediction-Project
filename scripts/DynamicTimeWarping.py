import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sbn
import os
from sklearn.cluster import KMeans
# we gon cook here a bit dw about it

class DynamicTimeWarping:
    def __init__(self):
        self.metric_list = ["euclidean", "dotprd", "cosdis"]
        self._distance_matrices = []
        self._warp_paths = []
        self._similarity_matrix = None

    def similarity(self, series1, series2, metric="euclidean"):
        # ensuring a numpy array can be used
        series1, series2 = (np.asarray(series1), np.asarray(series2))

        # preparing the distance matrix
        self._distance_matrix = np.matrix(np.ones((series1.shape[0], series2.shape[0])) * np.inf)
        self._distance_matrix[0, 0] = self._calc_distance(series1[0], series2[0], metric)

        for i in range(1, len(series1)):
            self._distance_matrix[i, 0] = self._calc_distance(series1[i], series2[0], metric) + self._distance_matrix[i-1, 0]  
        
        for j in range(1, len(series2)):
            self._distance_matrix[0, j] = self._calc_distance(series1[0], series2[j], metric) + self._distance_matrix[0, j-1]

        self.warp_path = [(0, 0)]

        # we calculate the sequence cost for every combination
        for i in range(1, len(series1)):
            for j in range(1, len(series2)):

                # calculating the distance between the data points (using the chosen metric)
                distance = self._calc_distance(series1[i], series2[j], metric)

                # assigning a new matrix value, based on the distance and previously chosen best matching sequence
                self._distance_matrix[i, j] = distance + min([self._distance_matrix[i-1, j],
                                                        self._distance_matrix[i, j-1],
                                                        self._distance_matrix[i-1, j-1]])
        
        # the last most value of the matrix is the optimal value
        return self._distance_matrix[-1, -1]
    
    def similarity_matrix(self, data, metric="euclidean"):
        # initialize the similarity matrix
        self._similarity_matrix = np.zeros((len(data), len(data)))

        self._distance_matrices = []
        # checking similarity for all series pairs
        for i, series1 in enumerate(data):
            for j, series2 in enumerate(data):
                self._similarity_matrix[i, j] = self.similarity(series1, series2, metric)
                self._distance_matrices.append(self._distance_matrix)
                self._warp_paths.append(self._get_path(self._distance_matrix))
                print(f"Pair ({i}, {j}) done - {int((i * len(data) + j) / ((len(data))**2) * 100)}%")
        
        return self._similarity_matrix
    
    def get_results(self):
        if len(self._distance_matrices) > 0 and len(self._warp_paths) > 0:
            return self._distance_matrices, self._warp_paths
    
    def plot_similarity_matrix(self, save=False, filename="DTW_similarity_plot"):
        if self._similarity_matrix is None:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))

        ax = sbn.heatmap(self._similarity_matrix, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)

        if save:
            fig.savefig(f"{filename}.png")

        return ax

    def save_distance_matrices(self, folder_dir="DTW_Matrices"):
        if len(self._distance_matrices) == 0:
            return

        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, f'{folder_dir}/')

        # creating a directory for the resulting matrix plots if it does not exist yet
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        nr_series = int(math.sqrt(len(self._distance_matrices)))
        for i, (distance_matrix, warp_path) in enumerate(zip(self._distance_matrices, self._warp_paths)):
            row = i // nr_series
            column = i % nr_series

            fig, ax = plt.subplots(figsize=(12, 8))

            ax = sbn.heatmap(distance_matrix, annot=True, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
            ax.invert_yaxis()

            # adding 0.5 to each path coordinate to center them
            ax.plot([p[1] + 0.5 for p in warp_path], 
                    [p[0] + 0.5 for p in warp_path], 
                    color='blue', linewidth=3, alpha=0.2)

            plt.savefig(results_dir + f"DTW_map({row}-{column}).png")
    
    def _calc_distance(self, series1, series2, metric):
        if metric not in self.metric_list:
            print(f'Metric "{metric}" not found in metric availabilities, defaulted to "euclidean".')
            print("Available metrics are: euclidean (euclidean distance), dotprd (dot product) and cosdis (cosine distance).")
            metric = "euclidean"
        
        if metric == "euclidean":
            return np.linalg.norm(series1 - series2)
        elif metric == "dotprd":
            return np.dot(series1, series2)
        elif metric == "cosdis":
            return 1 - (np.dot(series1, series2) / (np.linalg.norm(series1) * np.linalg.norm(series2)))
        else:
            print("ehhhhhh what happened")
    
    def _get_path(self, distance_matrix):
        m, n = distance_matrix.shape
        path = [(m - 1, n - 1)]

        i = m - 1
        j = n - 1

        # retracing the distance matrix, to record the path of shortest distance
        while i != 0 or j != 0:
            left = distance_matrix[i, j-1]
            up = distance_matrix[i-1, j]
            diag = distance_matrix[i-1, j-1]

            if left < up and left < diag:
                path.append((i, j-1))
                j -= 1
            elif up <= left and up < diag:
                path.append((i-1, j))
                i -= 1
            elif diag <= up and diag <= left:
                path.append((i-1, j-1))
                i -= 1
                j -= 1

        return path


def plot_inertia(input, nr_clusters_limit):
    inertia_list = []

    cluster_counts = range(1,nr_clusters_limit)

    for nr_clusters in cluster_counts:
        abiotic_kmeans = KMeans(n_clusters=nr_clusters, n_init='auto')
        abiotic_kmeans.fit(input)
        inertia_list.append(abiotic_kmeans.inertia_)

    fig = plt.figure(figsize=(6, 5))

    plt.plot(cluster_counts, inertia_list, marker='o')
    plt.xlabel('Nr Clusters')
    plt.ylabel('Kmean Inertia')

    plt.savefig("inertia.png")


def main():
    nr_series = 15
    data = []

    for _ in range(nr_series):
        data.append(np.random.rand(100, 1))

    DTW = DynamicTimeWarping()

    similarity_matrix = DTW.similarity_matrix(data)

    plot_inertia(similarity_matrix, 6)

    kmeans_model = KMeans(n_clusters = 3)
    clusters = kmeans_model.fit_predict(similarity_matrix)
    print(clusters)

    DTW.plot_similarity_matrix(save=True, filename="DTW_Test_Plot")
    




if __name__ == "__main__":
    main()