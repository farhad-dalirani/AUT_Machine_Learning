def euclidean_distance(p, q):
    """
    Euclidean Distance Between two points
    :param p: a d-dimentional point
    :param q: a d-dimentional point
    :return: a scalar, Euclidean Distance Between point p and q
    """
    import numpy as np

    dis = np.sqrt(((p-q)**2).sum())
    return dis


def sum_of_squared_error(clusters):
    """
    This function calculates SSE (Sum Of Squared Error), for measuring quality of clusters
    :param clusters: a list numpy list which demonstrates
                    points in different clusters
    :return: a scalar number
    """
    import numpy as np

    # Centers of clusters
    c = [cluster.mean(axis=0) for cluster in clusters]

    # squared of points distances to the center of their cluster
    sum_of_squareds = 0.0
    for i, cluster in enumerate(clusters):
        sse_i_th_cluster = 0
        for point in cluster:
            # Calculate sum of distances of points to the center of iTh cluster
            sse_i_th_cluster += euclidean_distance(point, c[i])**2

        sum_of_squareds += sse_i_th_cluster

    return sum_of_squareds


def kmeans(X, k, max_iter=100):
    """
    This function is a K-means clustering algorithm
    :param X: Input dataset
    :param max_iter: maximum number of iterations that clustering algorithm
                    does to coverage
    :return: Return SSE cost for each iteration, clusters, Centroids for each iteration
            if a cluster become empty, exit function by returning None, None, None
    """
    import numpy as np

    np.random.seed(0)

    # Randomly choose k centroids
    random_indexes = np.random.choice(a=range(X.shape[0]), size=k, replace=False)
    centroids = X[random_indexes]
    #centroids = np.random.randint(low=-5, high=5, size=X.shape)

    centroids_per_iteration = [np.copy(centroids)]
    sse_errors = []
    for _ in range(max_iter):

        clusters = [[] for i in range(k)]

        # Assign points to their nearest centroid
        for point in X:

            # Calculate distance of the point to all centroids
            distance_to_centers = []
            for cluster in range(k):
                distance_to_centers.append(euclidean_distance(point, centroids[cluster]))

            # Add point to cluster of nearest centroid
            nearest_centroid = np.argmin(distance_to_centers)
            clusters[nearest_centroid].append(np.copy(point))

        clusters = [np.array(cluster) for cluster in clusters]
        # update centroids
        for index, cluster in enumerate(clusters):
            if cluster.shape[0] != 0:
                centroids[index] = cluster.mean(axis=0)
            else:
                # A cluster is empty return None, None, None
                return None, None, None

        # Calculate Davies Bouldin for each iteration
        sse_errors.append(sum_of_squared_error(clusters))
        centroids_per_iteration.append(np.copy(centroids))
        # plot clusters in each iteration
        #plt.figure()
        #for i, cluster in enumerate(clusters):
        #    plt.plot(cluster[:, 0], cluster[:, 1], '*', label='Cluster {}'.format(i + 1))
        #plt.plot(centroids[:, 0], centroids[:, 1], 'o', label='Initial Centroid')
        #plt.legend(loc='upper left')
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.show()

    # Return Davies Bouldin cost for each iteration and clusters, Centroids for each iteration
    return sse_errors, clusters, centroids_per_iteration


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Read Input file
    dataset = pd.read_csv('data2.csv', header=None)

    # Points that we want to cluster them
    data = dataset.values

    # Plot data
    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Input Data')

    sse_errrors_ = []
    davies_bouldin_per_iteration_k = []
    clusters_k = []
    centroids_per_iteration_k=[]

    X= np.copy(data)
    X_normal= (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

    for inputData in [X, X_normal]:

        # For different K do clustering
        for k in [2, 3]:
            while True:
                sse_errrors_, clusters, centroids_per_iteration = kmeans(X=inputData, k=k)
                if clusters != None:
                    break

            # Store output of clustring with k cluster
            clusters_k.append(clusters)
            centroids_per_iteration_k.append(centroids_per_iteration)

            # plot clusters
            plt.figure()
            for i, cluster in enumerate(clusters):
                plt.plot(cluster[:, 0], cluster[:, 1], 'o',label='Cluster {}'.format(i+1))
            # plot initial centroids
            #plt.plot(centroids_per_iteration[0][:,0], centroids_per_iteration[0][:,1], 'v', label='Initial Centroids')
            # plot final centroids
            #plt.plot(centroids_per_iteration[len(centroids_per_iteration)-1][:, 0],
            #         centroids_per_iteration[len(centroids_per_iteration)-1][:, 1], '^', label='Final Centroids')
            #plt.legend(loc='upper left')
            plt.title('Clusters K={}'.format(k))
            plt.xlabel('X')
            plt.ylabel('Y')

    plt.show()