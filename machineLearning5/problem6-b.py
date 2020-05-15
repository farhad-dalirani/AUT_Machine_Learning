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
    This fundtion calculates SSE (Sum Of Squared Error), for measuring quality of clusters
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

    # Randomly choose k centroids
    random_indexes = np.random.choice(a=range(X.shape[0]), size=k, replace=False)
    centroids = X[random_indexes]

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


def top_down(X, n=100):
    """
    This function uses a top-down method, in each step it divides data to two cluster
    with k-means (k=2), then it selects a cluster with higher SSE, and it do the same
    on it, it continue to breaking until it reach a cluster with one point or it reach
    the limit of dividing steps(input argument 'n') or difference between SSE of to new
    cluster become low.
    :param X: Input dataset
    :param n: limit of number of dividing that is done of dataset
    :return: clusters
    """
    import numpy as np

    points = np.copy(X)
    final_clusters = []

    first_time = True

    for i in range(n):

        # if one or zero point is left, stop clustering
        if points.shape[0] < 2:
            final_clusters.append(np.copy(points))
            points = np.array([])
            break

        # if two new clusters have close SSE, stop dividing
        if first_time != True:
            if abs(SSE_0-SSE_1) < 0.2 * np.max([SSE_0, SSE_1]):
                final_clusters.append(np.copy(points))
                points = np.array([])
                break

        first_time = False


        # Call k-means for k=2
        while True:
            sse_errors, clusters, centroids_per_iteration = kmeans(X=points, k=2)
            if clusters != None:
                break

        # Calculate SSE for both cluster
        SSE_0 = sum_of_squared_error([clusters[0]])
        SSE_1 = sum_of_squared_error([clusters[1]])
        #SSE_0 = clusters[0].shape[0]
        #SSE_1 = clusters[1].shape[0]

        # Cluster with higher SSE is selected for next round and applying
        # K-means on it, cluster with lower SSE is added to clusters
        print("SEE0: {},  SS1: {}".format(SSE_0, SSE_1))
        if SSE_0 > SSE_1:
            final_clusters.append(np.copy(clusters[1]))
            points = np.copy(clusters[0])
        else:
            final_clusters.append(np.copy(clusters[0]))
            points = np.copy(clusters[1])

        # Plot clusters
        plt.figure()
        for i_th, cluster in enumerate(final_clusters):
            plt.scatter(cluster[:, 0], cluster[:, 1], label='Cluster {}'.format(i_th + 1), marker='o')
        plt.scatter(points[:, 0], points[:, 1], label='Cluster {} (Divide Next Step)'.format(len(final_clusters)+1), marker='^')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Clusters at step {}'.format(i+1))
        plt.legend(loc='upper left')

    # if after n step, some points are left, add them to clusters as a cluster
    if points.shape[0] != 0:
        final_clusters.append(np.copy(points))

    # return cluster
    return final_clusters


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Read Input file
    dataset = pd.read_csv('data2.csv', header=None)

    # Points that we want to cluster them
    X = dataset.values

    # Plot Data
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Input Data')

    clusters = top_down(X=X)

    # Plot clusters
    plt.figure()
    #plt.subplot(121)
    for i_th, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], label='Cluster {}'.format(i_th+1))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Final Clusters')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(loc='upper left')

    plt.show()