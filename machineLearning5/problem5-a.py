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


def davies_bouldin_index(clusters):
    """
    Davies Bouldin Index, for measuring quality of clusters
    :param clusters: a list numpy list which demonstrates
                    points in different clusters
    :return: a scalar number
    """
    import numpy as np

    # Centers of clusters
    c = [cluster.mean(axis=0) for cluster in clusters]

    # distance of points of a cluster to the center of cluster
    mu = []
    for i, cluster in enumerate(clusters):
        mu_temp = 0
        for point in cluster:
            # Calculate sum of distance of points to the center of cluster
            mu_temp += euclidean_distance(point, c[i])

        # Calculate average of distance of points to the center of cluster
        mu_temp /= cluster.shape[0]

        mu.append(mu_temp)

    mu = np.array(mu)

    # Calculate Davies Bouldin Index
    DB = 0
    for i in range(mu.shape[0]):

        # Calculate Rij = (MUi+Muj)/d(Ci, Cj)
        Rij = []
        for j in range(mu.shape[0]):
            if i != j:
                Rij.append((mu[i]+mu[j])/euclidean_distance(c[i], c[j]))
            else:
                Rij.append(-np.inf)

        Rij = np.array(Rij)
        #print(Rij,'<<<','i=',i)

        # find j that maximize Rij
        max_j = np.argmax(Rij)

        DB += Rij[max_j]

    DB /= len(clusters)
    return DB


def kmeans(X, k, max_iter=100):
    """
    This function is a K-means clustering algorithm
    :param X: Input dataset
    :param max_iter: maximum number of iterations that clustering algorithm
                    does to coverage
    :return: Return Davies Bouldin cost for each iteration, clusters, Centroids for each iteration
            if a cluster become empty, exit function by returning None, None, None
    """
    import numpy as np

    # Randomly choose k centroids
    centroids = np.random.randn(k, 2)

    centroids_per_iteration = [np.copy(centroids)]
    davies_bouldin_per_iteration = []
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
        davies_bouldin_per_iteration.append(davies_bouldin_index(clusters))
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
    return davies_bouldin_per_iteration, clusters, centroids_per_iteration


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Read Excel file
    df_data = pd.read_excel('data1.xlsx', header=None)

    # convert dataset from form (x, y1, y2) to form (x, y)
    data = np.vstack((df_data.iloc[:, [0,1]].values, df_data.iloc[:, [0,2]].values))

    # Plot data
    plt.plot(data[:, 0], data[:, 1], 'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Input Data')

    davies_bouldin_final_k = []
    davies_bouldin_per_iteration_k = []
    clusters_k = []
    centroids_per_iteration_k=[]

    # For different K do clustering
    for k in [2, 3, 4, 5]:
        while True:
            davies_bouldin_per_iteration, clusters, centroids_per_iteration = kmeans(X=data, k=k)
            if clusters != None:
                break

        # Store output of clustring with k cluster
        davies_bouldin_per_iteration_k.append(davies_bouldin_per_iteration)
        clusters_k.append(clusters)
        centroids_per_iteration_k.append(centroids_per_iteration)
        davies_bouldin_final_k.append(davies_bouldin_per_iteration[len(davies_bouldin_per_iteration)-1])

        # plot clusters
        plt.figure()
        for i, cluster in enumerate(clusters):
            plt.plot(cluster[:, 0], cluster[:, 1], 'o',label='Cluster {}'.format(i+1))
        # plot initial centroids
        plt.plot(centroids_per_iteration[0][:,0], centroids_per_iteration[0][:,1], 'v', label='Initial Centroids')
        # plot final centroids
        plt.plot(centroids_per_iteration[len(centroids_per_iteration)-1][:, 0],
                 centroids_per_iteration[len(centroids_per_iteration)-1][:, 1], '^', label='Final Centroids')
        plt.legend(loc='upper left')
        plt.title('Clusters K={}'.format(k))
        plt.xlabel('X')
        plt.ylabel('Y')


    # Print Davies Bouldin measure for all K
    for _ in range(4):
        print('DaviesBouldin Index when K={} equals to {}'.format(_+2, davies_bouldin_final_k[_]))


    # Find best K among [2, 3, 4, 5] according Davies Bouldin Index measure
    best_cluster = np.argmin(davies_bouldin_final_k)
    print('According Davies Bouldin Index Best K is: {}'.format(best_cluster+2))

    plt.figure()
    plt.plot(range(1, len(davies_bouldin_per_iteration_k[best_cluster]) + 1),
             davies_bouldin_per_iteration_k[best_cluster], 'b-')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Davies Bouldin Index [Log]')
    plt.title('Davies Bouldin Index during clustering when K={}'.format(best_cluster+2))

    # plot clusters
    plt.figure()
    for i, cluster in enumerate(clusters_k[best_cluster]):
        plt.plot(cluster[:, 0], cluster[:, 1], 'o', label='Cluster {}'.format(i + 1))
    # plot initial centroids
    plt.plot(centroids_per_iteration_k[best_cluster][0][:, 0], centroids_per_iteration_k[best_cluster][0][:, 1], 'v', label='Initial Centroids')
    # plot final centroids
    plt.plot(centroids_per_iteration_k[best_cluster][len(centroids_per_iteration_k[best_cluster]) - 1][:, 0],
             centroids_per_iteration_k[best_cluster][len(centroids_per_iteration_k[best_cluster]) - 1][:, 1], '^', label='Final Centroids')
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clusters for best K (K={})'.format(best_cluster+2))
    plt.show()