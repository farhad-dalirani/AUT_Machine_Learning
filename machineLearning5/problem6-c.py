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


def mean(average_dis1, num_of_distance1, average_dis2, num_of_distance2, num_of_distances):
    """
    Calculate average distance of two clusters
    :param averge_dis1: average distance of cluster 1
    :param num_of_distance1: number that we divided total distance by it, and
                                obtained average distance
    :param averge_dis2: average distance of cluster 2
    :param num_of_distance2: number that we divided total distance by it, and
                                obtained average distance
    :param num_of_distances: number of point in cluster 1 *  number of point in cluster 2
    :return: average distance of two clusters
    """
    return (average_dis1*num_of_distance1 + average_dis2*num_of_distance2)/num_of_distances


def create_distance_matrix(X):
    """
    This function creates a m*m matrix, which element i,j is
    distance between point i and j of dataset.
    :param X: Input dataset, a m*d numpy ndarray, m observations, each d features
    :return: distance matrix, headers of columns, number of distances
    """
    import numpy as np

    # Initial distance matrix
    distace_matrix = np.zeros(shape=(X.shape[0], X.shape[0]))

    # number of distances, is used for average link
    num_dis = np.ones(shape=(X.shape[0], X.shape[0]))

    # header of distance matrix
    headers = [[i] for i in range(X.shape[0])]


    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i != j:
                distace_matrix[i][j] = euclidean_distance(X[i], X[j])
            else:
                distace_matrix[i][j] = np.inf

    return distace_matrix, headers, num_dis


def update_distance_matrix(distance_matrix, headers, num_dis, distance_function, p, q):
    """
    This function gets distance matrix and deletes rows p and q,
    also it deletes columns p and q.
    After omitting rows and columns, it updates headers.
    Then it adds row and column {p, q} which is obtained by merging
    clusters corresponding p and q. Finally it calculates distance of this new cluster
    to other cluster
    :param distance_matrix:  A m*m matrix, which element i,j is
                distance between cluster i and j
    :param headers: cluster corresponding to each row and col
    :param num_dis: this matrix used when we want to calculate average link
                    it holds n1*n2 in (1/(n1*n2)) * sum of all distances of two clusters
    :param distance_function: can be numpy.max, numpy.min, numpy.mean function and etc.
    :param p: row and column p
    :param q: row and column q
    :return: new distance matrix and headers
    """
    import numpy as np
    import copy as cp

    # Create a (m-1)*(m-1) matrix
    new_distance_matrix = np.zeros(shape=(distance_matrix.shape[0]-1, distance_matrix.shape[0]-1))

    # This use for average link, it contains number of distances between all two clusters
    if distance_function.__name__ == 'mean':
        new_num_dis = np.ones(shape=(distance_matrix.shape[0] - 1, distance_matrix.shape[0] - 1))

    # create new headers by deleting header of columns i,j and adding header union(header i, header j)
    new_headers = []
    for i in range(len(headers)):
        if i == p or i == q:
            continue
        new_headers.append(cp.copy(headers[i]))
    new_headers.append(headers[p]+headers[q])

    # Delete rows p and q, columns p and q
    index_i = 0
    index_j = 0
    for i in range(distance_matrix.shape[0]):
        if i == p or i == q:
            continue

        index_j = 0
        for j in range(distance_matrix.shape[0]):
            if j == p or j == q:
                continue
            new_distance_matrix[index_i][index_j] = distance_matrix[i][j]
            if distance_function.__name__ == 'mean':
                new_num_dis[index_i][index_j] = num_dis[i][j]
            index_j += 1

        index_i += 1

    # Calculate distance of new merged cluster to other clusters
    index_i = 0
    index_j = new_distance_matrix.shape[0] - 1
    for i in range(distance_matrix.shape[0]):
        if i == p or i == q:
            continue
        # Use distance function(min, max, average,...) for finding distance of
        # new merged cluster to other clusters
        if distance_function.__name__ != 'mean':
            distance_new_cluster = distance_function([distance_matrix[p][i], distance_matrix[q][i]])
        else:
            distance_new_cluster = distance_function(distance_matrix[p][i], num_dis[p][i],
                                                     distance_matrix[q][i], num_dis[q][i],
                                                     (len(headers[p])+len(headers[q]))*len(headers[i]))

        new_distance_matrix[index_i][index_j] = distance_new_cluster
        new_distance_matrix[index_j][index_i] = distance_new_cluster

        if distance_function.__name__ == 'mean':
            new_num_dis[index_i][index_j] = (len(headers[p])+len(headers[q]))*len(headers[i])
            new_num_dis[index_j][index_i] = (len(headers[p])+len(headers[q]))*len(headers[i])
        # move index
        index_i += 1

    # Set distance of new merged matrix to itself equal to infinity
    new_distance_matrix[index_j][index_j] = np.inf

    if distance_function.__name__ == 'mean':
        new_num_dis[index_j][index_j] = 1

    if distance_function.__name__ == 'mean':
        return new_distance_matrix, new_headers, new_num_dis
    else:
        return new_distance_matrix, new_headers, num_dis


def agglomerative_clustering(X, distance_function, begin_level_dendrogram, end_level_dendrogram):
    """
    This function does agglomerative clustering, depends on distance_function, it is
    distance_function==np.min ==> single link
    distance_function==np.max ==> complete link
    distance_function==mean ==> average link
    distance_function==arbitrary function  ==> arbitrary link

    :param X: Input dataset, a m*d numpy ndarray, m observations d features
    :param distance_function: for selecting distance between two clusters
            can be man, mean, average(mean), ...
    :param begin_level_dendrogram, end_level_dendrogram: it returns clusters for different levels of
            dendrogram which are between thoese input argument
    :return: clusters at different levels of dendrogram. it return a dictionary which key i is
            equal clusters at level i.
    """
    import numpy as np

    # A dictionary which key i is equal clusters at level i.
    dendrogram = {}

    # Initial distance matrix, at the beginning each point in
    # dataset is a cluster
    # headers is cluster at different levels of dendrogram
    distance_matrix, headers, num_dis = create_distance_matrix(X)

    if distance_function.__name__ != 'mean':
        num_dis = None

    if (X.shape[0]-1) >= begin_level_dendrogram and (X.shape[0]-1) <= end_level_dendrogram:
        dendrogram[0] = [np.array(cluster) for cluster in headers]

    # For each level of dendrogram find clusters,
    # at each step find two most similar clusters and
    # merge them.(number of levels of dendrogram is equal number of points
    # mines one)
    for height in range(X.shape[0]-1):

        print('Calculating dendrogram at level {} ...'.format((X.shape[0]-1)-height))
        # find smallest value in distance matrix(two most similar
        # clusters)
        min_dist = np.inf
        min_ind = (None, None)
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                if i == j:
                    continue
                if distance_matrix[i][j] < min_dist:
                    min_dist = distance_matrix[i][j]
                    min_ind = [i, j]

        # delete clusters two most similar cluster x and y from distance matrix, merge
        # clusters x and y then and new merged cluster to distance matrix
        distance_matrix, headers, num_dis = update_distance_matrix(distance_matrix=distance_matrix, headers=headers, num_dis=num_dis,
                                                 distance_function=distance_function, p=min_ind[0], q=min_ind[1])

        current_level = (X.shape[0]-1)-height
        if current_level >= begin_level_dendrogram and current_level <= end_level_dendrogram:
            dendrogram[current_level] = [np.array(cluster) for cluster in headers]

    # return clusters at different level of dendrogram
    return dendrogram


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

    #X=np.array([[0,0],[1,1],[2,2],[3,3]])

    algorithm_name={'amin':'Single Link', 'amax':'Complete Link', 'mean': 'Average Link'}

    # for different distance functions calculates
    # dendrogram (do agglomerative clustring)
    for distance_function in [np.min, np.max, mean]:

        # Calculate dendrogram
        dendrogram = agglomerative_clustering(X=X, distance_function=distance_function, begin_level_dendrogram=0, end_level_dendrogram=10)

        for key in dendrogram:
            # plot clusters
            plt.figure()
            for i, cluster_index in enumerate(dendrogram[key]):
                plt.plot(X[cluster_index][:, 0], X[cluster_index][:, 1], 'o', label='Cluster {}'.format(i + 1))
            print('Plotting level {}  of dendrogram .({}) ...'.format(key, algorithm_name[distance_function.__name__]))
            plt.title('Level {} of dendrogram ({})'.format(key, algorithm_name[distance_function.__name__]))

    plt.show()
