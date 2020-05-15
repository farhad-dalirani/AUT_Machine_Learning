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


def plot_k_nearest_neighbor_distances(X, k):
    """
    This Function finds K-Nearest-Neighbour for all points in dataset
    and calculates average distance of all points to their KNN.
    After that it plots sorted distances, form plot we can find out
    proper EPS for given MNP( MNP==K)
    :param X: Dataset, a numpy ndarray
    :param k: first k nearest neighbour
    :return: None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    all_knn_average_dist = []

    # for all point finds their k nearest neighbours
    for i in range(X.shape[0]):
        # Initial Distance array for i th point
        distance_matrix = np.zeros(X.shape[0])

        # Set distance of a point to itself infinity
        # for avoiding recognizing a point as its neighbours
        distance_matrix[i] = np.inf

        # Find distance of other points to i th point
        for j in range(X.shape[0]):
            if i == j:
                continue

            distance_matrix[j] = euclidean_distance(X[i], X[j])

        # sort distances of other points to i th point
        distance_matrix.sort()

        # Calculate average distance of iTh point to its KNN
        knn_average_distance = (distance_matrix[0:k].sum())/k

        all_knn_average_dist = all_knn_average_dist + [knn_average_distance]

    # Sort average distances
    all_knn_average_dist.sort()

    #plot average distances
    plt.figure()
    plt.plot(range(len(all_knn_average_dist)), all_knn_average_dist)
    plt.axhline(y=0.209, color='red', linestyle='--', label='y= 0.209')
    plt.xlabel('points(sorted by distance)')
    plt.ylabel('K-NN Average Distance')
    plt.legend()
    plt.title('MNP=K={}'.format(k))


def neighbours_of_points(X, eps):
    """
    This function find neighbours for all points. Neighbours of a point
    are points that their distance to the point is less equal to EPS
    :param X: a numpy array which shape m*d. m point each d dimensions.
    :param eps: is radios of a circle centered at a point which determines
                neighbourhood of the point.
    :return: a list of lists, each internal list determines indexes of
            neighbours of a point
    """
    import numpy as np

    # initial list of neighbours for all points of dataset
    neighbours = [[] for _ in range(X.shape[0])]

    # for all point finds their neighbours
    for i in range(X.shape[0]):
        # Initial Distance Matrix
        distance_matrix = np.zeros(X.shape[0])

        # Set distance of a point to itself infinity
        # for avoiding recognizing a point as its neighbours
        distance_matrix[i] = np.inf

        # Find distance of other points to i th point
        for j in range(X.shape[0]):
            if i == j:
                continue

            distance_matrix[j] = euclidean_distance(X[i], X[j])

        # Find neighbours which their distances are equal less than eps
        neighbours_i = np.argwhere(distance_matrix <= eps)
        neighbours_i = neighbours_i.T.tolist()[0]

        # saves neighbours of i th point of dataset
        neighbours[i] = np.copy(neighbours_i).tolist()

    return neighbours


def is_core(neighbours, MNP):
    """
    This function determines points of datasets are core or boundary
    :param neighbours:a list of lists, each internal list determines indexes of
            neighbours of a point
    :param MNP: Minimum number of neighbours that makes a point a core
    :return:
    """
    _isCore = []
    for index, i_th_point in enumerate(neighbours):
        if len(i_th_point) >= MNP:
            _isCore.append(True)
        else:
            _isCore.append(False)

    return _isCore


def DBScan(X, eps, MNP):
    """
    DBSCAN Clustering
    :param X: a numpy array which shape m*d. m point each d dimensions.
    :param eps:  is radios of a circle centered at a point which determine
                neighbourhood of the point.
    :param MNP: Minimum number of neighbours that makes a point a core
    :return: It returns a pair: (clusters, Outliers/Noise)
    """
    from collections import deque

    MNP -= 1

    # Find neighbours of each node
    neighbours = neighbours_of_points(X=data, eps=eps)

    # Determine a node is core or boundary
    isCore = is_core(neighbours=neighbours, MNP=MNP)

    # A list that shows a point has been seen or not
    isSeen = [False for _ in range(X.shape[0])]

    # Select an unseen point, iterate its neighbours and
    # do the same for neighbours and do it
    # until all possible nodes are seen,
    # if there exist a core points create a cluster

    # clusters
    clusters = []

    # points that palce in no clusters
    no_cluster = np.array([], dtype=int)

    for i_th in range(X.shape[0]):
        # If point was seen before, ignore it
        if isSeen[i_th] == True:
            continue

        # candidate clusters
        candid_clus = []

        # If there is a point that is core, assign positive
        # value to this variable, It is used for creating a cluster
        isAnyCore = False

        # a queue for iterating reachable points
        queue = deque()

        queue.append(i_th)
        while len(queue) > 0:
            # pop a index of a point from queue
            index_point = queue.popleft()

            # see the point
            isSeen[index_point] = True

            if isCore[index_point] == True:
                isAnyCore = True

            # add point to temporary cluster
            candid_clus.append(index_point)

            # add unseen neighbour to queue
            for index_neighbour in neighbours[index_point]:
                if isSeen[index_neighbour] == True:
                    continue
                queue.append(index_neighbour)

        # if there is a core point create a cluster, otherwise those point
        # place in no cluster
        if isAnyCore == True:
            clusters.append(np.copy(candid_clus))
        else:
            no_cluster = np.hstack((no_cluster, candid_clus))

    # Return clusters and points that doesn't belong to any cluster
    return clusters, no_cluster


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
    for index, point in enumerate(data):
        plt.text(point[0], point[1], '{},'.format(index))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Input Data')

    #
    plot_k_nearest_neighbor_distances(X=data, k=3)

    eps = 0.209
    MNP = 3
    print('EPS={}, MNP={}'.format(eps, MNP))

    # Do DBSCAN
    clusters, no_cluster = DBScan(X=data, eps=eps, MNP=MNP)

    # plot clusters
    plt.figure()
    for i, cluster in enumerate(clusters):
        plt.plot(data[cluster][:, 0], data[cluster][:, 1], 'o', label='Cluster {}'.format(i + 1))

    # outlier-noise
    if no_cluster.shape[0] > 0:
        plt.plot(data[no_cluster][:, 0], data[no_cluster][:, 1], '^', label='Outlier|Noise')

    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clusters, EPS={}, MNP={}'.format(eps, MNP))

    plt.show()