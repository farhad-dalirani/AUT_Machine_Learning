#########################################################
# Problem 4
#########################################################


def euclidean_distance(x, y):
    """
    This function returns euclidean distance of two points
    :param x: d-dimensional point
    :param y: d-dimensional point
    :return: a scalar
    """
    from math import sqrt

    # Calculate euclidean distance
    distance = 0
    for xi, yi in zip(x, y):
        distance += (xi-yi) ** 2

    return sqrt(distance)


def manhattan_distance(x, y):
    """
    This function returns Manhattan distance of two points
    :param x: d-dimensional point
    :param y: d-dimensional point
    :return: a scalar
    """

    # Calculate Manhattan distance
    distance = 0
    for xi, yi in zip(x, y):
        distance += abs(xi-yi)

    return distance


def minkowski_distance(x, y, p):
    """
    This function returns Minkowski distance of two points
    :param x: d-dimensional point
    :param y: d-dimensional point
    :param p:
    :return: a scalar
    """

    # Calculate Minkowski distance
    distance = 0
    for xi, yi in zip(x, y):
        distance += abs(xi-yi) ** p

    return distance ** (1/p)


def minkowski_distance_p4(x, y):
    """
       This function returns Minkowski distance of two points
       when p is 4
       :param x: d-dimensional point
       :param y: d-dimensional point
       :return: a scalar
       """
    return minkowski_distance(x, y, 4)


def minkowski_distance_p_half(x, y):
    """
       This function returns Minkowski distance of two points
       when p is 1/2
       :param x: d-dimensional point
       :param y: d-dimensional point
       :return: a scalar
       """
    return minkowski_distance(x, y, 0.5)

def cosine_distance(x, y):
    """
    This function returns cosine distance of two points
    :param x: d-dimensional point
    :param y: d-dimensional point
    :return: a scalar
    """
    import numpy as np

    # Calculate Cosine distance
    distance = 1 - (np.dot(x,y))/(np.sqrt(np.dot(x,x))*np.sqrt(np.dot(y,y)))

    return distance


def KNN(distanceFunction, k, dataSet, instance):
    """
    This function determine label of input instance by
    using K-Nearest Neighbour algorithm
    :param distanceFunction: Is a function that measure
            distance between two points
    :param k: Determine how many nearest neighbours of
            a point should be consider
    :param dataSet: labeled data
    :return: return predicted label of instance
    """

    # Each element of this list contains a list
    # Which first element is a point of dataSet
    # and second element is its distance form
    # instance
    point_dis = []

    # Calculate distance of each point of dataSet
    # to instance. for calculating distance use
    # distanceFunction that can be euclidean,
    # Manhattan and ect.
    for point in dataSet:

        # Calculate distance between point and instance.
        # Last element of point is its label
        # omit it before
        dis = distanceFunction(instance, point[0:-1])

        # Add point and its distance to list
        point_dis.append([point, dis])

    # Sort points of dataSet from minimum to maximum
    # according to their distance from instance
    point_dis = sorted(point_dis, key=lambda p: p[1])

    # Just first k points which their distance to instance
    # are lower than other points
    point_dis = point_dis[0:k]

    # Just keep points and cut distance
    knnPoints = []
    for point in point_dis:
        knnPoints.append(point[0])

    # Find the class which has been occurred more than other
    # classes among k points
    label_occurrence = {}
    for point in knnPoints:
        if point[len(point)-1] not in label_occurrence:
            label_occurrence[point[len(point)-1]] = 1
        else:
            label_occurrence[point[len(point)-1]] += 1

    labels = sorted(list(label_occurrence.items()),
                    key=lambda key_num: key_num[1],
                    reverse=True)

    # Return major label
    return labels[0][0]



def ten_fold_cross_validation(learningFunction, argumentOFLearningFunction, dataSet):
    """
    This function use 10-fold-cv to evaluate learningFunction
    which can be KNN, and any other learning function.
    of dataSet
    :param learningFunction: Is a function that learns
            from data to predict label of new inputs.
            It can be any Machine Learning algorithms
            like KNN, decisionTree, SVM and ...
    :param argumentOFLearningFunction: is a list
        that contains necessary argument of
        learningFunction.
    :param dataSet: training data
    :return: return average 10-fold cv error
    """
    from math import floor
    import copy

    # average error on 10 folds
    averageError = 0

    # calculate size of each fold
    foldsize = floor(len(dataSet)/10)

    # A list that contain 10 folds
    folds=[]

    # Divide dataSet to ten fold
    for fold in range(9):
         folds.append(dataSet[fold*foldsize:(fold+1)*foldsize])
    folds.append(dataSet[(10-1) * foldsize::])

    # Train and test learning function with 10 different forms
    for index1, i in enumerate(folds):
        # Test contains fold[i]
        test = copy.deepcopy(i)
        # Train contains all folds except fold[i]
        train = []
        for index2, j in enumerate(folds):
            if index2 != index1:
                train = train + copy.deepcopy(j)

        # Evaluate performance of learningFunction
        foldError=0
        for point in test:

            # Add point to argument of learningFunction
            argumentOFLearningFunction['instance'] = copy.deepcopy(point[0:-1])
            # Use Train as training data
            argumentOFLearningFunction['dataSet'] = copy.deepcopy(train)

            # Find label of point by learningFunction
            label = learningFunction(**argumentOFLearningFunction)

            # If it is misclassified add it to error
            if label != point[len(point)-1]:
                foldError += 1

        averageError += foldError/len(test)

    averageError /= 10

    return averageError


#########################################################
# Problem 4-a
#########################################################
import matplotlib.pylab as plt
import random

# Read data set file, Each line is an observation that all
# elements of line are features and last element is label
from math import floor
dataSet = []
with open('seeds_dataset.txt', 'r') as f:
    for line in f:
        # Convert each observation in file to a numerical
        # list
        observation = list(map(float, line.split()))
        observation[len(observation)-1] = int(observation[len(observation)-1])

        # Add observation to dataSet
        dataSet.append(observation)

# shuffle dataSet
random.shuffle(dataSet)

# For different neighbour size calculate error
errors = []
differentK = [1, 3, 5, 7, 10]
for k in differentK:
    ten_fold_cv_error = \
        ten_fold_cross_validation(learningFunction=KNN,
                                  argumentOFLearningFunction={'distanceFunction': euclidean_distance,
                                                              'k': k,
                                                              'dataSet': [],
                                                              'instance': []},
                                  dataSet=dataSet
                                  )

    # Add error to list
    errors.append(ten_fold_cv_error)

print('=====================================')
print("Part A:")
for i in zip(differentK, errors):
    print('K={}, \tError={}'.format(i[0], i[1]))
print('=====================================')

plt.plot(differentK, errors, 'go-')
plt.xlabel('K')
plt.ylabel('10-Fold-CV Error')
plt.title('Problem 4, part A')


#########################################################
# Problem 4-b
#########################################################

# For different distance function calculate 10-fold-cv error
errors = []
differentDistance = [euclidean_distance, manhattan_distance,
                     minkowski_distance_p4, minkowski_distance_p_half,
                     cosine_distance]

for distanceFun in differentDistance:
    ten_fold_cv_error = \
        ten_fold_cross_validation(learningFunction=KNN,
                                  argumentOFLearningFunction={'distanceFunction': distanceFun,
                                                              'k': 5,
                                                              'dataSet': [],
                                                              'instance': []},
                                  dataSet=dataSet
                                  )

    # Add error to list
    errors.append(ten_fold_cv_error)


# Plot errors for different distance funciton
plt.figure()
nameOfFunction = list(map( lambda fun: fun.__name__,differentDistance))

print("Part B:")
for i in zip(nameOfFunction, errors):
    print('Distance Function={}, \tError={}'.format(i[0], i[1]))
print('=====================================')

for index, fun in enumerate(nameOfFunction):

    plt.plot( index, errors[index], 'o-', label=nameOfFunction[index])

plt.xlabel('Different Distance Function')
plt.ylabel('10-Fold-CV Error')
plt.legend()
plt.title('Problem 4, part B')


plt.show()
