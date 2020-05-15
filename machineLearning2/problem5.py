#########################################################
# Problem 5
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


def WDKNN(distanceFunction, k, dataSet, instance):
    """
    This function determine label of input instance by
    using Weighted Distance K-Nearest Neighbour algorithm(WDKNN)
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

    # Find the class which has more weight
    # among k points
    label_occurrence = {}
    for ind, point in enumerate(knnPoints):

        # Calculate weighed of a class, use inverse of
        # distance measure for weighting
        if point_dis[ind][1] != 0:
            weight = 1/point_dis[ind][1]
        else:
            weight = 1/(point_dis[ind][1]+0.001)

        if point[len(point)-1] not in label_occurrence:
            label_occurrence[point[len(point)-1]] = weight
        else:
            label_occurrence[point[len(point)-1]] += weight

    labels = sorted(list(label_occurrence.items()),
                    key=lambda key_num: key_num[1],
                    reverse=True)

    # Return major label
    return labels[0][0]


def major_neighbour_of_a_minor(distanceFunction, dataSet, minor, k, majorClass):
    """
    This function gets an instace which belongs to
    minor class (class with lower number of instances)
    and calculates number of major instances in k neighbour of
    it
    :param distanceFunction: a function for measuring distance
            it can be eul
    :param dataSet: list of instances
    :param minor:   an instance of dataSet which belongs to
                minor class
    :param k:   how many of minor's neighbour should be considered
    :param majorClass: label of major class
    :return: number of major neighbour in k-neighbourhood
    """

    # Each element of this list contains a list
    # Which first element is a point of dataSet
    # and second element is its distance form
    # instance
    point_dis = []

    for point in dataSet:

        # Calculate distance between point and instance.
        # Last element of point is its label
        # omit it before
        dis = distanceFunction(minor, point[0:-1])

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

    # Look at k neighbours and count number of major
    # instances
    majorNum = 0
    for point in knnPoints:
        if point[len(point)-1] == majorClass:
            majorNum += 1

    # Return number of major neighbours
    return majorNum


def proposed_KNN(distanceFunction, k, dataSet, instance):
    """
    This function determine label of input instance by
    using a kind of K-Nearest Neighbour algorithm which is
    proposed in "An Improved KNN Algorithm Based on Minority
                    Class Distribution for Imbalanced Dataset"
    :param distanceFunction: Is a function that measure
            distance between two points, like euclidean,
             minkowski, cosine
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

    # classes and number of their members
    class_member = {}

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

        # count number of member of each class
        if point[len(point)-1] not in class_member:
            class_member[point[len(point)-1]] = 1
        else:
            class_member[point[len(point) - 1]] += 1

    # Find major class, class wchich has more instance
    major_class = max([(x[1], x[0]) for x in class_member.items()])[1]

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

    # Check there is a non major class instance or not
    onlyMajorInstace = True
    for point in knnPoints:
        if point[len(point)-1] != major_class:
            onlyMajorInstace = False
            break

    #weight of minor neighbours
    w_minor = []
    if onlyMajorInstace == True:
        # If there is no minor neighbours
        w_minor = [1] * k
    else:
        # if there is some minor instances calculate
        # there wight
        for point in knnPoints:
            if point[len(point) - 1] == major_class:
                w_minor.append(1)
            else:
                # find major neighbour of a minor instance
                majorNeighbour = major_neighbour_of_a_minor(
                    distanceFunction=distanceFunction,
                    dataSet=dataSet,
                    minor=point[0:-1],
                    k=k,
                    majorClass=major_class
                )

                # Calculate weight of minor neighbour according to
                # its major neighbour
                w_minor.append(1+(majorNeighbour/k))


    # Find the class which has more weight
    # among k points
    label_occurrence = {}
    for ind, point in enumerate(knnPoints):

        # Calculate weighed of a class, use inverse of
        # distance measure for weighting
        if point_dis[ind][1] != 0:
            weight = (1/point_dis[ind][1]) * w_minor[ind]
        else:
            weight = (1/(point_dis[ind][1]+0.001)) * w_minor[ind]

        if point[len(point)-1] not in label_occurrence:
            label_occurrence[point[len(point)-1]] = weight
        else:
            label_occurrence[point[len(point)-1]] += weight

    labels = sorted(list(label_occurrence.items()),
                    key=lambda key_num: key_num[1],
                    reverse=True)

    # Return major label
    return labels[0][0]


def five_fold_cross_validation(learningFunction, argumentOFLearningFunction, dataSet):
    """
    This function use 5-fold-cv to evaluate learningFunction
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
    :return: return average 5-fold cv errors {'F-measure': , 'G-mean': }
    """
    from math import floor, sqrt
    import copy

    # average error on 5 folds
    averageError = {'F-measure': 0, 'G-mean': 0}

    # calculate size of each fold
    foldsize = floor(len(dataSet)/5)

    # A list that contain 5 folds
    folds=[]

    # Divide dataSet to 5 fold
    for fold in range(5-1):
         folds.append(dataSet[fold*foldsize:(fold+1)*foldsize])
    folds.append(dataSet[(5-1) * foldsize::])

    # Train and test learning function with 5 different forms
    for index1, i in enumerate(folds):
        # Test contains fold[i]
        test = copy.deepcopy(i)
        # Train contains all folds except fold[i]
        train = []
        for index2, j in enumerate(folds):
            if index2 != index1:
                train = train + copy.deepcopy(j)

        # Evaluate performance of learningFunction
        foldError= {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for point in test:

            # Add point to argument of learningFunction
            argumentOFLearningFunction['instance'] = copy.deepcopy(point[0:-1])
            # Use Train as training data
            argumentOFLearningFunction['dataSet'] = copy.deepcopy(train)

            # Find label of point by learningFunction
            label = learningFunction(**argumentOFLearningFunction)

            # If it is misclassified add it to error
            if label == 1:
                if label == point[len(point)-1]:
                    foldError['TP'] += 1  # True positive
                else:
                    foldError['FP'] += 1  # false positive
            elif label == 0:
                if label == point[len(point) - 1]:
                    foldError['TN'] += 1  # True negative
                else:
                    foldError['FN'] += 1  # False negative
            else:
                print("Label of sample aren't correct!")
                exit()

        # calculate F-measure and G-mean
        precision = foldError['TP'] / (foldError['TP']+foldError['FP'])
        recall = foldError['TP'] / (foldError['TP'] + foldError['FN'])
        averageError['F-measure'] += (2 * recall * precision) / (recall + precision)

        if foldError['TN'] == 0 and foldError['FP'] == 0:
            foldError['TN'] = 1
        averageError['G-mean'] += sqrt(recall *
                                       (foldError['TN'])/(foldError['TN']+foldError['FP']))

    averageError['F-measure'] /= 5
    averageError['G-mean'] /= 5

    return averageError


#########################################################
# Problem 5-E
#########################################################
import random

print("It takes few seconds please wait, ...\n")

# Different dataSet which is mentioned in this paper:
# "An Improved KNN Algorithm Based on Minority
# Class Distribution for Imbalanced Dataset"
datasetName = ['yeast3.dat', 'ecoli3.dat',
               'yeast-2_vs_4.dat', 'yeast-0-3-5-9_vs_7-8.dat',
               'yeast-0-2-5-6_vs_3-7-8-9.dat', 'yeast-0-2-5-7-9_vs_3-6-8.dat',
               'ecoli-0-2-6-7_vs_3-5.dat']

# Result on WDKNN and proposed improved KNN
result = {'WDKNN':[], 'ImprovedKNN':[]}

# For different datasets calculate F-measure and G-mean
for dataSetPath in datasetName:
    # Read data set file, Each line is an observation that all
    # elements of line are features and last element is label
    dataSet = []
    with open(dataSetPath, 'r') as f:
        for line in f:
            # If first line contains @ ignore it
            if line[0] == '@':
                continue

            # Convert each observation in file to a numerical
            # list
            line = line.replace(' ','')
            lineSegment = line.split(',')
            observation = list(map(float, lineSegment[0:-1]))

            # label of sample
            if lineSegment[len(lineSegment) - 1].strip() == 'negative':
                observation.append(0)  # negative class(0)
            else:
                observation.append(1)  # positive class(1)

            # Add observation to dataSet
            dataSet.append(observation)

    # shuffle dataSet
    random.shuffle(dataSet)

    # Consider k neighbour
    k = 5

    # Calculate F-measure and g-mean for WDKNN
    five_fold_cv_error = \
        five_fold_cross_validation(learningFunction=WDKNN,
                                   argumentOFLearningFunction={'distanceFunction': euclidean_distance,
                                                               'k': k,
                                                               'dataSet': [],
                                                               'instance': []},
                                   dataSet=dataSet
                                   )
    five_fold_cv_error['F-measure'] = round(five_fold_cv_error['F-measure'], 4)
    five_fold_cv_error['G-mean'] = round(five_fold_cv_error['G-mean'], 4)
    result['WDKNN'].append((dataSetPath.split('.')[0], five_fold_cv_error))

    # Calculate F-measure and g-mean for proposed KNN
    five_fold_cv_error = \
        five_fold_cross_validation(learningFunction=proposed_KNN,
                                   argumentOFLearningFunction={'distanceFunction': euclidean_distance,
                                                               'k': k,
                                                               'dataSet': [],
                                                               'instance': []},
                                   dataSet=dataSet
                                   )

    five_fold_cv_error['F-measure'] = round(five_fold_cv_error['F-measure'],4)
    five_fold_cv_error['G-mean'] = round(five_fold_cv_error['G-mean'],4)
    result['ImprovedKNN'].append((dataSetPath.split('.')[0], five_fold_cv_error))

for key in result:
    print('=====================================')
    print(key,' :')
    print('F-measure - G-mean')
    for dataset_measure in result[key]:
        print(dataset_measure)
print('=====================================')





