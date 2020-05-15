def find_priors(train):
    """
    find prior of classes
    :param train:
    :return: {0:prior of y=0,..., n:prior of y=n}
    """
    priors = {}
    n = 0
    for observation in train:
        if observation[len(observation)-1] in priors:
            priors[observation[len(observation)-1]] += 1
        else:
            priors[observation[len(observation)-1]] = 1
        n += 1

    for key in priors:
        priors[key] = priors[key]/n

    # return priors
    return priors


def probabilities(train, attributes):
    """
    Find probabilities for p(feature| y=i) for all features
    :param train: training dataset
    :param attributes: a list of dictionaries that contains
            name of different categorical features and classes
    :return: is a list that contains probabilities of each feature that
         shows p(feature i| y=j) for all features and classes.
        format of list is:
        [class 0{feature1: prob feature given class 0,..., featureN: prob feature given class 0},
        ...,
        class N{feature1: prob feature given class N,..., featureN: prob feature given class N}
        ]
    """
    import numpy as np

    if len(train) == 0:
        print('Input train dataset has problem!')
        exit(3)

    # this list contains probabilities of each feature that
    # shows p(feature i| y=j) for all features and classes.
    # format of list is:
    #[class 0{feature1: #,..., featureN: #},
    # ...,
    # class N{feature1: #,..., featureN: #}
    #]
    # initial list that contains n dictionaries for
    # each class
    probabilitiesOfFeatures = []
    for _class in range(len(attributes[6])):
        probabilitiesOfFeatures.append({})

    #this dictionary contains number of elements of each class
    numberOfEachClass = {}

    # For each feature calculate p(feature| y=i) for all features
    for observation in train:
        for index,feature in enumerate(observation[0:-1]):

            # count number of features given class
            if feature in probabilitiesOfFeatures[observation[len(observation)-1]]:
                probabilitiesOfFeatures[observation[len(observation)-1]][feature] += 1
            else:
                probabilitiesOfFeatures[observation[len(observation)-1]][feature] = 1

            # count number of elements of each class
            if observation[len(observation)-1] in numberOfEachClass:
                numberOfEachClass[observation[len(observation)-1]] += 1
            else:
                numberOfEachClass[observation[len(observation)-1]] = 1

    # divide x given y by size of y
    for index, classes in enumerate(probabilitiesOfFeatures):
        for feature in classes:
            probabilitiesOfFeatures[index][feature] /= numberOfEachClass[index]

    # return probabilities of features given classes
    return probabilitiesOfFeatures


def predictClassOfInstance(probabilitiesOfFeatures, instance, priors):
    """
    This function predict class(1 or -1) of an input instance
    :param probabilitiesOfFeatures:
        is a list that contains probabilities of each feature that
         shows p(feature i| y=j) for all features and classes.
        format of list is:
        [class 0{feature1: prob feature given class 0,..., featureN: prob feature given class 0},
        ...,
        class N{feature1: prob feature given class N,..., featureN: prob feature given class N}
        ]
    :param instance: an input instance we want to calculate its class
    :param priors: is a dictionationay which contains priors of different
            classes {0:prior class 0,...,1:prior classN}
    :return: class of input instance
    """
    import numpy as np

    # Initial posteriors
    p_yi = {}
    for key in priors:
        p_yi[key] = np.log(priors[key])

    for feature in range(0, len(instance)-1):

        # Calculate value of instance[feature] in
        # distribution p(feature|y=i) for all classes
        for _class in range(len(probabilitiesOfFeatures)):
            p_yi[_class] += np.log(probabilitiesOfFeatures[_class][instance[feature]])


    # Determine class of instance
    #print(p_yi)
    return max(p_yi, key=lambda key: p_yi[key])


def ten_fold_cross_validation(learningFunction, dataSet, attributes):
    """
    This function use 10-fold-cv to evaluate learningFunction
    which can be NaiveBayes,KNN, and any other learning function.
    of dataSet
    :param learningFunction: Is a function that learns
            from data to predict label of new inputs.
            It can be any Machine Learning algorithms
            like NaiveBayes, KNN, decisionTree, SVM and ...
    :param dataSet: training data
    :param attributes: a list of dictionaries that contains
            name of different categorical features and classes
    :return: return average 10-fold cv acc
    """
    from math import floor
    import copy

    # average error on 10 folds
    averageAcc = 0

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


        # calculate priors of classes
        priors = find_priors(train)
        # print(priors)

        # calculate probabilities for feature|y=i for all feature
        probabilitiesOfFeatures = probabilities(train=train, attributes=attributes)
        # print(probabilitiesOfFeatures)

        # predict class of test data
        truePositiveAndNegative = 0
        for instance in test:
            predictedClass = predictClassOfInstance(probabilitiesOfFeatures=probabilitiesOfFeatures,
                                                        instance=instance,
                                                        priors=priors)
            # print('> ', predictedClass, ' -- ',instance)
            if predictedClass == instance[len(instance) - 1]:
                truePositiveAndNegative += 1

        averageAcc += truePositiveAndNegative / len(test)

    averageAcc /= 10

    return averageAcc


def runCode():
    """
    Main function which I read inputs and calls
    different function for training and testing
    :return: none
    """
    import scipy.io
    import numpy as np
    import matplotlib.pylab as plt
    import copy as cp

    # for working with attributes and classes
    attributes = [{'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},     #buying attribute
                 {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},      #maint
                 {'2': 0, '3': 1,'4': 2, '5more': 3},              #doors
                 {'2': 0, '4': 1, 'more': 2},                      #person
                 {'small': 0, 'med': 1, 'big': 2},                 #lug_boot
                 {'low': 0, 'med': 1, 'high': 2},                  #safety
                 {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}     #classes
                 ]

    # Read Train file: car.data
    # Read Train
    trainFile = open('car.data','r')
    train = []
    # read each line of dataset
    for line in trainFile:
        # remove white spaces
        line = line.replace(' ', '')
        line = line.replace('\t', '')
        line = line.replace('\n', '')
        # split line by ',' for seperating features
        features = line.split(',')

        # each line of dataset is an observation
        # [feature1,...,featureN, class i]
        observation = []
        for index, feature in enumerate(features):
            if not bool(attributes[index]):
                # for continuous features
                observation.append(int(feature))
            else:
                # for categorical features
                observation.append(attributes[index][feature])

        # add observation to training set
        train.append(cp.copy(observation))

    #print(train)

    # Shuffle Data
    np.random.shuffle(train)

    # calculate accuracy 10 fold cross validation of NaiveBayes Classifier for cars dataset
    acc = ten_fold_cross_validation(learningFunction=probabilities, dataSet=train, attributes=attributes)
    print("10-Fold-CV Accuracy: {}".format(acc))
    print("10-Fold-CV Error: {}".format(1-acc))

if __name__ == '__main__':
    print('Please wait, It takes several second...')
    runCode()
