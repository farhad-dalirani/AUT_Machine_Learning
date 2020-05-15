def ten_fold_cross_validation(dataSet):
    """
    This function use 10-fold-cv to evaluate learningFunction

    :param dataSet: training data
    :return: return average 10-fold cv acc
    """
    from math import floor
    import copy
    from sklearn.naive_bayes import GaussianNB
    import numpy as np

    # average error on 10 folds
    averageAcc = 0

    # calculate size of each fold
    foldsize = int(floor((len(dataSet))/10))
    #print foldsize

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
        yTrain = []
        for index2, j in enumerate(folds):
            if index2 != index1:
                train = train + copy.deepcopy(j[0:-1])

        for index2, j in enumerate(train):
            yTrain.append(j[len(j)-1])
            train[index2] = j[0:-1]

        clf = GaussianNB()
        clf.fit(train, yTrain)

        truePositiveAndNegative = 0
        for index,testcase in enumerate(test):
            if clf.predict([testcase[0:-1]]) == testcase[len(testcase)-1]:
                truePositiveAndNegative += 1


        averageAcc += (truePositiveAndNegative+0.0) / (len(test)+0.0)
        print '> ', (truePositiveAndNegative+0.0) / (len(test)+0.0)

    averageAcc /= 10.0

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
    import random

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

                if index < len(features)-1:
                    observation.append(attributes[index][feature] + random.random()/5)
                else:
                    if index != len(features)-1:
                        observation.append(attributes[index][feature])
                    else:
                        observation.append(feature)

        # add observation to training set
        train.append(cp.copy(observation))

    print(train)

    # Shuffle Data
    np.random.shuffle(train)

    # calculate accuracy 10 fold cross validation of NaiveBayes Classifier for cars dataset
    acc = ten_fold_cross_validation(dataSet=train)
    print("10-Fold-CV Accuracy: {}".format(acc))
    print("10-Fold-CV Error: {}".format(1-acc))

if __name__ == '__main__':
    print('Please wait, It takes several second...')
    runCode()