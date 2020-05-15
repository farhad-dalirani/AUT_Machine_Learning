def find_priors(train):
    """
    find prior of class y=1 and y=-1
    :param train:
    :return: {-1:prior of y=-1, 1:prior of y=1}
    """
    priors = {-1:0, 1:0}
    n = 0
    for observation in train:
        if observation[len(observation)-1] == 1:
            priors[1] += 1
        else:
            priors[-1] += 1
        n += 1

    priors[1] /= n
    priors[-1] /= n

    # return priors
    return priors


def find_distributions(train):
    """
    Find normal distribution for p(feature| y=1)
    and p(feature| y=-1) for all features
    :param train: training dataset
    :return: a dictionary in form:
    {-1:{feature1:(mean,variance),...,featureN:(mean,variance)},
     1:{feature1:(mean,variance),...,featureN:(mean,variance)},
    }
    """
    import numpy as np

    # this dictionary contains estimated normal distribution
    # of p(feature| y=1) and p(feature| y=-1) for all features
    distributionsOfFeatures = {-1: {}, 1: {}}

    # For each feature calculate p(feature| y=1) and
    # p(feature| y=-1)
    for feature in range(0, np.shape(train)[1] - 1):
        featurelabel1 = []
        featurelabelmines1 = []

        # Iterate through train dataset for finding
        # where label of feature is 1 and where -1
        for observation in train:
            if observation[len(observation) - 1] == 1:
                featurelabel1.append(observation[feature])
            else:
                featurelabelmines1.append(observation[feature])

        # Estimate normal distribution of p(feature| y=1) and
        # p(feature| y=-1)
        featureMean1 = np.mean(featurelabel1, axis=0)
        featureVariance1 = np.cov(featurelabel1).tolist()

        featureMeanMines1 = np.mean(featurelabelmines1, axis=0)
        featureVarianceMines1 = np.cov(featurelabelmines1).tolist()

        # save normal distribution of feature
        distributionsOfFeatures[1][feature] = (featureMean1, featureVariance1)
        distributionsOfFeatures[-1][feature] = (featureMeanMines1, featureVarianceMines1)

    return distributionsOfFeatures


def univariate_normal(mean, variance, x):
    """
    value of a point in univariate normal distribution
    :param mean: mean of the normal distribution
    :param variance: variance of the normal distribution
    :param x: a point which we seek to find its probability in
            normal distribution
    :return:
    """
    import numpy as np
    if variance == 0:
        variance += 0.00001
    return (1 / (np.sqrt(2 * np.pi * variance)) *
            np.exp((-1 / (2 * variance)) * ((x - mean) ** 2)))


def predictClassOfInstance(distributionsOfFeatures, instance, threshold, prior1, priorMines1):
    """
    This function predict class(1 or -1) of an input instance
    :param distributionsOfFeatures: a dictionary in form:
        {-1:{feature1:(mean,variance),...,featureN:(mean,variance)},
         1:{feature1:(mean,variance),...,featureN:(mean,variance)},
        }
    :param instance: an input instance we want calculate its class
    :param threshold: if p(instance|y=1) > p(instance|y=-1) + threshold
            then instace belongs to class 1 else class -1
    :param prior1: prior of class y=1
    :param prior2: prior of class y=2
    :return:
    """
    import numpy as np

    # Initial posteriors
    p_y1 = np.log(prior1)
    p_yMines1 = np.log(priorMines1)

    for feature in range(0, len(instance)-1):
        # Calculate value of instance[feature] in
        # distribution p(feature|y=1) and p(feature|y=-1)
        p_y1 += np.log(univariate_normal(mean=distributionsOfFeatures[1][feature][0],
                                  variance=distributionsOfFeatures[1][feature][1],
                                  x=instance[feature]))
        p_yMines1 += np.log(univariate_normal(mean=distributionsOfFeatures[-1][feature][0],
                                  variance=distributionsOfFeatures[-1][feature][1],
                                  x=instance[feature]))
    #print('>',p_y1-p_yMines1)
    # Determine class of instance
    if p_y1 > p_yMines1 + threshold:
        return 1
    else:
        return -1


def runCode():
    """
    Main function which I read inputs and calls
    different function for training and testing
    :return: none
    """
    import scipy.io
    import numpy as np
    import matplotlib.pylab as plt

    # Read Train and Test file. which are .mat files
    # Read Train
    mat = scipy.io.loadmat('Train_data.mat')
    train = mat['train']
    # Shuffle Data
    np.random.shuffle(train)

    # Read Test
    mat = scipy.io.loadmat('Test_Data.mat')
    test = mat['test']
    # Shuffle Data
    np.random.shuffle(test)


    # calculate priors of y=1 and y=-1
    priors = find_priors(train)

    # calculate distributions for feature|y=1 and
    # feature | y = -1 for all feature
    distributionsOfFeatures = find_distributions(train=train)

    # predict class of test data
    truePositiveAndNegative = 0
    for instance in test:
        predictedClass = predictClassOfInstance(distributionsOfFeatures=distributionsOfFeatures,
                                                instance=instance,
                                                threshold=0,
                                                prior1=priors[1],
                                                priorMines1=priors[-1])

        if predictedClass == instance[len(instance)-1]:
            truePositiveAndNegative += 1

    # calculate accuracy and error
    accuracy = truePositiveAndNegative / np.shape(test)[0]
    error = 1 - accuracy

    print('Accuracy Test:{}, Error Test:{}'.format(accuracy,error))

    # predict class of test data
    truePositiveAndNegative = 0
    for instance in train:
        predictedClass = predictClassOfInstance(distributionsOfFeatures=distributionsOfFeatures,
                                                instance=instance,
                                                threshold=0,
                                                prior1=priors[1],
                                                priorMines1=priors[-1])

        if predictedClass == instance[len(instance) - 1]:
            truePositiveAndNegative += 1

    # calculate accuracy and error
    accuracy = truePositiveAndNegative / np.shape(train)[0]
    error = 1 - accuracy

    print('Accuracy Train:{}, Error Train:{}'.format(accuracy, error))

    # calculate ROC
    rocPoint = [(0,0),(1,1)]
    for threshold in [0, -20, 20, 50, -50, 100, -150]:
        # predict class of test data
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for instance in test:
            predictedClass = predictClassOfInstance(distributionsOfFeatures=distributionsOfFeatures,
                                                    instance=instance,
                                                    threshold=threshold,
                                                    prior1=priors[1],
                                                    priorMines1=priors[-1])

            if predictedClass == 1 and instance[len(instance) - 1] ==1:
                tp += 1
            if predictedClass == 1 and instance[len(instance) - 1] ==-1:
                fp += 1
            if predictedClass == -1 and instance[len(instance) - 1] ==1:
                fn += 1
            if predictedClass == -1 and instance[len(instance) - 1] ==-1:
                tn += 1


        # calculate true positive rate and false positive rate
        tpr = tp / (tp + fn)
        fpr = 1 - (tn/(tn+fp))
        rocPoint.append((fpr, tpr))

    rocPoint = sorted(rocPoint)
    plt.plot([fpr[0] for fpr in rocPoint], [fpr[1] for fpr in rocPoint], 'b-')
    plt.plot(np.linspace(start=0,stop=1,num=100), np.linspace(start=0,stop=1,num=100), 'r--')
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

    print(rocPoint)

if __name__ == '__main__':
    print('Please wait, It takes several second...')
    runCode()
