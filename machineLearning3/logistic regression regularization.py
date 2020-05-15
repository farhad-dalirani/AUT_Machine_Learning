def ten_fold_cross_validation(learningFunction, _lambda, xtrain, ytrain, threshold, iteration, learningRate):
    """
    This function use 10-fold-cv to evaluate learningFunction
    which can be logistic regression, and any other learning function.
    :param  learningFunction: Is a function that learns
            from data to predict label of new inputs.
            It can be any Machine Learning algorithms
            like KNN, decisionTree, SVM, LR and ...
    :param _lambda: parameter of regularization
    :param xtrain: training data
    :param ytrain: class of training data
    :param learningRate: learning rate of gradient decent
    :param iteration: maximum number of iteration for gradient decent
    :return: return average 10-fold cv error
    """
    from math import floor
    import copy
    import numpy as np

    # average error on 10 folds
    averageError = 0

    # calculate size of each fold
    foldsize = floor(np.shape(xtrain)[0]/10)

    # A list that contain 10 folds
    folds=[]
    yfolds=[]

    # Divide dataSet to ten fold
    for fold in range(9):
         folds.append(xtrain[fold*foldsize:(fold+1)*foldsize])
         yfolds.append(ytrain[fold * foldsize:(fold + 1) * foldsize])
    folds.append(xtrain[(10-1) * foldsize::])
    yfolds.append(ytrain[(10 - 1) * foldsize::])

    argumentOFLearningFunction = {}

    # Train and test learning function with 10 different forms
    for index1, i in enumerate(folds):
        # Test contains fold[i]
        test = copy.deepcopy(i)
        ytest = copy.deepcopy(yfolds[index1])

        # Train contains all folds except fold[i]
        train = np.ndarray([])
        yprimetrain = np.ndarray([])
        first = True
        for index2, j in enumerate(folds):
            if index2 != index1:
                if first == True:
                    first = False
                    train = copy.deepcopy(j)
                    yprimetrain = copy.deepcopy(yfolds[index2])
                else:
                    train = np.vstack((train, copy.deepcopy(j)))
                    #print('> ',yprimetrain,'\n>',copy.deepcopy(yfolds[index2]))
                    yprimetrain = np.concatenate((yprimetrain, copy.deepcopy(yfolds[index2])),axis=0)



        #xTrain, yTrain, numberOfIter, learningRate, xTest, yTest, _lambda
        # Add point to argument of learningFunction
        argumentOFLearningFunction['xTrain'] = copy.deepcopy(train)
        argumentOFLearningFunction['yTrain'] = copy.deepcopy(yprimetrain)
        argumentOFLearningFunction['numberOfIter'] = copy.deepcopy(iteration)
        argumentOFLearningFunction['learningRate'] = copy.deepcopy(learningRate)
        argumentOFLearningFunction['xTest'] = copy.deepcopy(test)
        argumentOFLearningFunction['yTest'] = copy.deepcopy(ytest)
        argumentOFLearningFunction['_lambda'] = copy.deepcopy(_lambda)
        argumentOFLearningFunction['threshold'] = copy.deepcopy(threshold)

        # learn parameter
        weight, scales, costOfTrainDataSet, accuracyOfTestDataSet, rateForRoc = \
            learningFunction(**argumentOFLearningFunction)

        averageError += (1-accuracyOfTestDataSet)

    averageError /= 10

    return averageError


def sigmoid(x):
    """
    Sigmoid function
    :param x:
    :return: 1/(1+e^(-x))
    """
    import numpy as np
    e = np.exp(-1 * x)

    if e != -1:
        return 1/(1 + e)
    else:
        return 1/0.0001


def cost_function(thetaVec, xMat, y, _lambda):
    """
    This function calculates cross Entropy Error
    :param thetaVec: weight vector
    :param xMat: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :return: return a scalar
    """
    import numpy as np

    # Initial output
    error = 0
    for i, x in enumerate(xMat):
        # Calculate cross entropy of h(x(i)) and y(i)
        sig = sigmoid(np.dot(x, thetaVec))
        if sig == 0 or sig == 1:
            sig += 0.0001
        signalError = -(y[i])*(np.log(sig)/np.log(2))\
                      -(1-y[i])*(np.log(1-sig)/np.log(2))

        # Add error of i th input to total error
        error += signalError

    # Calculate entropy cost function
    error = (1 / (len(xMat) * 2.0)) * error

    #print(np.sum([theta**2 for theta in thetaVec]))
    # regularization
    error += (_lambda/(len(xMat) * 2.0))*(np.sum([theta**2 for theta in thetaVec]))

    # Return error
    return error


def accuracy_of_test(thetaVec, xMat, y, threshold):
    """
    This function calculates accuracy of test: correct classifications/total
    :param thetaVec: weight vector
    :param xMat: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :param threshold: it's a cut-off for classification
    :return: return a scalar
    """
    import numpy as np

    # Initial output
    accuracy = 0
    for i, x in enumerate(xMat):
        # Calculate cross entropy of h(x(i)) and y(i)

        if sigmoid(np.dot(x, thetaVec)) >= threshold and y[i]==1:
            accuracy += 1
        elif sigmoid(np.dot(x, thetaVec)) < threshold and y[i]==0:
            accuracy +=1

    # Calculate 1-accuracy
    accuracy = accuracy / len(xMat)

    # Return accuracy
    return accuracy


def tpr_fpr_of_test(thetaVec, xMat, y, threshold):
    """
    This function calculates true positive rate and false positive rate
     of test
    :param thetaVec: weight vector
    :param xMat: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :param cut-off for classification
    :return: return a [False positive rate, True positive rate]
    """
    import numpy as np

    # Initial output
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, x in enumerate(xMat):
        # calculate tp, fp, tn, fn
        sig = sigmoid(np.dot(x, thetaVec))
        if  sig >= threshold and y[i]==1:
            tp += 1
        if sig >= threshold and y[i]==0:
            fp +=1
        if sig < threshold and y[i]==0:
            tn += 1
        if sig < threshold and y[i]==1:
            fn +=1

    # True Positive Rate
    if tp!=0:
        TPR = tp / (tp+fn)
    else:
        TPR = 0

    # false Positive Rate
    if tn != 0:
        FPR = 1- tn/(tn+fp)
    else:
        FPR = 1

    # Return
    return [FPR,TPR]


def gradients(thetaVec, xMat, y, _lambda):
    """
    This function calculates gradient of cost function
    :param thetaVec: weight vector
    :param xMat: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :return: return a vector which i th element of it, is derivative of
            cost function on theta[i]
    """
    import numpy as np

    # Initial output
    newTheta = [0] * len(thetaVec)

    for i, x in enumerate(xMat):
        # Calculate difference of h(x) and y
        signalError = (sigmoid(np.dot(x, thetaVec)) - y[i]) / (len(xMat) * 1.0)

        # Update derivatives of cost function for all
        # parameters. d J(theta[0],theta[1], ...,theta(n)]/d(theta[index]
        for index, theta in enumerate(thetaVec):
            newTheta[index] += signalError * x[index]
            # Consider regularization
            if index != 0:
                newTheta[index] += _lambda/(len(xMat)*1.0) * (thetaVec[index])

    # return derivatives of cost function
    return newTheta


def gradient_descent(xMat, y, numberOfIter, learningRate, _lambda):
    """
    This function use gradient descent to minimize cost function j
    over theta parameters.
    :param xMat: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :param numberOfIter: This argument determines number of iteration
            that gradient descent allowed to do for minimizing cost function
    :param: indicates learningRate
    :return: return Theta vector and a list of values of cost function
            in each iteration
    """
    import numpy as np

    # Randomly, initial theta vector, Use normanl distribution(0,1)
    # for choosing weight independently.
    #thetaVec = np.random.normal(loc=0, scale=1, size=len(xMat[0]))
    thetaVec = [0] * len(xMat[0])

    # values of cost function in each iteration
    iterCost = [cost_function(xMat=xMat, thetaVec=thetaVec, y=y, _lambda= _lambda)]

    # In each iteration update weight vector
    for iter in range(numberOfIter):
        # Calculate gradients
        gradientsOfthetaVec = gradients(thetaVec=thetaVec, xMat=xMat, y=y, _lambda= _lambda)
        # Update weights
        for index, theta in enumerate(thetaVec):
            thetaVec[index] = theta - learningRate * gradientsOfthetaVec[index]
        #print thetaVec,'*'

        # Update learning rate
        #learningRate = learningRate * 0.95
        #print np.sqrt(np.dot(gradientsOfthetaVec,gradientsOfthetaVec))

        # Add value of cost function to list of weight
        iterCost.append(cost_function(xMat=xMat, thetaVec=thetaVec, y=y,_lambda=_lambda))

    # Return list of weight and costs in each iteration
    #print thetaVec
    return thetaVec, iterCost


def logistic_gradient_descent(xTrain, yTrain, numberOfIter, learningRate, xTest, yTest, _lambda, threshold):
    """
    This function use gradient descent to minimize cost function j
    over theta parameters. Before starting fradient descent it does
    a nonlinear transformation.
    :param xTrain: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :param numberOfIter: This argument determines number of iteration
            that gradient descent allowed to do for minimizing cost function
    :param learningRate: indicates learning rate
    :param xTest, yTest: for plotting
    :param threshold: cut-off of logistic regression
    :return: Return weight [Theta0,...,Theta(n)], scale of each feature, cost on train, cost on test
    """
    import matplotlib.pylab as plt
    import numpy as np

    x_train = np.copy(xTrain)
    x_test = np.copy(xTest)

    x_train = np.transpose(x_train)
    x_test = np.transpose(x_test)

    # Find mean, min, max for normalization each feature
    scales = {}
    for index, feature in enumerate(x_train):
        meanOfFeature = np.mean(feature)
        maxOfFeature = np.max(feature)
        minOfFeature = np.min(feature)
        scales[index] = (meanOfFeature, minOfFeature, maxOfFeature)

        if index == 0:
            continue

        # Normalize each feature of train set and test set
        for i, element in enumerate(x_train[index]):
            if maxOfFeature != minOfFeature:
                x_train[index][i] = (element - meanOfFeature) / (1.0 * (maxOfFeature - minOfFeature))
            else:
                x_train[index][i] = (element - meanOfFeature) / 1.0

        for i, element in enumerate(x_test[index]):
            if maxOfFeature != minOfFeature:
                x_test[index][i] = (element - meanOfFeature) / (1.0 * (maxOfFeature - minOfFeature))
            else:
                x_test[index][i] = (element - meanOfFeature) / 1.0

    # After transpose each row is point and each column is a feature
    x_train = np.transpose(x_train)
    x_test = np.transpose(x_test)

    # Do gradient Descent
    weight, errors = gradient_descent(xMat=x_train, y=yTrain, numberOfIter=numberOfIter,
                                      learningRate=learningRate, _lambda=_lambda)

    #####################
    # Plot error rate for each Iteration
    #plt.plot(errors, 'b--')
    #plt.title('Cost Function during training:\n Iteration: {},Alpha: {}, Lambda: {}'.format(
    #            numberOfIter, learningRate, _lambda))
    #plt.ylabel('Cross Entropy')
    #plt.xlabel('Each step during training')
    ############
    #plt.show()
    ############
    ############
    #####################

    # Cost function value for train data set
    costOfTrainDataSet = errors[ len(errors)-1]
    # Cost function value for test data set
    accuracyOfTestDataSet = accuracy_of_test(thetaVec=weight,xMat=x_test,y=yTest, threshold=threshold)

    #TPR, FPR
    rateForROC = tpr_fpr_of_test(thetaVec=weight,xMat=x_test,y=yTest, threshold=threshold)

    # Return weight, scale of each feature, cost on train, accuracy on test
    #print('>', costOfTrainDataSet, accuracyOfTestDataSet)
    return weight,scales, costOfTrainDataSet, accuracyOfTestDataSet,rateForROC


def runCode():
    """
    Main function which I read inputs and calls
    different function for training and testing
    :return: none
    """
    import scipy.io
    import numpy as np
    import matplotlib.pylab as plt

    numberOfIteration = 100
    learningRate = 0.5

    # Read Train and Test file. which are .mat files
    # Read Train
    mat = scipy.io.loadmat('Train_data.mat')
    train = mat['train']
    # Shuffle Data
    np.random.shuffle(train)

    # Separate Label from train
    train = np.transpose(train)
    yTrain = train[len(train)-1]
    train = train[0:-1]
    # Add feature X0 which is all one
    RowOfOnes = np.array([1.0]*np.shape(train)[1])
    train = np.vstack([RowOfOnes, train])
    train = np.transpose(train)
    yTrain = np.transpose(yTrain)
    # Convert labels from -1,1 to 0,1
    for ind, y in enumerate(yTrain):
        if y == -1:
            yTrain[ind] = 0

    # Read Test
    mat = scipy.io.loadmat('Test_Data.mat')
    test = mat['test']
    # Shuffle Data
    np.random.shuffle(test)

    # Separate Label from train
    test = np.transpose(test)
    yTest = test[len(test) - 1]
    test = test[0:-1]

    # Add feature X0 which is all one
    RowOfOnes = np.array([1.0] * np.shape(test)[1])
    test = np.vstack([RowOfOnes, test])
    test = np.transpose(test)
    yTest = np.transpose(yTest)
    # Convert labels from -1,1 to 0,1
    for ind, y in enumerate(yTest):
        if y == -1:
            yTest[ind] = 0

    result=[]
    for _lambda in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
        # Use Gradient Decent to minimize optimal weights
        #print('Doing LR:')
        #weight, scales, costOfTrainDataSet, accuracyOfTestDataSet,rateForRoc = logistic_gradient_descent(xTrain=train,
        #                      yTrain=yTrain, numberOfIter=100,
        #                      learningRate=0.1, xTest=test, yTest=yTest, _lambda=_lambda,threshold=0.5)

        #print("Lambda: {}".format(_lambda))
        #result.append([costOfTrainDataSet, accuracyOfTestDataSet,rateForRoc])
        #print('Weight: ', weight, '\nScales: ', scales, '\nCostOfTrainDataSet', costOfTrainDataSet,
        #      'AccuracyOfTestDataSet:', accuracyOfTestDataSet,
        #      '\nRate for ROC:', rateForRoc)

        # Do ten fold cross validation for each lambda
        errorOfThenFold_cv = ten_fold_cross_validation(learningFunction=logistic_gradient_descent,
                                                       _lambda=_lambda,
                                                       xtrain=train,
                                                       ytrain=yTrain,
                                                       threshold=0.5,
                                                       iteration=numberOfIteration,
                                                       learningRate=learningRate)

        print("=====Lambda:",_lambda,"=========")
        print("10-fold-CV Training Error For Test: ", errorOfThenFold_cv)
        print("10-fold-CV Training Error Accuracy For Test: ", 1-errorOfThenFold_cv)
        print("================================")
        result.append([errorOfThenFold_cv, _lambda])

    # Best lambda
    result = sorted(result)
    print('Best Lambda is: {}, Its accuracy is:{}'.format(result[0][1],1-result[0][0]))


    # Use Gradient Decent to minimize optimal weights for best lambda with different threshold
    # print('Doing LR for best lambda:')
    rocPoints = [[0,0],[1,1]]
    for threshold in [0.5, 0.4,0.6,0.3,0.7]:
        weight, scales, costOfTrainDataSet, accuracyOfTestDataSet,rateForRoc = logistic_gradient_descent(xTrain=train,
                          yTrain=yTrain, numberOfIter=numberOfIteration,
                          learningRate=learningRate, xTest=test, yTest=yTest,
                          _lambda=result[0][1],threshold=threshold)
        rocPoints.append(rateForRoc)

    # plot roc
    rocPoints = sorted(rocPoints)
    print(rocPoints)

    plt.plot([fpr[0] for fpr in rocPoints], [tpr[1] for tpr in rocPoints], 'b-')
    plt.plot(np.linspace(start=0,stop=1,num=100), np.linspace(start=0,stop=1,num=100), 'r--')
    plt.title('ROC for lambda: {}'.format(result[0][1]))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

    # Use Gradient Decent to minimize optimal weights for lambda  0 with different threshold
    # print('Doing LR for lambda: 0')
    rocPoints = [[0, 0], [1, 1]]
    for threshold in [0.5, 0.4, 0.6, 0.3, 0.7]:
        weight, scales, costOfTrainDataSet, accuracyOfTestDataSet, rateForRoc = logistic_gradient_descent(xTrain=train,
                                                                                                          yTrain=yTrain,
                                                                                                          numberOfIter=numberOfIteration,
                                                                                                          learningRate=learningRate,
                                                                                                          xTest=test,
                                                                                                          yTest=yTest,
                                                                                                          _lambda=0,
                                                                                                          threshold=threshold)
        rocPoints.append(rateForRoc)

    # plot roc
    rocPoints = sorted(rocPoints)
    print(rocPoints)

    plt.plot([fpr[0] for fpr in rocPoints], [tpr[1] for tpr in rocPoints], 'b-')
    plt.plot(np.linspace(start=0, stop=1, num=100), np.linspace(start=0, stop=1, num=100), 'r--')
    plt.title('ROC for lambda: {}'.format(0))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


if __name__ == '__main__':
    runCode()