def sigmoid(x):
    """
    Sigmoid function
    :param x:
    :return: 1/(1+e^(-x))
    """
    import numpy as np
    return 1/(1 + np.exp(-1 * x))


def cost_function(thetaVec, xMat, y):
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
        signalError = -(y[i])*(np.log(sigmoid(np.dot(x, thetaVec)))/np.log(2))\
                      -(1-y[i])*(np.log(1-sigmoid(np.dot(x, thetaVec)))/np.log(2))

        # Add error of i th input to total error
        error += signalError

    # Calculate entropy cost function
    error = (1 / (len(xMat) * 2.0)) * error

    # Return error
    return error


def accuracy_of_test(thetaVec, xMat, y):
    """
    This function calculates accuracy of test: correct classifications/total
    :param thetaVec: weight vector
    :param xMat: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :return: return a scalar
    """
    import numpy as np

    # Initial output
    accuracy = 0
    for i, x in enumerate(xMat):
        # Calculate cross entropy of h(x(i)) and y(i)

        if sigmoid(np.dot(x, thetaVec)) >= 0.5 and y[i]==1:
            accuracy += 1
        elif sigmoid(np.dot(x, thetaVec)) < 0.5 and y[i]==0:
            accuracy +=1

    # Calculate accuracy
    accuracy = accuracy / len(xMat)

    # Return accuracy
    return accuracy


def gradients(thetaVec, xMat, y):
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

    # return derivatives of cost function
    return newTheta


def gradient_descent(xMat, y, numberOfIter, learningRate):
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
    iterCost = [cost_function(xMat=xMat, thetaVec=thetaVec, y=y)]

    # In each iteration update weight vector
    for iter in range(numberOfIter):
        # Calculate gradients
        gradientsOfthetaVec = gradients(thetaVec=thetaVec, xMat=xMat, y=y)
        # Update weights
        for index, theta in enumerate(thetaVec):
            thetaVec[index] = theta - learningRate * gradientsOfthetaVec[index]
        #print thetaVec,'*'

        # Update learning rate
        #learningRate = learningRate * 0.95
        #print np.sqrt(np.dot(gradientsOfthetaVec,gradientsOfthetaVec))

        # Add value of cost function to list of weight
        iterCost.append(cost_function(xMat=xMat, thetaVec=thetaVec, y=y))

    # Return list of weight and costs in each iteration
    #print thetaVec
    return thetaVec, iterCost


def logistic_gradient_descent(xTrain, yTrain, numberOfIter, learningRate, xTest, yTest):
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
                                      learningRate=learningRate)

    #####################
    # Plot error rate for each Iteration
    plt.plot(errors, 'b--')
    plt.title('Cost Function during training:\n Iteration: {},Alpha: {}'.format(
                numberOfIter,learningRate))
    plt.ylabel('Cross Entropy')
    plt.xlabel('Each step during training')
    plt.show()
    #####################

    # Cost function value for train data set
    costOfTrainDataSet = errors[ len(errors)-1]
    # Cost function value for test data set
    accuracyOfTestDataSet = accuracy_of_test(thetaVec=weight,xMat=x_test,y=yTest)

    # Return weight, scale of each feature, cost on train, accuracy on test
    print('>', costOfTrainDataSet, accuracyOfTestDataSet)
    return weight,scales, costOfTrainDataSet, accuracyOfTestDataSet


def runCode():
    """
    Main function which I read inputs and calls
    different function for training and testing
    :return: none
    """
    import scipy.io
    import numpy as np

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

    # Use Gradient Decent to minimize optimal weights
    weight, scales, costOfTrainDataSet, accuracyOfTestDataSet = logistic_gradient_descent(xTrain=train,
                              yTrain=yTrain, numberOfIter=100,
                              learningRate=0.5, xTest=test, yTest=yTest)

    print(weight,scales, costOfTrainDataSet, accuracyOfTestDataSet)
    print("Training Error: ", costOfTrainDataSet)
    print("Test accuracy: ", accuracyOfTestDataSet)

if __name__ == '__main__':
    runCode()