def cost_function(thetaVec, xMat, y):
    """
    This function calculates Mean Square Error
    :param thetaVec: weight vector
    :param xMat: Train data, each row is a train data and each
            column represent a feature
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :return: return a scalar which is mean square error
    """
    import numpy as np

    # Initial output
    error = 0
    for i, x in enumerate(xMat):
        # Calculate difference of h(x) and y
        signalError = (np.dot(x, thetaVec) - y[i]) ** 2

        # Add error of i th input to total error
        error += signalError

    # Calculate mean of error. 2 is for mathematical convenient
    error = (1 / (len(xMat) * 2.0)) * error

    # Return error
    return error


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
        signalError = (np.dot(x, thetaVec) - y[i]) / (len(xMat) * 1.0)

        # Update derivatives of cost function for all
        # parameters. d J(theta[0],theta[1], ...,theta(n)]/d(theta[index]
        for index, theta in enumerate(thetaVec):
            newTheta[index] += signalError * x[index]

    # return derivatives of cost function
    return newTheta


def gradient_descent(xMat, y, numberOfIter, learningRate):
    """
    This functiofn use gradient descent to minimize cost function j
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


def nonlinear_transform_gradient_descent(inputX, y, dim, numberOfIter, learningRate, xTest, yTest):
    """
    This function use gradient descent to minimize cost function j
    over theta parameters. Before starting fradient descent it does
    a nonlinear transformation.
    :param inputX: Train data, each row is a train data and each
            column represent a feature
    :dim: This argument shows : [1 x x^2 x^3 ... x^d]
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

    # Add feature X0
    x_train = [len(inputX) * [1]]
    x_test = [len(xTest) * [1]]

    # Add feature X1
    x_train.append( np.copy(inputX))
    x_test.append( np.copy(xTest))

    # Construct Higher dimensions x^i
    for i in range(2, dim + 1):
        # Append inputX ^ i as a feature
        x_train.append([element ** i for element in inputX])
        x_test.append([element ** i for element in xTest])

    # Find mean, min, max for normalization each feature
    scales = {}
    for index, featurde in enumerate(x_train):
        meanOfFeature = np.mean(feature)
        maxOfFeature = np.max(feature)
        minOfFeature = np.min(feature)
        scales[index] = (meanOfFeature, minOfFeature, maxOfFeature)

        if index == 0:
            continue

        # Normalize each feature of train set and test set
        for i, element in enumerate(x_train[index]):
            x_train[index][i] = (element - meanOfFeature) / (1.0 * (maxOfFeature - minOfFeature))
        for i, element in enumerate(x_test[index]):
            x_test[index][i] = (element - meanOfFeature) / (1.0 * (maxOfFeature - minOfFeature))


    # After transpose each row is point and each column is a feature
    x = np.transpose(x_train)
    x_test = np.transpose(x_test)

    # Do gradient Descent
    weight, errors = gradient_descent(xMat=x, y=y, numberOfIter=numberOfIter, learningRate=learningRate)

    #####################
    # Plot error rate for each Iteration
    plt.plot(errors, 'b--')
    plt.title('Cost Function during training:\n Dim={},Iteration: {},Alpha: {}'.format(
                dim,numberOfIter,learningRate))
    plt.ylabel('MSE')
    plt.xlabel('Each step during training')
    plt.show()
    #####################

    # Plot curves
    t = np.arange(-1, 1, 0.01)
    curve = []
    for i in t:
        temp = weight[0]
        for dimention, w in enumerate(weight[1::]):
            temp += w * (i ** (dimention + 1))
        curve.append(temp)
    plt.plot(t, curve, 'g-')

    # Plot input point
    plt.plot([(xAxis - scales[1][0]) / (1.0 * (scales[1][1] - scales[1][0])) for xAxis in inputX], y, 'bo',
             label='Train Data')
    plt.plot([(xAxis - scales[1][0]) / (1.0 * (scales[1][1] - scales[1][0])) for xAxis in xTest], yTest, 'ro',
             label='Test Data')
    plt.xlim(-1,1)
    plt.ylim(-2, 2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dim: {},Iteration: {},Alpha: {}'.format(dim,numberOfIter,learningRate))
    plt.show()

    # Cost function value for train data set
    costOfTrainDataSet = errors[ len(errors)-1]
    # Cost function value for test data set
    costOfTestDataSet = cost_function(thetaVec=weight,xMat=x_test,y=yTest)

    # Return weight, scale of each feature, cost on train, cost on test
    print '>', costOfTrainDataSet, costOfTestDataSet
    return weight,scales, costOfTrainDataSet, costOfTestDataSet


def runCode():
    # Import
    import pandas as pd
    import matplotlib.pylab as plt
    import numpy as np
    from sklearn.cross_validation import train_test_split

    # Read Data from 'data.xlsx' file
    dfs = pd.read_excel('data.xlsx', sheetname='Sheet1', header=None)
    values = list(dfs.values.T.tolist())

    # Split data to two part: train and test
    inputX_train, X_test, y_train, y_test = train_test_split(values[0],values[1],test_size=0.20)


    # Do following for different dimensions as question asked
    learningRate = {3: 0.03, 5: 0.03, 7: 0.02}
    allCostFunction = {100:{}, 1000:{}, 10000:{}}
    for dim in [3, 5, 7]:
        for iter in [100, 1000, 10000]:
            # Do Gradient Descent with specific nonlinear transformation and iteration
            weight, scales, costTrain, costTest= nonlinear_transform_gradient_descent(inputX=inputX_train,
                                                            y=y_train, dim=dim,
                                                            numberOfIter=iter, learningRate=learningRate[dim],
                                                            xTest=X_test,yTest=y_test)
            allCostFunction[iter][dim] = (costTrain,costTest)

    print allCostFunction
    color = {3:'r', 5:'b', 7:'g'}
    for iter in [100, 1000, 10000]:
        for dim in [3, 5, 7]:
            plt.plot(dim,allCostFunction[iter][dim][0], color[dim]+'o', label="Train Dim "+str(dim))
            plt.plot(dim, allCostFunction[iter][dim][1], color[dim] +'*', label="Test Dim "+str(dim))
            plt.title('Iter: {}'.format(iter))
            plt.ylabel('MSE')
            plt.xlim(1,9)

        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()

if __name__ == '__main__':
    runCode()