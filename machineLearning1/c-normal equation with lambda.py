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


def nonlinear_transform_normal_equation(inputX, y, dim, Lambda, xTest, yTest):
    """
    This function use normal equation to minimize cost function j
    over theta parameters. Before starting, it does
    a nonlinear transformation.
    :param inputX: Train data, each row is a train data and each
            column represent a feature
    :dim: This argument shows : [1 x x^2 x^3 ... x^d]
    :param y: A vector that its rows are value of corresponding
                rows of x.
    :param Lambda: for dealing with over fitting
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

    # After transpose each row is point and each column is a feature
    x = np.transpose(x_train)
    x_test = np.transpose(x_test)

    # Initial necessary matrices for doing normal equation
    x_train_mat = np.matrix(x)
    x_train_tran_mat = np.transpose(x_train_mat)
    y_train_mat = np.matrix(y)
    y_train_mat = np.transpose(y_train_mat)
    identity_mat = np.identity(np.size(x_train_mat,axis=1)) * Lambda
    identity_mat[0][0] = 0

    # Do normal equation
    weight = np.linalg.inv(x_train_tran_mat*x_train_mat + identity_mat) * x_train_tran_mat * y_train_mat
    weight = np.transpose(weight).tolist()
    weight = weight[0]
    print 'weights >>> ', weight

    # Plot curves
    t = np.arange(min(inputX), max(inputX), 0.01)
    curve = []
    for i in t:
        temp = weight[0]
        for dimention, w in enumerate(weight[1::]):
            temp += w * (i ** (dimention + 1))
        curve.append(temp)
    plt.plot(t, curve, 'g-')

    # Plot input point
    plt.plot([(xAxis) for xAxis in inputX], y, 'bo',
             label='Train Data')
    plt.plot([(xAxis) for xAxis in xTest], yTest, 'ro',
             label='Test Data')
    plt.xlim(min(inputX), max(inputX))
    plt.ylim(min(y), max(y))
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dim: {},Lambda: {}'.format(dim, Lambda))
    plt.show()

    # Cost function value for train data set
    costOfTrainDataSet = cost_function(thetaVec=weight, xMat=x, y=y)
    # Cost function value for test data set
    costOfTestDataSet = cost_function(thetaVec=weight, xMat=x_test, y=yTest)

    # Return weight, scale of each feature, cost on train, cost on test
    print '>', costOfTrainDataSet, costOfTestDataSet
    return weight, costOfTrainDataSet, costOfTestDataSet


def runCodeC():
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

    allCostFunction = {5: {}, 50: {}, 500: {}}
    # Do following for different dimensions as question asked
    for dim in [7]:
        for Lambda in [5, 50, 500]:
            # Solve Normal Equation with specific nonlinear transformation and Lambda
            weight, costTrain, costTest= nonlinear_transform_normal_equation(inputX=inputX_train,
                                                        y=y_train, dim=dim,
                                                        Lambda=Lambda,
                                                        xTest=X_test,yTest=y_test)

            allCostFunction[Lambda] = (costTrain,costTest)

    print allCostFunction
    color = {5: 'r', 50: 'b', 500: 'g'}
    for index, Lambda in enumerate([5, 50, 500]):
        plt.plot(index * 5,allCostFunction[Lambda][0], color[Lambda]+'o',
                 label="Train Dim 7, Lambda "+str(Lambda))
        plt.plot(index * 5, allCostFunction[Lambda][1], color[Lambda]+'*',
                 label="Test Dim 7, Lambda"+str(Lambda))
        plt.title('Dim: {}'.format(7))
        plt.xlabel('Lambda')
        plt.ylabel('MSE')
        plt.xlim(-1,20)

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


if __name__ == '__main__':
    runCodeC()