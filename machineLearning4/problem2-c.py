import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import svmutil as svm
import random

# Open .CSV file
dataset = pd.read_csv('parkinsons.data')

# Indexes of features for subtracting additional columns
features = dataset.columns.values[np.multiply(dataset.columns.values != 'name', dataset.columns.values != 'status')]
# Pick 60 percentage of data for training
trainSec, valid_test = train_test_split(dataset, train_size=0.6)
# Pick 20 percentage of data for test and 20% for validation
validSec, testSec = train_test_split(valid_test, train_size=0.5)

# Convert training section to list and just keep features
train = []
for i in range(np.shape(trainSec)[0]):
    train.append(trainSec[features].iloc[i].tolist())
# Extract labels of observations in train part
trainLabel = [int(float(label)) for label in trainSec['status'].tolist()]

# Convert validation section to list and just keep features
valid = []
for i in range(np.shape(validSec)[0]):
    valid.append(validSec[features].iloc[i].tolist())
# Extract labels of observations in valid part
validLabel = [int(float(label)) for label in validSec['status'].tolist()]

# Convert test section to list and just keep features
test = []
for i in range(np.shape(testSec)[0]):
    test.append(testSec[features].iloc[i].tolist())
# Extract labels of observation in test part
testLabel = [int(float(label)) for label in testSec['status'].tolist()]

#print train
#print trainLabel


########################################################################
# A
########################################################################
# Range of c and gamma: 10^i [i| -10,11]
cRange = (-10, 11)
gammaRange = (-10, 11)

# Make empty grid(10*10) which each element is a trained svm model with specific C and gamma
grid_c_gamma = {c: {} for c in range(cRange[0], cRange[1])}

for c in range(cRange[0], cRange[1]):
    for gamma in range(gammaRange[0], gammaRange[1]):
        # Set sigmoid Kernel
        parameters = svm.svm_parameter('-t 3 -q')

        # Set parameter 'C': 10^i [i| -10,10]
        parameters.C = 10 ** c

        # Set parameter 'gamma': 10^i [i| -10,10]
        parameters.gamma = 10 ** gamma

        # Train svm with specific "C" and "Gamma"
        problem = svm.svm_problem(trainLabel, train)
        trainedSVM = svm.svm_train(problem, parameters)

        # Calculate accuracy of SVM model on validation set
        valid_acc = svm.svm_predict(validLabel, valid, trainedSVM)[1][0]

        # Add tuple (C, Gamma, validation accuracy) to grid
        grid_c_gamma[c][gamma] = (parameters.C, parameters.gamma, valid_acc)

# Print grid
#print 'Grid of C, Gamma, Validation Accuracy'
#for row in grid_c_gamma:
#    print row


# find maximum validation accuracy
maxAcc = -1
bestC = None
bestGamma = None

for c in range(cRange[0], cRange[1]):
    for gamma in range(gammaRange[0], gammaRange[1]):
        if maxAcc < grid_c_gamma[c][gamma][2]:
            maxAcc = grid_c_gamma[c][gamma][2]
            bestC = c
            bestGamma = gamma

print '\n'
print 'Part A:'
print '> Best Validation Accuracy is {} Which belongs to C:{}, Gamma:{}'.format(grid_c_gamma[bestC][bestGamma][2],
                                                                              10**bestC,
                                                                              10**bestGamma)

# Train a SVM with best C and Gamma that were founded
# Set RBF Kernel
parameters = svm.svm_parameter('-t 2 -q')

# Set parameter 'C': Best C which was founded
parameters.C = 10 ** bestC

# Set parameter 'gamma': Best gamma which was founded
parameters.gamma = 10 ** bestGamma

# Train svm with best "C" and "Gamma"
problem = svm.svm_problem(trainLabel, train)
bestTrainedSVM = svm.svm_train(problem, parameters)

# Calculate accuracy of SVM model on test set
test_acc = svm.svm_predict(testLabel, test, bestTrainedSVM)[1][0]

print '> Accuracy of SVM (with best validation accuracy) on test set is {} ,\nWhich belongs to C:{}, Gamma:{}'.format(
                                                                              test_acc,
                                                                              10**bestC,
                                                                              10 **bestGamma)
print 'Number Of SV:', len(bestTrainedSVM.get_sv_indices())


########################################################################
# B
########################################################################
# In previous part(exhaustive) validation accuracy is saved for different
# values of C and Gamma, in this part It chooses 20 different value of
# (c, gamma) randomly and pick one with highest validation accuracy
randomC = [random.choice(range(cRange[0], cRange[1])) for i in range(20)]
randomGamma = [random.choice(range(gammaRange[0], gammaRange[1])) for i in range(20)]

# Find a svm that has highest validation accuracy
maxAcc = -1
bestC = None
bestGamma = None
for c, gamma in zip(randomC, randomGamma):
    if maxAcc < grid_c_gamma[c][gamma][2]:
        maxAcc = grid_c_gamma[c][gamma][2]
        bestC = c
        bestGamma = gamma

print '\n'
print 'Part B:'
print '> Best Validation Accuracy is {} Which belongs to C:{}, Gamma:{}'.format(grid_c_gamma[bestC][bestGamma][2],
                                                                                    10 ** bestC,
                                                                                    10 ** bestGamma)


# Train a SVM with best C and Gamma that randomly were founded among
# 20 different random svm

# Set RBF Kernel
parameters = svm.svm_parameter('-t 2 -q')

# Set parameter 'C': Best C which was founded
parameters.C = 10 ** bestC

# Set parameter 'gamma': Best gamma which was founded
parameters.gamma = 10 ** bestGamma

# Train svm with best "C" and "Gamma"
problem = svm.svm_problem(trainLabel, train)
bestTrainedSVMRand = svm.svm_train(problem, parameters)

# Calculate accuracy of SVM model on test set
test_acc_rand = svm.svm_predict(testLabel, test, bestTrainedSVMRand)[1][0]

print '> Accuracy of SVM (with best validation accuracy) on test set is {} ,\nWhich belongs to C:{}, Gamma:{}'.format(
                                                                              test_acc_rand,
                                                                              10**bestC,
                                                                              10 **bestGamma)
print 'Number Of SV:', len(bestTrainedSVMRand.get_sv_indices())

