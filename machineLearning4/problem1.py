import numpy as np
import matplotlib.pylab as plt
import svmutil as svm
import random
import csv
from mpl_toolkits import mplot3d

#########################################################
# A
#########################################################

# Read input data-set
alldataset = []
metaData = True
with open('data1.csv', 'rb') as csvfile:
    lines = csv.reader(csvfile, delimiter=' ', quotechar='|')

    for row in lines:

        # Skip meta data of input file
        if metaData == True:
            metaData = False
            continue

        row = row[0]
        # split features and labels by ','
        rowElements = row.split(',')
        # convert string to float
        alldataset.append([float(x) for x in rowElements])

# Shuffle dataset
random.shuffle(alldataset)

# Seperate features and label
dataset = []
label = []
for observataion in alldataset:
    dataset.append([float(x) for x in observataion[0:-1]])
    label.append(float(observataion[len(observataion) - 1]))

# print dataset and its label
#print dataset
#print label

# data of class1 and class 2
w1 = []
w2 = []
for index, element in enumerate(dataset):
    if label[index] == 1:
        w1.append(dataset[index])
    else:
        w2.append(dataset[index])

plt.figure()
#plot data of class1
plt.plot([x0[0] for x0 in w1], [x1[1] for x1 in w1], 'b*', label='Class 1')
#plot data of class2
plt.plot([x0[0] for x0 in w2], [x1[1] for x1 in w2], 'r^', label='Class 2')
plt.title('Original Data')
plt.legend()


#########################################################
# B
#########################################################

def two_to_one_dim(point):
    """
    This function gets a point in two dimension and map it
    to on dimension according to: [x1, x2] => [sqrt(x1^2 + x2^2)]
    :param point: a point in 2D
    :return: mapped point to 1D
    """
    import math
    return [math.sqrt(abs(point[0])**2+abs(point[1])**2)]


# data of class1 and class 2 in 1 dimention
w1_1d = []
w2_1d = []
dataset_1d = []
for index, element in enumerate(dataset):
    if label[index] == 1:
        # map data of class 1 to 1-D
        w1_1d.append(two_to_one_dim(dataset[index]))
    else:
        # map data of class 2 to 1-D
        w2_1d.append(two_to_one_dim(dataset[index]))

    dataset_1d.append(two_to_one_dim(dataset[index]))

plt.figure()
#plot data of class1 in one dimension
plt.plot([x0[0] for x0 in w1_1d], [0]*len(w1_1d), 'b*', label='Class 1')
#plot data of class2 in one dimension
plt.plot([x0[0] for x0 in w2_1d], [0]*len(w2_1d), 'r^', label='Class 2')
plt.title('Data in 1-D')
plt.legend()


#########################################################
# C
#########################################################

def two_to_two_dim(point):
    """
    This function gets a point in two dimensions and map it
    to two dimensions space which data is separable in it.
    according to: [x1, x2] ==> [x1^2, x2^2]
    :param point: a point in 2D
    :return: mapped point to 1D
    """
    import math
    return [point[0]**2, point[1]**2]


# data of class1 and class 2 in new 2 dimensions
w1_2d = []
w2_2d = []
dataset_2d = []
for index, element in enumerate(dataset):
    if label[index] == 1:
        # map data of class 1 to 1-D
        w1_2d.append(two_to_two_dim(dataset[index]))
    else:
        # map data of class 2 to 1-D
        w2_2d.append(two_to_two_dim(dataset[index]))

    dataset_2d.append(two_to_two_dim(dataset[index]))

plt.figure()
#plot data of class1 in one dimention
plt.plot([x0[0] for x0 in w1_2d], [x1[1] for x1 in w1_2d], 'b*', label='Class 1')
#plot data of class2 in one dimention
plt.plot([x0[0] for x0 in w2_2d], [x1[1] for x1 in w2_2d], 'r^', label='Class 2')
plt.title('Data in 2-D')
plt.legend()


#########################################################
# D
#########################################################

def two_to_three_dim(point):
    """
    This function gets a point in two dimensions and map it
    to three dimensions space which data is separable in it.
    according to: [x1, x2] ==> [ sqrt(2) * x1 * x2, x1^2, x2^2]
    :param point: a point in 2D
    :return: mapped point to 1D
    """
    import math
    return [math.sqrt(2)*point[0]*point[1], point[0]**2, point[1]**2]


# data of class1 and class 2 in new 2 dimensions
w1_3d = []
w2_3d = []
dataset_3d = []
for index, element in enumerate(dataset):
    if label[index] == 1:
        # map data of class 1 to 1-D
        w1_3d.append(two_to_three_dim(dataset[index]))
    else:
        # map data of class 2 to 1-D
        w2_3d.append(two_to_three_dim(dataset[index]))

    dataset_3d.append(two_to_three_dim(dataset[index]))


# Plot data in 3-D model
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D([x0[0] for x0 in w1_3d],
          [x1[1] for x1 in w1_3d],
          [x2[2] for x2 in w1_3d], 'b*')

ax.plot3D([x0[0] for x0 in w2_3d],
          [x1[1] for x1 in w2_3d],
          [x2[2] for x2 in w2_3d], 'r^')

plt.title('Data in 3-D')
#plt.legend()


#########################################################
# E
#########################################################
# Computed kernels matrix for mapping to 1-d, 2-d, 3-d.
# calculate kernel matrix for each pair of data

# Kernel matrix for mapping to on dimension.
k1train = [[] for i in range(len(dataset))]
# Kernel matrix for mapping to on dimension.
k2train = [[] for i in range(len(dataset))]
# Kernel matrix for mapping to on dimension.
k3train = [[] for i in range(len(dataset))]

for indexi, observataion1 in enumerate(dataset):
    for indexj, observataion2 in enumerate(dataset):
        # Calculate Kij for mapping to one dimensional space. kij = phi(Xi).phi(Xj)
        k1train[indexi].append(np.dot(two_to_one_dim(observataion1), two_to_one_dim(observataion2)))

        # Calculate Kij for mapping to two dimensional space. kij = phi(Xi).phi(Xj)
        k2train[indexi].append(np.dot(two_to_two_dim(observataion1), two_to_two_dim(observataion2)))

        # Calculate Kij for mapping to three dimensional space. kij = phi(Xi).phi(Xj)
        k3train[indexi].append(np.dot(two_to_three_dim(observataion1), two_to_three_dim(observataion2)))


#print 'Kernel Matrix of Mapping To One dimensional data:\n', 'Size:', np.shape(k1), '\n', k1train
#print 'Kernel Matrix of Mapping To Two dimensional data:\n', 'Size:', np.shape(k2),'\n', k2train
#print 'Kernel Matrix of Mapping To Three dimensional data:\n', 'Size:', np.shape(k3),'\n', k3train
#for ind,i in enumerate(k3):
#    print i
#    if ind == 20:
#        break



#########################################################
# F
#########################################################
# train linear svm for original data without mapping and kernel trick

five_cv_accuracy = svm.svm_train(label, dataset, '-t 0 -v 5')

print "Accuracy of model without kernel and data mapping: ", five_cv_accuracy,' %'


#########################################################
# G
#########################################################

# train linear svm for mapped data to 1-d
five_cv_accuracy_1d = svm.svm_train(label, dataset_1d, '-t 0 -v 5')
print "Accuracy of model with data mapped to 1-D: ", five_cv_accuracy_1d,' %'

# train linear svm for mapped data to 2-d
five_cv_accuracy_2d = svm.svm_train(label, dataset_2d, '-t 0 -v 5')
print "Accuracy of model with data mapped to 2-D: ", five_cv_accuracy_2d,' %'

# train linear svm for mapped data to 3-d
five_cv_accuracy_3d = svm.svm_train(label, dataset_3d, '-t 0 -v 5')
print "Accuracy of model with data mapped to 3-D: ", five_cv_accuracy_3d,' %'


#########################################################
# H
#########################################################
# Train and evaluate kernel with kernel 1(2D to 1D)
# Prepare kernel matrix
kernel = []
for index in range(len(k1train)):
    kernel.append([index + 1] + k1train[index])

problem = svm.svm_problem(label, kernel, isKernel=True)
parameters = svm.svm_parameter('-t 4 -v 5') # kernel matrix was computed, 5-CV
five_cv_accuracy_kernel_1d = svm.svm_train(problem, parameters)
print "Accuracy of model with Kernel to 1-D: ", five_cv_accuracy_kernel_1d,' %'

# Train and evaluate kernel with kernel 2(2D to 2D)
# Prepare kernel matrix
kernel = []
for index in range(len(k2train)):
    kernel.append([index + 1] + k2train[index])

problem = svm.svm_problem(label, kernel, isKernel=True)
parameters = svm.svm_parameter('-t 4 -v 5') # kernel matrix was computed, 5-CV
five_cv_accuracy_kernel_2d = svm.svm_train(problem, parameters)
print "Accuracy of model with Kernel to 2-D: ", five_cv_accuracy_kernel_2d,' %'

# Train and evaluate kernel with kernel 3(2D to 3D)
# Prepare kernel function
kernel = []
for index in range(len(k3train)):
    kernel.append([index + 1] + k3train[index])

problem = svm.svm_problem(label, kernel, isKernel=True)
parameters = svm.svm_parameter('-t 4 -v 5') # kernel matrix was computed, 5-CV
five_cv_accuracy_kernel_3d = svm.svm_train(problem, parameters)
print "Accuracy of model with Kernel to 3-D: ", five_cv_accuracy_kernel_3d,' %'


plt.show()
