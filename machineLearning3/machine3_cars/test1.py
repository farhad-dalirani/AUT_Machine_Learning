import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import scipy.io
import numpy as np
import matplotlib.pylab as plt
import copy as cp


inputF = open('car.data','r')
train = []

for eachcar in inputF:
    eachcar = eachcar.replace(' ', '')
    eachcar = eachcar.replace('\t', '')
    eachcar = eachcar.replace('\n', '')

    carFeaturesString = eachcar.split(',')

    carFeatures = []
    for index, featureString in enumerate(carFeaturesString):
        carFeatures.append(featureString)

    train.append(cp.copy(carFeatures))

np.random.shuffle(train)

# structure of dataset:
# [[feature1, feature2, ..., featureN],
#  [feature1, feature2, ..., featureN],
#  ...
#  [feature1, feature2, ..., featureN]]

raw_data = cp.copy(train)

data = pd.DataFrame(raw_data, columns=['B','M','D','P','L','S','Category'])

print data.set_value(0,'D',10)
print data


train_data = data[:750]


network = BayesianModel(([('Category','B'),('Category','M'),
                                ('Category','D'),('Category','P'),
                                ('Category', 'L'), ('Category', 'S')]))

network.fit(train_data)

#extra line, can be deleted
print network.get_cpds()

test = data[750:]
test = test.drop('Category',axis=1)
prediction = network.predict(test)
print prediction['Category'].tolist()
