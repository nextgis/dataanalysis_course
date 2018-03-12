import pandas as pd
import numpy as np

from mlp import MLP

url="http://mlr.cs.umass.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone = pd.read_csv(url, header=None, 
    names=['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 
          'Viscera_weight', 'Shell_weight', 'Rings']
)

sample = abalone.loc[ ~ (abalone['Sex'] =='M')]
sample = sample.copy()

sample['Class'] = -1
sample.loc[(sample['Sex'] =='I'), 'Class'] = 1.0


data = np.array(sample)
sample_count = data.shape[0]


np.random.seed(42)
train_idx = np.random.randint(sample_count, size=int(sample_count*0.67))


x_train = data[train_idx, 1:9]
y_train = data[train_idx, 9]
train_count = x_train.shape[0]


x_test = data[~train_idx, 1:9]
y_test = data[~train_idx, 9]
test_count = x_test.shape[0]



network = MLP(8, 20, 1)

def get_avg_err(net, x, y):
     count = x.shape[0]
     err = 0
     for j in range(count):
        res = network.propagate_forward(x[j])
        if res * y[j] < 0:
            err += 1
     return err/count
    

for i in range(100000):
    n = np.random.randint(train_count)
    network.propagate_forward(x_train[n])
    err = network.propagate_backward(y_train[n], lrate=0.001, momentum=0.001)
    if i % 100 == 0:
        print(i, get_avg_err(network, x_test, y_test), get_avg_err(network, x_train, y_train))
