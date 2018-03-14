import pandas as pd
import numpy as np


url="http://mlr.cs.umass.edu/ml/machine-learning-databases/abalone/abalone.data"
abalone = pd.read_csv(url, header=None, 
    names=['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 
          'Viscera_weight', 'Shell_weight', 'Rings']
)

sample = abalone.loc[ ~ (abalone['Sex'] =='M')]
sample = sample.copy()

sample['Class'] = -1
sample.loc[(sample['Sex'] =='I'), 'Class'] = 1.0

sample = sample.ix[:, ['Class', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 
          'Viscera_weight', 'Shell_weight', 'Rings']]

data = np.array(sample)

sample_count, dimension = data.shape
b = np.ones((sample_count, 1), dtype=float)
data = np.hstack((b, data))

np.random.seed(42)
train_idx = np.random.randint(sample_count, size=int(sample_count*0.67))


x_train = data[train_idx, 1:9]
y_train = data[train_idx, 9]
train_count = x_train.shape[0]


x_test = data[~train_idx, 1:9]
y_test = data[~train_idx, 9]
test_count = x_test.shape[0]

def learn_step(curr_w, in_x, answ_y_, alpha):
    x = in_x.astype(float)
    curr_w = curr_w.astype(float)

    # w_j(t+1) = w_j(t) - alpha ( y - y_ ) x_j
    y = np.sum(curr_w * x)
    new_w = curr_w - alpha*(y - answ_y_)*in_x

    return new_w
  

def learn(inX, answ_y, steps=10000, alpha=0.1):
    count, dimension = inX.shape
    w = np.zeros(dimension, dtype=float)
    for i in range(steps):
        alpha = alpha - alpha*i/steps
        idx = np.random.randint(count)
        x = inX[idx, :]
        y_ = answ_y[idx]
        w = learn_step(w, x, y_, alpha)
        if i % 99 == 0:
            test_err = get_err(x_test, y_test, w)
            train_err = get_err(x_train, y_train, w)
            print(i, test_err, train_err)
    
    return w

def get_err(x, answ, w):
    count, _ = x.shape
    y = x.dot(w)

    err = np.sum(np.abs(y-answ))/count

    return err

 

w = learn(x_train, y_train, steps=10000, alpha=0.7)
print(w)

