import pandas as pd
import numpy as np

from keras.utils import to_categorical

def read_file(path='./data/NewYork_Data.txt'):
    f = pd.read_csv(open(path, 'rb'), delimiter=':', header=None)
    x = [f[1][i].split(',') for i in range(len(f[0]))]
    x = np.array([np.array(xi) for xi in x])
    #y, y_dict = one_hot(f[0])
    x_h = np.array([x[i][0] for i in range(len(x))])
    x = np.array([x[i][1:] for i in range(len(x))])
    #print(x.shape, x[0].shape, x[1])
    for j in range(500, 550):
        print(np.sort([len(x[i]) for i in range(len(x))])[-j])

def y_one_hot(y):
    t=np.sort(np.unique(np.array([y])))
    y_dict = {t[i]:i for i in range(len(t))}
    y_t = [y_dict[y[i]] for i in  range(len(y)) if y[i] in y_dict.keys()]
    y_dict = {v: k for k, v in y_dict.items()}

    return to_categorical(y_t), y_dict

def loc_one_hot(loc=None ,path='./data/NYID_info.txt'):
    loc = ['4bdb01633904a5930c3c489e']
    f = open(path, 'rb')
    loc_df = pd.read_csv(f, sep='\t', header=None)
    f.close()
    loc_id = np.array(loc_df[0])
    loc_dict = {loc_id[i]:i for i in range(loc_id.shape[0])}
    loc = np.array([loc_dict[loc[i]] for i in range(len(loc)) if loc[i] in loc_dict.keys()])

    return to_categorical(loc, num_classes=59283), loc_dict

if __name__ == '__main__':
    read_file()
    #loc_one_hot()
