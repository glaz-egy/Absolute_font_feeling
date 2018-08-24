# coding: utf-8
import numpy as np

"""
import cupy as np
np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
np.add.at = np.scatter_add
"""

font2num = {'HGRSGU':0,
            'JGTR00M':1,
            'meiryo':2,
            'msgothic':3,
            'UDDigiKyokashoN-R':4,
            'YuGothL':5,
            'DFJHSGW5':6,
            'HGRGE':7,
            'HGRGM':8,
            'HGRPRE':9}

num2font = np.array(['HGRSGU', 'JGTR00M', 'meiryo', 'msgothic', 'UDDigiKyokashoN-R', 'YuGothL', 'DFJHSGW5', 'HGRGE', 'HGRGM', 'HGRPRE'])


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
