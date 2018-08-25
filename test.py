import numpy as np
import matplotlib.pyplot as plt
from make_dataset import load_mydata
from deep_convnet import DeepConvNet

network = DeepConvNet() 

(x_train, t_train), (x_test, t_test) = load_mydata(flatten=False)

network.load_params(file_name='deep_convnet_params.pkl')
print(network.accuracy(x_test, t_test, batch_size=10))