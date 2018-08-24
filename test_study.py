from make_dataset import load_mydata
from deep_convnet import DeepConvNet
import numpy as np
import random
from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mydata(flatten=False)

network = DeepConvNet()
network.load_params('deep_convnet.pkl')

index = random.randint(0, 59)

img = np.array(Image.open('gray_testdata/YuGothL/4.jpg'))
img = img.reshape(-1, 28, 28)
img = img.reshape(-1, 1, 28, 28)
print(img.shape)
ans = network.predict(img)
print(ans)