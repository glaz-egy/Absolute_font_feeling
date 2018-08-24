from make_dataset import load_mydata
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mydata(flatten=False)

img = []

i = np.array(Image.open('gray_testdata/meiryo/1.jpg'))
img.append((i > 128)* 255)
img.append(np.array(Image.open('gray_testdata/meiryo/2.jpg')))

print(x_test[0][0] * 255)
print(t_test[0])
img_show(x_test[0][0] * 255)