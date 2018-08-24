# coding: utf-8
import os.path
import pickle
import os
import numpy as np
from PIL import Image

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

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mydataset.pkl"

font_list = []
for name in os.listdir('fonts'):
    font_list.append(os.path.splitext(name)[0])

img_dim = (1, 28, 28)
img_size = 784

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def _load_label(train=False):
    label_list = []
    if train:
        file_num = len(os.listdir('gray_datas/'+font_list[0]))+1
        one_dot = int(file_num * 0.05)
        for num in range(1, file_num):
            for name in font_list:
                label_list.append(font2num[name])
            print("Load train label Done: {}".format(name))
    else:
        for num in range(1, len(os.listdir('testdata/'+font_list[0]))+1):
            for name in font_list:
                label_list.append(font2num[name])
            print("Load test label Done: {}".format(name))
    labels = np.array(label_list)
    print("Done")

    return labels

def _load_img(train=False):
    data_list = []

    if train:
        file_num = len(os.listdir('gray_datas/'+font_list[0]))+1
        one_dot = int(file_num * 0.05)
        for num in range(1, file_num):
            for name in font_list:
                data_list.append(np.array(Image.open('gray_datas/'+name+'/'+str(num).zfill(7)+'.jpg')))
            if num % one_dot == 0:
                print("#", end="")
        print("\nLoad train image Done: {}".format(name))
    else:
        for num in range(1, len(os.listdir('gray_testdata/'+font_list[0]))+1):
            for name in font_list:
                data_list.append(np.array(Image.open('gray_testdata/'+name+'/'+str(num)+'.jpg')))
        print("Load test image Done: {}".format(name))
    data = np.array(data_list)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(train=True)
    dataset['train_label'] = _load_label(train=True)
    dataset['test_img'] = _load_img(train=False)
    dataset['test_label'] = _load_label(train=False)

    return dataset

def init_mydataset():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mydata(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
        init_mydataset()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
        
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
