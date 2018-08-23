# coding: utf-8
import os.path
import pickle
import os
import numpy as np
from PIL import Image

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mydataset.pkl"

font_name = ('meiryo','ms_gothic')

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 10000

def _load_label():
    label_list = []

    for name in font_name:
        for num in range(1, 901):
            label_list.append(name)
    labels = np.array(label_list)
    print("Done")

    return labels

def _load_img():
    data_list = []

    for name in font_name:
        for num in range(1, 901):
            data_list.append(np.array(Image.open('datas/'+name+'/'+str(num).zfill(5)+'.jpg')))
    data = np.array(data_list)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['img'] =  _load_img()
    dataset['label'] = _load_label()

    return dataset

def init_mnist():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
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
        init_mnist()

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
            dataset[key] = dataset[key].reshape(-1, 1, 100, 100)

    return (dataset['img'], dataset['label'])


if __name__ == '__main__':
    init_mnist()
