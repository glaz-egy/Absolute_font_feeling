# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import cupy as np
from config import GPU
import matplotlib.pyplot as plt
from make_dataset import load_mydata
from deep_convnet import DeepConvNet
from trainer import Trainer
np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
np.add.at = np.scatter_add

(x_train, t_train), (x_test, t_test) = load_mydata(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
