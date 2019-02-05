
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import chainer
import chainer.links as L
import chainer.functions as F


# In[3]:


from chainer.datasets import get_mnist
train, test = get_mnist(ndim=3)  # ndim=3はcnn用のデータセット形式


# In[4]:


class CNN(chainer.Chain):

    def __init__(self, n_mid=100, n_out=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=1, out_channels=3, ksize=3, stride=1, pad=1)
            self.fc1 = L.Linear(None, n_mid)
            self.fc2 = L.Linear(None, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, 3)
        h = self.fc1(h)
        h = self.fc2(h)
        return h


# In[5]:


import random

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


# In[6]:


# CPUとGPU関連のシードをすべて固定
reset_seed(0)


# In[7]:


# インスタンス化
model = L.Classifier(CNN())


# In[8]:


if chainer.cuda.available:
    gpu_id = 0  # 使用したGPUに割り振られているID
    model.to_gpu(gpu_id)
else:
    gpu_id = -1


# In[9]:


# Optimizerの定義とmodelとの紐づけ
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)


# In[10]:


batchsize = 4096
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)


# In[11]:


from chainer import training
from chainer.training import extensions

epoch = 30

updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

trainer = training.Trainer(updater, (epoch, 'epoch'), out='mnist')

# バリデーション用のデータで評価
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))

# 学習結果の途中を表示する
trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))

# １エポックごとに結果をlogファイルに出力させる
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy', 'main/loss', 'validation/main/loss', 'elapsed_time']), trigger=(1, 'epoch'))


# In[12]:


trainer.run()

