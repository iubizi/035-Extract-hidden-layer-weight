####################
# 避免独占
####################

import gc
gc.enable()

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

####################
# 读取mnist
####################

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

print('x_train.shape =', x_train.shape)
print('x_test.shape =', x_test.shape)
print()

print('y_train_onehot.shape =', y_train_onehot.shape)
print('y_test_onehot.shape =', y_test_onehot.shape)
print()

####################
# 构造模型
####################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout

def get_model():
    
    model = Sequential()

    model.add( Flatten( input_shape=(28, 28) ) )

    model.add( Dense(1024, activation='relu') )
    model.add( Dropout(0.5) )

    model.add( Dense(512, activation='relu') )
    model.add( Dropout(0.5) )

    model.add( Dense(256, activation='relu') )
    model.add( Dropout(0.5) )

    model.add( Dense(128, activation='relu') )
    model.add( Dropout(0.5) )

    model.add( Dense(64, activation='relu', name='extract') )
    model.add( Dropout(0.5) )

    model.add( Dense(10, activation='softmax'))

    return model

####################
# 编译模型
####################

from tensorflow.keras.optimizers import Adam

model = get_model()

model.compile( optimizer = Adam( learning_rate=1e-4 ),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'],
               )

model.summary()

####################
# 回调函数
####################

# 早退
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping( monitor = 'val_accuracy',
                                patience = 5,
                                restore_best_weights = True )

####################
# 训练
####################

history = model.fit( x_train, y_train_onehot,
                     validation_data = (x_test, y_test_onehot),

                     epochs = 1000, batch_size = 32,
                     verbose = 1, # 2 只显示一行
                     callbacks = [early_stopping],

                     # max_queue_size = 1000,
                     workers = 4,
                     use_multiprocessing = True,
                     )

####################
# 提取隐藏层
####################

from tensorflow.keras.models import Model

# 创建新模型
extract_layer = Model( inputs = model.input,
                       outputs = model.get_layer('extract').output )
extract_output = extract_layer.predict(x_test)

####################
# 存储为npz压缩格式
####################

import numpy as np

np.savez_compressed( 'extract_output.npz',
          
                     extract_output = extract_output,
                     y_test = y_test )

input() # 看一下训练结果，避免闪退
