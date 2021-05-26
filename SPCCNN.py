# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:59:19 2020

@author: yin
"""
import datetime
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 
from scipy.io import loadmat
trainroad = 'D:/chengxu//FJCNN/FJ.XYSC_FZCM_FDQY_DSXP_YXBM.mat'

testroad = 'D:/chengxu//FJCNN/FJ.XYSC_FZCM_FDQY_DSXP_YXBM.mat'
#testroad='E:/chenxu/DetectWave.mat'
Datatrain = loadmat(trainroad)
Datatest = loadmat(testroad)
All_data=Datatrain['images']#mnist.train.images：numpy.ndarray
All_target=Datatrain['labels']
x_mid=All_data[1:20001];y_train1=All_target[1:20001];
#
#y=np.vstack((x_mid,x_add))
x_test_mid=All_data[0:-2001];y_test1=All_target[0:-2001];
x_train,x_test,y_train,y_test= [],[],[],[]
# 
for i in range(1000):
    x_train.append(x_mid[i][list(range(0,5400,6))]);
    x_test.append(x_test_mid[i][list(range(0,5400,6))])
    #print(i)
x_train=np.array(x_train);
x_test=np.array(x_test);
for i in range(1000):
    for j in range(10):
        if y_train1[i][j]==1:
            y_train.append(j)
for i in range(1000):
    for j in range(10):
        if y_test1[i][j]==1:
            y_test.append(j)
y_train=np.array(y_train);
y_test=np.array(y_test);
#print("x_train.shape", x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 30, 30, 1) 
 #X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols) 
# 
x_test = x_test.reshape(x_test.shape[0], 30, 30, 1)



#print("x_train.shape", x_train.shape)


class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')   
        self.b1 = BatchNormalization()  # BN layer
        self.a1 = Activation('relu')  # active layer
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  
        self.d1 = Dropout(0.2)  # dropout layer
        
        self.c2 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  
        self.b2 = BatchNormalization()  # BN layer
        self.a2 = Activation('relu')  # active layer
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  
        self.d2 = Dropout(0.2)  # dropout layer
        
        
        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)
        

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = Baseline()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
#log_dir = "logs/fit/" 
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_/' + current_time + '/train'
test_log_dir = 'logs/gradient_/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
#d=(50002,50003,50004,50005)
for i in range(200):
    x_predict = x_test[i]
    x_predict = x_predict.reshape(1, 30, 30, 1)
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    print('predict value:')
    tf.print(pred)

    print('actual value:'+str(y_test[i]))
