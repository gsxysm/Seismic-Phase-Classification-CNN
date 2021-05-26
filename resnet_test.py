import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 
from scipy.io import loadmat
np.set_printoptions(threshold=np.inf)
trainroad = 'D:/chengxu/FJCNN/FJ.XYSC_FZCM_FDQY_DSXP_YXBM.mat'

testroad = 'D:/chengxu/FJCNN/FJ.XYSC_FZCM_FDQY_DSXP_YXBM.mat'
#testroad='E:/test/DetectWave.mat'
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
# 
x_test = x_test.reshape(x_test.shape[0], 30, 30, 1)


class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # 
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  
        
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 
        for block_id in range(len(block_list)): 
            for layer_id in range(block_list[block_id]): 

                if block_id != 0 and layer_id == 0:  
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  
            self.out_filters *= 2 
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/ResNet18.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

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
