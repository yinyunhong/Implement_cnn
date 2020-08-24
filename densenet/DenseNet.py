import keras
import math
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D,Dense,Input,add,Activation,AveragePooling2D,GlobalAveragePooling2D,Lambda,concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler,TensorBoard,ModelCheckpoint
from keras.models import Model
from keras import optimizers,regularizers
from data_process import *

#每个DenseBlock内部的结构是input->BN->ReLU->1*1 conv->BN->Relu->3*3 conv
#1*1的卷积核为了降低特征的数量
#每组DenseBlock使用12个卷积核，所以最终每个DenseBlock得到的feature map为 12通道
growth_rate = 12
#深度:网络的深度，不算无参数的pooling和BN层：除去第一个卷积层和两个transition中的卷积层和最后的linear层，共剩36层
depth=100
#transition层中进行压缩时，上接的DenseBlock得到的特征图的channels数为m,则transition层可以产生compressing*m个特征
compressing = 0.5

#图像的大小
img_rows,img_cols = 32,32
img_channels = 3 #图像的通道数
num_classes = 10 #分类数为10
batch_size = 64 #一次训练64个样本
epochs = 300 #所有的样本完成一次训练，需要训练多少次
iterations = 782 #所有的样本迭代训练782次
#weight decay也叫做L2正则化，在一定的程度上可以减少模型的过拟合问题，系数lamda就是权重衰减系数weight_decay
weight_decay = 1e-4
mean = [125.307,122.95,113.865]
std = [62.9932,62.0887,66.7048]

from keras import backend as K
#如果keras使用的是tensorflow的后端支持
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

def scheduler(epoch):
    '''
    当一次训练 训练到50%和75%时，更改learning_rate的值
    '''
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001

def DenseNet(img_input,classes_num):

    #定义卷积层函数
    def conv(x,out_filters,k_size):
        '''
        x-输入
        out_filters:输出的通道数
        k_size:卷积核大小
        '''
        #卷积核数量：out_filters,卷积核大小，步长1*1
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,strides=(1,1),
                      padding='same',kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)
    def bn_relu(x):
        x = BatchNormalization(momentum=0.9,epsilon=1e-5)(x) #数据的标准化，避免梯度消失或爆炸的情况出现
        x = Activation('relu')(x)
        return x
    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x) #BN-RELU
        x = conv(x,channels,(1,1)) #1*1的卷积层，增大通道数，即增多特征图的个数，将特征一步步映射到更高的维度
        x = bn_relu(x)#BN-RELU
        x = conv(x,growth_rate,(3,3)) #3*3的卷积层，最终通道数growth_rate=12
        return x

    def dense_layer(x):
        '''全连接层'''
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    def transition(x,inchannels):
        '''
        Parameters:
            x - 输入
            inchannels - 输入的通道数
        Transition层包括一个1x1的卷积和2x2的AvgPooling，结构为BN+ReLU+1x1 Conv+2x2 AvgPooling。
        另外，Transition层可以起到压缩模型的作用。
        假定Transition的上接DenseBlock得到的特征图channels数为 m ，Transition层可以产生 theta*m 个特征
        '''
        outchannels = int(inchannels*compressing) #计算输出的通道数
        x = bn_relu(x) #BN->RELU
        x = conv(x,outchannels,(1,1)) #1*1 卷积层
        x = AveragePooling2D((2,2),strides=(2,2))(x) #平均池化层
        return x,outchannels
    def dense_block(x,blocks,nchannels):
        '''
        dense_block层
        Parameters:
            x - 输入
            blocks - 有多少个中间卷积层
            nchannels - 输出通道数
        '''
        concat = x #输入
        #每个中间卷积层的函数：x = H([x0,x1,x2,...,x(l-1)]),其中H为:BN-RELU-1*1-BN-RELU-3*3
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat],axis=-1) #得到的结果进行通道上的拼接
            nchannels += growth_rate
        return concat,n_channels
    #共有多少了block块，总的网络深度减去（第一层卷积层，中间两层的transition层，还有最后的Linear层，剩下每个block中设置6个卷积层）
    n_blocks = (depth-4) // 6
    #DenseNet-BC中第一层卷积是后面最终输出通道的两倍
    n_channels = growth_rate * 2

    #第一层卷积: 输出n_channels通道，卷积核大小（3,3）
    x = conv(img_input,n_channels,(3,3))
    #dense_block1
    x,n_channels = dense_block(x,n_blocks,n_channels)
    #transition1
    x,n_channels = transition(x,n_channels)
    #dense_block2
    x,n_channels = dense_block(x,n_blocks,n_channels)
    #transition2
    x,n_channels = transition(x,n_channels)
    #dense_block3
    x,n_channels = dense_block(x,n_blocks,n_channels)
    x = bn_relu(x) #BN->RELU
    #全局平均池化
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

if __name__ == '__main__':
    '''
    #加载数据
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    #将label转换为one-hot编码
    y_train = keras.utils.to_categorical(y_train,num_classes)
    y_test = keras.utils.to_categorical(y_test,num_classes)
    #将数据集类型转换为float32类型
    x_train = x_train.astype('float32')
    x_test = x_train.astype('float32')

    #将数据标准化处理
    for i in range(3):
        x_train[:,:,:,i] = x_train[:,:,:,i] - mean[i]
    '''
    train_data, train_labels, test_data, test_labels = process_data()
    print('train_data_shape',train_data.shape)
    #创建网络
    inputs = Input(shape=(img_rows,img_cols,img_channels))
    output = DenseNet(inputs,num_classes)
    #定义模型对象
    model = Model(inputs,output)

    #打印模型的信息
    print(model.summary())

    #设置优化器对象
    sgd = optimizers.SGD(lr=.1,momentum=0.9,nesterov=True)
    #编译模型：交叉熵损失，SGD优化算法，衡量的指标accuracy
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    #设置Tensorboard对象
    tb_cb = TensorBoard(log_dir='./densenet/',histogram_freq=0)
    #调整learning_rate
    change_lr = LearningRateScheduler(scheduler)
    #存放日志文件
    ckpt = ModelCheckpoint('./ckpt.h5',save_best_only=False,mode='auto',period=10)
    cbks = [change_lr,tb_cb,ckpt]

    #进行数据增强
    print('Using real-time data argumentation')
    datagen = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.12,height_shift_range=0.125,fill_mode='constant',cval=0.)
    datagen.fit(train_data)
    #训练模型
    model.fit_generator(datagen.flow(train_data,train_labels,batch_size=batch_size),steps_per_epoch=iterations,
                                     callbacks=cbks,validation_data=(test_data,test_labels))
    model.save('densenet.h5')



