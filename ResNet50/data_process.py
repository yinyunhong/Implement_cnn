import tensorflow as tf
import numpy as np
import time
import random
import pickle
import math
import datetime
from keras.preprocessing.image import ImageDataGenerator
import os
#初始化的变量

class_num = 10  #分类数目
image_size = 32 #图像尺寸
img_channels = 3 #图像通道数
iterations = 200    #迭代次数
batch_size = 250    #批数量
weight_decay = 0.0003   #权重衰减率
dropout_rate = 0.5  #dropout丢弃率
momentum_rate = 0.9 #训练时动量率
data_dir = './VGGNet/cifar-10-batches-py/' #存放数据集的文件夹
log_save_path = './DenseNet_logs' #存放训练日志的文件夹
model_save_path = './DenseNet_model/'    #存放模型的路径

#********************数据预处理****************
def load_file(filename):
    '''读取文件,将文件反序列化，得到数据字典'''
    with open(filename,'rb') as f:
        data_dict = pickle.load(f,encoding='bytes')
    return data_dict

def parse_data(filename):
    '''根据文件名，调用load_file()反序列化数据，并解析出数据data和标签labels返回'''
    data_dict = load_file(filename)
    image = data_dict[b'data']  #数据
    labels = data_dict[b'labels']   #标签
    print("file:%s has %d data" % (filename,len(labels)))
    return image,labels

def load_data(files,data_dir,classNum):
    '''
    读取data_batch_1 ~ data_batch_5的内容，并将数据转化成模型可以输出的格式并归一化，labels转化成[-1,10]的浮点数组
    Parameters:
        files:存放data_batch_1 ~ 5 文件名的列表
        data_dir: 存放数据的目录名
        classNum: 最终分类数目
    returns:
        data:处理好的数据
        labels:处理好的标签
    '''
    data,labels = parse_data(data_dir + files[0]) #得到第一组数据和标签
    if len(files) > 1:
        for i in range(1,5):
            data1,labels1 = parse_data(os.path.join(data_dir,files[i])) #得到剩余组的数据和标签，并按行添加数据
            data = np.append(data,data1,axis=0)
            labels = np.append(labels,labels1,axis=0)
    print(data.shape)

    #将数据集转换为图片向量格式
    data = data.reshape(-1,3,32,32).transpose(0,2,3,1)
    #将标签集转换成[-1,10]的数组,对应分类位置1，其他位置0
    labels = np.array([[float(label == i)for i in range(classNum)]for label in labels])

    #对图片的各个通道进行归一化
    for i in range(3):
        data[:,:,:,i] = (data[:,:,:,i] - np.mean(data[:,:,:,i])) / np.std(data[:,:,:,i])
    return data,labels

def process_data():
    '''对数据处理的总函数'''
    print("loading data......")
    image_dim = image_size * image_size * img_channels
    meta = load_file(data_dir + 'batches.meta') #对batches.meta反序列化,meta中信息记录了包含的几个文件名
    labels_name = meta[b'label_names']  #10中label的名称
    files = ['data_batch_%d' % i for i in range(1,6) ]
    train_data,train_labels = load_data(files,data_dir,class_num)   #加载所有的训练集数据 和 标签
    #加载测试集数据和标签
    file = data_dir + 'test_batch'
    test_data,test_labels = load_data(['test_batch'],data_dir,class_num)
    print("train_shape:",np.shape(train_data),np.shape(train_labels))
    print("test_shape:",np.shape(test_data),np.shape(test_labels))

    #随机打乱训练集的顺序
    index = np.random.permutation(len(train_data))  #打乱索引排列顺序
    train_data = train_data[index]
    train_labels = train_labels[index]
    print("loading data successfully....")

    return train_data,train_labels,test_data,test_labels


