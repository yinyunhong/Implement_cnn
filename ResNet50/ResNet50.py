'''模型的输入：64*64*3；模型的输出：6种分类，6个节点'''
import tensorflow as tf
from resnets_utils import *
import numpy as np
from data_process import process_data

'''
Resnet50:
有两个基本的block,包括identity block(输入和输出是一样的)，可以串联多个
另一个是conv block,输入的输出是不一样的，所以不能直接串联
    因为CNN最后都是要把输入的图像，转换为尺寸很小但是depth很深的的feature map
    随着网络的加深，输出的通道数也越来越大，所以需要在进行identity block前，用conv block转换一下维度，才能在后面接identity block
'''
#变量TRANING:类型bool,默认值False
TRAINING = tf.Variable(initial_value=True,dtype=tf.bool,trainable=False)

#实体残差块的实现
def identity_block(x,kernel_size,out_filters,stage,block):
    '''
    x_shortcut和输入x一样
    1*1 -> kernel_size*kernel_size -> 1*1
    Parameters:
        x-输入
        kernel_size-卷积核大小
        out_filters:输出的维度，卷积核的个数
        stage:第几块
        training:
    identity_block(x,3,[64,64,256],stage=2,block='b')
    '''
    conv_name_base = 'res'+str(stage) + block #卷积块名
    bn_name_base = 'bn' + str(stage) + block #普通块名
    f1,f2,f3 = out_filters #三个卷积层，每个的输出维度
    with tf.variable_scope("id_block_stage"+str(stage)):
        x_shortcut = x  #shortcut

        #conv1: 卷积核大小：1*1，步长为1，前后尺寸不变
        x = tf.layers.conv2d(x,f1,kernel_size=(1,1),strides=(1,1),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2a',training=TRAINING)
        x = tf.nn.relu(x)

        #conv2：卷积核大小：kernel_size*kernel_size,步长为1，same填充，前后尺寸不变
        x = tf.layers.conv2d(x,f2,kernel_size=(kernel_size,kernel_size),strides=[1,1],padding='SAME',name=conv_name_base+'2b')
        #x = tf.layers.batch_normalization(x,axis=3,training=training)
        x = tf.nn.relu(x)

        #conv3：卷积核大小：1*1，步长为1，前后尺寸不变
        x = tf.layers.conv2d(x,f3,kernel_size=(1,1),strides=[1,1],name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x,axis=3,training=TRAINING,name=bn_name_base+'2c')

        #add
        add = tf.add(x,x_shortcut)
        add_result = tf.nn.relu(add)
    return add_result

#***********shortcut含有卷积层*********************
def convolutional_block(x,kernel_size,out_filters,stage,block,stride=2):
    '''
    卷积残差块：允许输入和输出的尺寸维度不一致，所以stride可以不为1
    1*1(stride) -> kernel_size*kernel_size(stride)  -> 1*1(stride)
    conv_shortcut: (1*1)stride , filter_nums = f3 将输入的通道数转换为与残差块相同的输出
    convolutional_block(x,kernel_size=3,out_filters=[64,64,256],stage=2,block='a',stride=1)
    '''
    conv_name_base = 'res' + str(stage) + block #res2a
    bn_name_base = 'bn'+ str(stage) + block
    f1,f2,f3 = out_filters
    #定义命名空间
    with tf.variable_scope("conv_block_stage"+str(stage)):
        x_shortcut = x

        #conv1:卷积核1*1， 步长：stride 
        x = tf.layers.conv2d(x,f1,kernel_size=(1,1),strides=(stride,stride),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2a',training=TRAINING)
        x = tf.nn.relu(x)

        #conv2:卷积核：kernel_size*kernel_size, 步长：stride, 填充方式：same
        x = tf.layers.conv2d(x,f2,kernel_size=(kernel_size,kernel_size),strides=(stride,stride),name=conv_name_base+'2b',padding='same')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2b',training=TRAINING)
        x = tf.nn.relu(x)

        #conv3:卷积核：1*1, 步长：stride
        x = tf.layers.conv2d(x,f3,kernel_size=(1,1),name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x,axis=3,name=bn_name_base+'2c',training=TRAINING)

        #conv_shortcut: 输入的维度input_filter,输出的维度f3,  步长：stride
        x_shortcut = tf.layers.conv2d(x_shortcut,f3,(1,1),strides=(stride,stride),name=conv_name_base+'1')
        x_shortcut = tf.layers.batch_normalization(x_shortcut,axis=3,name=bn_name_base+'1',training = TRAINING)

        x_shortcut = tf.add(x_shortcut,x)
        add_result = tf.nn.relu(x_shortcut)
    return add_result

def ResNet50_reference(x,classes=6):
    '''
    ResNet50架构：
    conv2D -> BN -> RELU -> MAXPOOL -> conv_block -> IDblock*2 -> conv_block -> IDblock*3
    -> conv_block -> IDblock*5 -> conv_block -> IDblock*2 ->avgpool -> toplayer
    '''
    #x = tf.pad(x,tf.constant([[0,0],[3,3],[3,3],[0,0]]),"CONSTANT")
    #assert(x.shape == (x.shape[0],70,70,3))  #检查图片的维度是否为：num*70*70*3

    #32*32*3 -> 12*12*3
    #stage1:卷积层（卷积核：7*7*64）
    x = tf.layers.conv2d(x,filters=64,kernel_size=(7,7),strides=(2,2),name='conv1')
    x = tf.layers.batch_normalization(x,axis=3,name='bn_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2)) #2*2*3

    #stage2：(conv_block)，通道数发生改变，经过三层卷积由64->256，shortcut增加1*1的卷积层相应变化
    #       identity_block:叠加两个该层，通道数不发生改变，仍为256
    x = convolutional_block(x,kernel_size=3,out_filters=[64,64,256],stage=2,block='a',stride=1)
    x = identity_block(x,3,[64,64,256],stage=2,block='b')
    x = identity_block(x,3,[64,64,256],stage=2,block='c')

    #stage3：conv_block(通道数发生改变由256->128->128->512)
    #       identity_block(叠加三层该层，通道数仍为512)
    x = convolutional_block(x, kernel_size=3, out_filters=[128, 128, 512],
                            stage=3, block='a', stride=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # stage 4：conv_block:通道数 512->256->256->1024
    #           叠加5层identity_block
    x = convolutional_block(x, kernel_size=3, out_filters=[256, 256, 1024], stage=4, block='a', stride=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5:conv_block:通道数 1023->512->512->2048
    #         identity_block:叠加两层
    x = convolutional_block(x, kernel_size=3, out_filters=[512, 512, 2048], stage=5, block='a', stride=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = tf.layers.average_pooling2d(x,pool_size=(2,2),strides=(1,1)) #池化

    #卷积层和全连接层中间加入flatten,展平为一维输入
    flatten = tf.layers.flatten(x,name='flatten')
    dense1 = tf.layers.dense(flatten,units=50,activation=tf.nn.relu) #全连接层
    logits = tf.layers.dense(dense1,units=6,activation=tf.nn.softmax)
    return logits

if __name__ == '__main__':
    x_train,y_train,x_test,y_test = process_data() #对数据进行预处理
    m,H_size,W_size,C_size = x_train.shape
    classes = 6
    assert((H_size,W_size,C_size) == (32,32,3))
    print(H_size,W_size,C_size)

    #定义运行时需要填充的变量 32*32*3
    x = tf.placeholder(tf.float32,shape=(None,H_size,W_size,C_size),name='X')
    y = tf.placeholder(tf.float32,shape=(None,classes),name='Y')
    
    #得到网络的输出结果
    logits = ResNet50_reference(x)
    
    #求损失
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y,logits=logits))
    
    #最小化损失: 学习率0.001
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    
    #准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,axis=1),tf.argmax(y,axis=1)),tf.float32))
    
    #运行会话
    with tf.Session() as sess:
        sess.run(tf.global_variable_initializer()) #初始化全局变量
        assert(x_train.shape == (x_train.shape[0],32,32,3))
        assert(y_train.shape[1] == 10)
        #取本次需要训练的样本数据
        mini_batches = random_mini_batches(x_train,y_train,mini_batch_size=32)
        
        for i in range(10000):
            x_mini_batch,y_mini_batch = mini_batches[np.random.randint(0,len(mini_batches))]
            #最小化损失  和  计算最后的损失
            _,loss_ = sess.run([train_op,loss],feed_dict={x:x_mini_batch, y:y_mini_batch}) #最小化损失，计算得到损失值
            # 每隔50次打印一下损失值
            if i%50 == 0:
                print(i,loss)
            sess.run(tf.assign(TRAINING,False)) #设置trainable=False
            
            #得到训练准确率
            training_acc = sess.run(accuracy,feed_fict={x:x_train,y:y_train})
            test_acc = sess.run(accuracy,feed_fict={x:x_test,y:y_test})
            print('第{}次迭代，训练集准确率：{.2f}'.format(i,training_acc))
            print('第{}次迭代，测试集准确率：{:.2f}'.format(i,test_acc))
