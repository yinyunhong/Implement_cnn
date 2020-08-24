import tensorflow as tf
from tensorflow.python.training import moving_averages

UPDATE_OPS_COLLECTION = "_update_ops_"

#变量的声明函数tf.Variable,最终是一个张量。可以通过tf.global_variable()函数拿到当前计算图上的所有变量
#可以通过声明函数中的trainable参数来区分需要优化的参数（神经网络的参数）和其他的参数（迭代参数）
#若变量为True,则将这个变量加入到GraphKeys.TRAINABLE_VARIABLES集合中
#通过tf.trainable_variables()得到所有需要优化的参数
def create_variable(name,shape,initializer,dtype=tf.float32,trainable=True):
    '''
    创建变量的函数
    Parameters:
        name - 变量名称
        shape - 变量维度
        initializer - 初始化方式
        dtype - 变量类型
        trainable - 训练时是否进行更新
    '''
    #使用tf.get_variable()定义变量
    return tf.get_variable(name,shape=shape,dtype=dtype,initializer=initializer,trainable=trainable)

def batchNorm(inputs,scope,epsilon=1e-05,momentum=0.99,is_training=True):
    '''
    正则化函数
    Parameters:
        inputs - 输入
        scope - 变量作用域名称
        epsilon - 防止归一化时分母为0加的一个常量
    '''
    #得到输入数据的维度
    inputs_shape = inputs.get_shape().as_list()
    param_shape = inputs_shape[-1]
    axis = list(range(len(inputs_shape)-1)) #得到数据的维度
    #定义变量作用域
    with tf.variable_scope(scope):
        #定义BN中唯一需要学习的两个参数 y = gamma*x + beta
        gamma = create_variable('gamma',param_shape,initializer=tf.zeros_initializer())
        beta = create_variable('beta',param_shape,initializer=tf.ones_initializer())
        #创建变量 均值和方差
        moving_mean = create_variable("moving_mean",param_shape,initializer=tf.zeros_initializer())
        moving_variance = create_variable("moving_variance",param_shape,initializer=tf.ones_initializer())
    if is_training:
        #求得输入数据的均值和方差
        mean,variance = tf.nn.moments(inputs,axes=axis)
        #相当于计算:variable = variable * decay + value * (1 - decay)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,mean,decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,variance,decay=momentum)
        #将这两个变量添加到_update_op_集合中
        tf.add_to_collection(UPDATE_OPS_COLLECTION,update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION,update_move_variance)
    else:
        mean,variance = moving_mean,moving_variance
    return tf.nn.batch_normalization(inputs,mean,variance,beta,gamma,epsilon)

def depthwise_conv2d(inputs,scope,filter_size=3,channel_multiplier=1,strides=1):
    '''
    depthwise层卷积
    Parameters:
        inputs - 输入的数据
        scope - 变量作用域
        filter_size - 卷积核大小
        channel_multiplier - 相当于M
    '''
    #得到输入数据的维度
    input_shape = inputs.get_shape().aslist()
    in_channels = input_shape[-1]
    #定义变量作用域
    with tf.variable_scope(scope):
        filter = create_variable("filter",shape=[filter_size,filter_size,in_channels,channel_multiplier],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.depthwise_conv2d(inputs,filter,strides=[1,strides,strides,1],padding="SAME",rate=[1,1])

def conv2d(inputs,scope,num_filters,filter_size=1,strides=1):
    '''
    普通的卷积函数，默认下为1*1的卷积核（depthwise separable的第二阶段）
    Parameters:
        inputs - 输入的数据
        scope - 变量作用域的名称
        num_filters - 卷积核的数量
        filter_size - 卷积核大小
        strides - 卷积步长
    '''
    #输入图像的维度
    input_shape = inputs.get_shape().as_list()
    in_channels = input_shape[-1]
    #定义变量filter
    with tf.variable_scope(scope):
        #卷积核的权重初始化采用截断式正态分布
        filter = create_variable("filter",[filter_size,filter_size,in_channels,num_filters],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.conv2d(inputs,filter,strides=[1,strides,strides,1],padding="SAME")

def avg_pool(inputs,pool_size,scope):
    '''
    平均池化层
    Parameters:
        inputs - 输入
        pool_size - 池化大小
        scope - 变量作用域
    '''
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs,[1,pool_size,pool_size,1],strides=[1,pool_size,pool_size,1],padding="VALID")

def fc(inputs,n_out,scope,use_bias=True):
    '''
    全连接层
    Parameters:
        inputs - 输入数据
        n_out - 神经元个数
        scope - 变量作用域
        use_bias — 是否有偏置
    '''
    input_shape = inputs.get_shape().as_list()
    in_channels = input_shape[-1] #得到输入的维度
    with tf.variable_scope(scope):
        #定义全连接层的参数weight[in_channels,n_out]
        weight = create_variable("weight",shape=[in_channels,n_out],initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:
            bias = create_variable("bias",shape=[n_out],initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs,weight,bias) #有偏置时，返回w*x+b
        return tf.nn.matmul(inputs,weight) #没有偏置时，返回w*x

def depthwise_separable_conv2d(self,inputs,num_filters,width_multiplier,scope,downsample=False):
    '''
    深度可分离卷积
    '''
    num_filters = round(num_filters * width_multiplier) #卷积核的数量
    strides = 2 if downsample else 1  #如果进行下采样，则strides=2

    with tf.variable_scope(scope):
        #depthwise conv
        dw_conv = depthwise_conv2d(inputs,"depthwise_conv",strides=strides)
        relu = tf.nn.relu(batchNorm(dw_conv,"depth_conv/bn",is_training=self.is_training)) #正则化后，经过激活函数
        #pointwise conv
        pw_conv = conv2d(relu,"pointwise_conv",num_filters)
        return tf.nn.relu(batchNorm(pw_conv,"pw_conv/bn",is_training=self.is_training))#正则化后激活输出
#定义mobileNet模型类
class MobileNet(object):
    def __init__(self,inputs,num_classes=1000,is_training=True,width_multipler=1,scope="MobileNet"):
        '''
        inputs  - 输入数据
        num_classes - 分类数
        is_training - 模型是否需要训练
        width_multiple - 控制模型的大小
        scope - 变量的作用域
        '''
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.width_multiplier = width_multipler

        #创建模型
        with tf.variable_scope(scope):

            #conv1： 普通的卷积层，卷积核3*3，卷积核数量：32，步长：2
            net = conv2d(inputs,"conv_1",round(32*width_multipler),filter_size=3,strides=2)
            net = tf.nn.relu(batchNorm(net,"conv_1/bn",is_training=self.is_training))#经过BN规范正则化后，经过激励函数
            #得到[N,112,112,32]



            #深度可分离卷积
            #ds_conv_1
            net = depthwise_separable_conv2d(net,num_filters=64,width_multiplier=self.width_multiplier,scope="ds_conv_2")
            #ds_conv_2:下采样，步长为2的深度可分离卷积
            net = depthwise_separable_conv2d(net,num_filters=128,width_multiplier=self.width_multiplier,scope="ds_conv_3",downsample=True)
            #ds_conv_3:卷积核个数128
            net = depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                                   "ds_conv_3", downsample=True)  # ->[N, 56, 56, 128]
            #ds_conv_4:卷积核个数128
            net = depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                                   "ds_conv_4")  # ->[N, 56, 56, 128]
            #ds_conv_5:卷积核个数256，下采样
            net = depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                                   "ds_conv_5", downsample=True)  # ->[N, 28, 28, 256]
            #ds_conv_6:卷积核个数256
            net = depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                                   "ds_conv_6")  # ->[N, 28, 28, 256]
            #ds_conv_7:卷积核个数512，下采样
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_7", downsample=True)  # ->[N, 14, 14, 512]
            #ds_conv_8 :卷积核个数512
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_8")  # ->[N, 14, 14, 512]
            #ds_conv9 : 卷积核个数512
            net = depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_9")  # ->[N, 14, 14, 512]
            #ds_conv10、11、12: 卷积核个数512
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_10")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_11")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_12")  # ->[N, 14, 14, 512]
            #ds_conv13 : 卷积核个数1024 ,下采样
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                                   "ds_conv_13", downsample=True)  # ->[N, 7, 7, 1024]
            #ds_conv14 : 卷积核个数1024
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                                   "ds_conv_14")  # ->[N, 7, 7, 1024]


            #平均池化:池化后尺寸为1*1*1024
            net = avg_pool(net,7,"avg_pool_15")
            #压缩张量为1的轴
            net = tf.squeeze(net,[1,2],name="SpatialSqueeze")
            #全连接层: 得到每个分类的值
            self.logits = fc(net,self.num_classes,scope="fc_16")
            #经过softmax，得到每个分类的概率
            self.predictions = tf.nn.softmax(self.logits)

if __name__ == "__main__":
    inputs = tf.random_normal(shape=[4,224,224,3]) #输入的数据
    #实例化MobileNet对象
    mobileNet = MobileNet(inputs)
    #指定保存模型图的文件夹
    writer = tf.summary.FileWriter("./logs",graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred = sess.run(mobileNet.predictions)