# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/15$ 12:12$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : model_darknet19$.py
# Description :yolo2网络模型——darknet19.
# --------------------------------------

import os
import tensorflow as tf
import numpy as np

################# 基础层：conv/pool/reorg(带passthrough的重组层) #############################################
# 激活函数
def leaky_relu(x):
	return tf.nn.leaky_relu(x,alpha=0.1,name='leaky_relu') # 或者tf.maximum(0.1*x,x)

# Conv+BN：yolo2中每个卷积层后面都有一个BN层
def conv2d(x,filters_num,filters_size,pad_size=0,stride=1,batch_normalize=True,
		   activation=leaky_relu,use_bias=False,name='conv2d'):
	# padding，注意: 不用padding="SAME",否则可能会导致坐标计算错误
	if pad_size > 0:
		x = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
	# 有BN层，所以后面有BN层的conv就不用偏置bias，并先不经过激活函数activation
	out = tf.layers.conv2d(x,filters=filters_num,kernel_size=filters_size,strides=stride,
						   padding='VALID',activation=None,use_bias=use_bias,name=name)
	# BN，如果有，应该在卷积层conv和激活函数activation之间
	if batch_normalize:
		out = tf.layers.batch_normalization(out,axis=-1,momentum=0.9,training=False,name=name+'_bn')
	if activation:
		out = activation(out)
	return out

# max_pool
def maxpool(x,size=2,stride=2,name='maxpool'):
	return tf.layers.max_pooling2d(x,pool_size=size,strides=stride)

# reorg layer(带passthrough的重组层)
def reorg(x,stride):
	return tf.space_to_depth(x,block_size=stride)
	# 或者return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],
	# 								rates=[1,1,1,1],padding='VALID')
#########################################################################################################

################################### Darknet19 ###########################################################
# 默认是coco数据集，最后一层维度是anchor_num*(class_num+5)=5*(80+5)=425
def darknet(images,n_last_channels=425):
	net = conv2d(images, filters_num=32, filters_size=3, pad_size=1, name='conv1')
	net = maxpool(net, size=2, stride=2, name='pool1')

	net = conv2d(net, 64, 3, 1, name='conv2')
	net = maxpool(net, 2, 2, name='pool2')

	net = conv2d(net, 128, 3, 1, name='conv3_1')
	net = conv2d(net, 64, 1, 0, name='conv3_2')
	net = conv2d(net, 128, 3, 1, name='conv3_3')
	net = maxpool(net, 2, 2, name='pool3')

	net = conv2d(net, 256, 3, 1, name='conv4_1')
	net = conv2d(net, 128, 1, 0, name='conv4_2')
	net = conv2d(net, 256, 3, 1, name='conv4_3')
	net = maxpool(net, 2, 2, name='pool4')

	net = conv2d(net, 512, 3, 1, name='conv5_1')
	net = conv2d(net, 256, 1, 0,name='conv5_2')
	net = conv2d(net,512, 3, 1, name='conv5_3')
	net = conv2d(net, 256, 1, 0, name='conv5_4')
	net = conv2d(net, 512, 3, 1, name='conv5_5')
	shortcut = net # 存储这一层特征图，以便后面passthrough层
	net = maxpool(net, 2, 2, name='pool5')

	net = conv2d(net, 1024, 3, 1, name='conv6_1')
	net = conv2d(net, 512, 1, 0, name='conv6_2')
	net = conv2d(net, 1024, 3, 1, name='conv6_3')
	net = conv2d(net, 512, 1, 0, name='conv6_4')
	net = conv2d(net, 1024, 3, 1, name='conv6_5')

	net = conv2d(net, 1024, 3, 1, name='conv7_1')
	net = conv2d(net, 1024, 3, 1, name='conv7_2')
	# shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
	# 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图
	shortcut = conv2d(shortcut, 64, 1, 0, name='conv_shortcut')
	shortcut = reorg(shortcut, 2)
	net = tf.concat([shortcut, net], axis=-1) # channel整合到一起
	net = conv2d(net, 1024, 3, 1, name='conv8')

	# detection layer:最后用一个1*1卷积去调整channel，该层没有BN层和激活函数
	output = conv2d(net, filters_num=n_last_channels, filters_size=1, batch_normalize=False,
				 activation=None, use_bias=True, name='conv_dec')

	return output
#########################################################################################################

if __name__ == '__main__':
	x = tf.random_normal([1, 416, 416, 3])
	model_output = darknet(x)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		# 必须先restore模型才能打印shape;导入模型时，上面每层网络的name不能修改，否则找不到
		saver.restore(sess, "./yolo2_model/yolo2_coco.ckpt")
		print(sess.run(model_output).shape) # (1,13,13,425)
