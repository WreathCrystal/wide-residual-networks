from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf



def variable(name, shape,  dtype, initializer, trainable):
    var = tf.get_variable(name,
                          shape=shape,
                          dtype=dtype,
                          initializer=initializer,
                          trainable=trainable)
    return var


def stride_arr(stride_h, stride_w):
    stride = [1, stride_h, stride_w, 1]
    return stride



def zero_pad(input_data, in_features, out_features):
    output_data = tf.pad(input_data, [[0, 0], [0, 0], [0, 0],
                                     [(out_features-in_features) // 2,
                                      (out_features-in_features) // 2]])
    return output_data



@tf.contrib.framework.add_arg_scope
def batch_norm(input_data,
                 decay=0.999,
                 center=True,
                 scale=False,
                 epsilon=0.001,
                 is_training=True,
                 reuse=None,
                 scope=None):
    return tf.contrib.layers.batch_norm(input_data,
                                          decay=decay,
                                          center=center,
                                          scale=scale,
                                          epsilon=epsilon,
                                          activation_fn=None,
                                          param_initializers=None,
                                          updates_collections=tf.GraphKeys.UPDATE_OPS,
                                          is_training=is_training,
                                          reuse=reuse,
                                          trainable=True,
                                          fused=True,
                                          data_format='NHWC',
                                          zero_debias_moving_mean=False,
                                          scope=scope)


@tf.contrib.framework.add_arg_scope
def conv2d(input_data,
           num_features_out,
           kernel_size,
           stride=1,
           scope=None,
           reuse=None):
    with tf.variable_scope(scope,'Conv',[input_data],reuse=reuse):
        num_channel = int(input_data.shape[3])
        weights_shape = [kernel_size, kernel_size, num_channel, num_features_out]
        
        n = int(weights_shape[0] * weights_shape[1] * weights_shape[3])
        weights_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n))
        
        weight = variable(name='weights',
                         shape=weights_shape,
                         dtype=tf.float32,
                         initializer = weights_initializer,
                         trainable =True)
    strides = stride_arr(stride, stride)
    output_data = tf.nn.conv2d(input_data, weight, strides, padding='SAME',data_format='NHWC')   
    return output_data

@tf.contrib.framework.add_arg_scope
def fc(input_data, 
       num_units_out,
       scope=None,
       reuse=None):
    with tf.variable_scope(scope, 'FC', [input_data], reuse=reuse):
        num_units_in = input_data.shape[1]
        weight_shape = [num_units_in, num_units_out]
        unif_init_range =  1.0 /(num_units_out**0.5)
        weight_initializer = tf.random_uniform_initializer(-unif_init_range,unif_init_range)
        weight = variable(name='weights',
                          shape= weight_shape,
                          dtype = tf.float32,
                          initializer = weight_initializer,
                          trainable=True)
        bias_shape = [num_units_out,]
        bias_initializer = tf.constant_initializer(0.0)
        bias = variable(name='bias',
                        shape = bias_shape,
                        dtype = tf.float32,
                        initializer = bias_initializer,
                        trainable=True)
        output_data = tf.nn.xw_plus_b(input_data,weight,bias)
        return output_data


@tf.contrib.framework.add_arg_scope
def avg_pool(input_data, 
             kernel_size, 
             stride=2, 
             padding='VALID',
             scope=None):
    with tf.name_scope(scope, 'AvgPool', [input_data]):
        kernel_size = stride_arr(kernel_size,kernel_size)
        strides = stride_arr(stride, stride)
        return tf.nn.avg_pool(input_data,
                           ksize = kernel_size,
                           strides = strides,
                           padding = padding,
                           data_format='NHWC')
    
    
    
    
@tf.contrib.framework.add_arg_scope
def global_avg_pool(input_data,
                    scope=None):
    with tf.name_scope(scope,'GlobalAvgPool',[input_data]):
        kernel_size = [1,int(input_data.shape[1]),int(input_data.shape[2]),1]
        queeze_dim=(1,2)
        output_data = tf.nn.avg_pool(input_data,
                                  ksize=kernel_size,
                                  strides=[1,1,1,1],
                                  padding='VALID',
                                  data_format='NHWC')
        return tf.squeeze(output_data,queeze_dim)
                                  
                       
                    
    
    

