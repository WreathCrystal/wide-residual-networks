from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import custom_ops as ops
import numpy as np
import tensorflow as tf



def res_block(x,
              in_features,
              out_features,
              stride,
              activate_before_res=False):
    
    if activate_before_res:
        with tf.variable_scope('shared_activation'):
            x = ops.batch_norm(x,scope='init_bn')
            x = tf.nn.relu(x)
            orig_x =x
    else:
        orig_x = x
        
    block_x = x
    if not activate_before_res:
        with tf.variable_scope('res_only_activation'):
            block_x = ops.batch_norm(block_x,scope='init_bn')
            block_x = tf.nn.relu(block_x)
        
           
    with tf.variable_scope('sub1'):
        block_x = ops.conv2d(block_x,out_features,3,stride=stride,scope='conv1')
        
    with tf.variable_scope('sub2'):
        block_x = ops.batch_norm(block_x,scope='bn2')
        block_x = tf.nn.relu(block_x)
        block_x = ops.conv2d(block_x,out_features, 3, stride=1, scope='conv2')
        
    with tf.variable_scope('sub_add'):
        if in_features != out_features:
            orig_x = ops.avg_pool(orig_x, stride, stride)
            orig_x = ops.zero_pad(orig_x, in_features, out_features)
    output_data = orig_x + block_x
    return output_data


def res_add(x,  orig_x, in_features, out_features, stride):
    if in_features != out_features:
        orig_x = ops.avg_pool(orig_x, stride,stride)
        orig_x = ops.zero_pad(orig_x, in_features,out_features)
    x = x + orig_x
    orig_x = x
    return x, orig_x



def build_wrn_model(input_data, num_classes,  wrn_size):   
    kernel_size = 3
    features = [min(wrn_size, 16), wrn_size,  wrn_size * 2, wrn_size * 4]
    strides = [1, 2, 2]  # stride for each resblock
    
    # create the first convolutional layer
    with tf.variable_scope('init'):
        x = ops.conv2d(input_data, features [0], kernel_size, scope='init_conv')
    
    first_x = x
    orig_x =x
    
    # create 2nd, 3rd, 4th resnet layers, two convs per res_block,  four blocks per res layer.  n=(28-4)/6
    for num_res_layer in range(1,4):
        with tf.variable_scope('unit_{}_0'.format(num_res_layer)):
            activate_before_res = True if num_res_layer == 1 else False
            x = res_block(orig_x,
                          features[num_res_layer-1],
                          features[num_res_layer],
                          stride = strides[num_res_layer-1],
                          activate_before_res = activate_before_res)
            for num_block in range(1,4):
                with tf.variable_scope('unit_{}_{}'.format(num_res_layer,num_block)):
                    x = res_block(x,
                                  features[num_res_layer],
                                  features[num_res_layer],
                                  1,
                                  activate_before_res = False)
        x, orig_x = res_add(x, orig_x, features[num_res_layer-1], features[num_res_layer], strides[num_res_layer-1])
    final_stride = np.prod(strides)
    x , _ = res_add(x, first_x, features[0],features[3],final_stride)
    with tf.variable_scope('unit_last'):
        x = ops.batch_norm(x, scope='final_bn')
        x = tf.nn.relu(x)
        x = ops.global_avg_pool(x)
        logits = ops.fc(x, num_classes)
    return logits
        