from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import augmentation_transforms
import numpy as np
import policies as found_policies
import tensorflow as tf


class DataSet(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.epochs = 0
        self.curr_train_index = 0
        
        all_labels = []
        
        self.good_policies = found_policies.good_policies()
        
        total_dataset_size = 50000 
        train_dataset_size = 50000
        if hparams.eval_test:
            total_dataset_size+= 10000
        
        all_data = np.empty((1, 50000, 3072), dtype=np.uint8)
        if hparams.eval_test:
            test_data = np.empty((1, 10000, 3072), dtype=np.uint8)
        
        datafiles = ['train']  # 'train'  is the  filename of train dataset
        if hparams.eval_test:
            datafiles.append('test')  # 'test' is the filename of test dataset
        
        num_classes = 100   # fine labels 
        
        # Loading train and test dataset
        for file_num, f in enumerate(datafiles):
            d = unpickle(os.path.join(hparams.data_path, f))
            if f == 'test':
                test_data[0] = copy.deepcopy(d['data'])
                all_data = np.concatenate([all_data, test_data], axis=1)
            else:
                all_data[file_num] = copy.deepcopy(d['data'])
            
            labels = np.array(d['fine_labels'])
            nsamples = len(labels)
            for idx in range(nsamples):
                all_labels.append(labels[idx])
                
        # Data processing 
        all_data = all_data.reshape(total_dataset_size, 3072)
        all_data = all_data.reshape(-1, 3, 32, 32)
        all_data = all_data.transpose(0, 2, 3, 1).copy()
        all_data = all_data / 255.0
        mean = augmentation_transforms.MEANS
        std = augmentation_transforms.STDS
        tf.logging.info('mean:{}    std: {}'.format(mean, std))

        all_data = (all_data - mean) / std
        all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
        assert len(all_data) == len(all_labels)
        tf.logging.info(
            'In CIFAR100 loader, number of images: {}'.format(len(all_data)))
        
        # Break off test data
        if hparams.eval_test:
            self.test_images = all_data[train_dataset_size:]
            self.test_labels = all_labels[train_dataset_size:]
            
        # Shuffle the rest of the data
        all_data = all_data[:train_dataset_size]
        all_labels = all_labels[:train_dataset_size]
        np.random.seed(0)
        perm = np.arange(len(all_data))
        np.random.shuffle(perm)
        all_data = all_data[perm]
        all_labels = all_labels[perm]

        train_size, val_size = hparams.train_size, hparams.validation_size
        self.train_images = all_data[:train_size]
        self.train_labels = all_labels[:train_size]
        self.num_train = self.train_images.shape[0]
        
    def next_batch(self):
        next_train_index = self.curr_train_index + self.hparams.batch_size
        if next_train_index > self.num_train:
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
        batched_data = (
            self.train_images[self.curr_train_index:self.curr_train_index + self.hparams.batch_size],
            self.train_labels[self.curr_train_index:self.curr_train_index + self.hparams.batch_size])
            
        final_imgs = []
        
        images, labels = batched_data
        # Data transformation
        for data in images:
            epoch_policy = self.good_policies[np.random.choice(len(self.good_policies))]
            final_img = augmentation_transforms.apply_policy(epoch_policy,data)
            final_img = augmentation_transforms.random_flip(
                augmentation_transforms.zero_pad_and_crop(final_img, 4))
            final_img = augmentation_transforms.cutout_numpy(final_img)
            final_imgs.append(final_img)
        batched_data = (np.array(final_imgs, np.float32), labels)
        self.curr_train_index += self.hparams.batch_size
        return batched_data
    
    def reset(self):
        self.epochs = 0
        # Shuffle the training data
        perm = np.arange(self.num_train)
        np.random.shuffle(perm)
        assert self.num_train == self.train_images.shape[0], 'Error: incorrect shuffling mask'
        self.train_images = self.train_images[perm]
        self.train_labels = self.train_labels[perm]
        self.curr_train_index = 0

"""   """
        
# For python2 env   
import cPickle
def unpickle(f):
    tf.logging.info('loading file: {}'.format(f))
    fo = tf.gfile.Open(f, 'r')
    d = cPickle.load(fo)
    fo.close()
    return d



"""  
# For python3 env
import pickle
def unpickle(f):
    tf.logging.info('loading file: {}'.format(f))
    fo = tf.gfile.Open(f, 'rb')
    d = pickle.load(fo, encoding='iso-8859-1')
    fo.close()
    return d
"""
 