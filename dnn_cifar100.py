
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import time

import custom_ops as ops
import dataset
import training_base
import numpy as np
import tensorflow as tf
from wrn import build_wrn_model

tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')

FLAGS = tf.flags.FLAGS

arg_scope = tf.contrib.framework.arg_scope


def setup_arg_scopes(is_training):
    batch_norm_decay = 0.9
    batch_norm_epsilon =1e-5
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': True,
        'is_training': is_training,
    }
    scopes = []
    scopes.append(arg_scope([ops.batch_norm], **batch_norm_params))
    return scopes

class Model(object):
    def __init__(self, hparams):
        self.hparams = hparams
    
    def build(self, mode):
        assert mode in ['train', 'eval']
        self.mode = mode
        self._setup_misc(mode)
        self._setup_images_and_labels()
        self._build_graph(self.images, self.labels, mode)
        
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      
    def _setup_misc(self, mode):
        self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable = False)
        self.reuse = None if (mode=='train') else True
        self.batch_size = self.hparams.batch_size
        if mode == 'eval':
             self.batch_size = 25
        
    def _setup_images_and_labels(self):
        self.num_classes = self.hparams.num_classes
        self.images = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.num_classes])
    
    
            
    def _build_graph(self, images, labels, mode):
        is_training = 'train' in mode
        if is_training:
            self.global_step = tf.train.get_or_create_global_step()
        
        scopes = setup_arg_scopes(is_training)
        with scopes[0]:
            logits = build_wrn_model(images,
                                self.num_classes,
                                self.hparams.wrn_size)
        self.predictions, self.loss = training_base.setup_loss(logits, labels)
        self.accuracy, self.eval_op = tf.metrics.accuracy(
            tf.argmax(labels,1), tf.argmax(self.predictions, 1))
        
        
        self.loss = training_base.decay_weights(self.loss, self.hparams.weight_decay_rate)
        tf.logging.info('is_training : {}'.format(is_training))
        if is_training:
            self._build_train_op()
        
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(max_to_keep = 2)
            
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            
    def _build_train_op(self):
        hparams = self.hparams
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        if hparams.gradient_clipping_by_global_norm > 0.0:
            grads, norm = tf.clip_by_global_norm(grads,
                                                 hparams.gradient_clipping_by_global_norm)
            tf.summary.scalar('grad_norm', norm)
        initial_lr = self.lr_rate_ph
        optimizer = tf.train.MomentumOptimizer(initial_lr,
                                               0.9,
                                               use_nesterov=True)
        self.optimizer = optimizer
        apply_op = optimizer.apply_gradients(zip(grads, tvars),
                                             global_step = self.global_step, 
                                             name='train_step')
        train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([apply_op]):
            self.train_op = tf.group(*train_ops)

            
            
class Trainer(object):
    
    def __init__(self, hparams):        
        self.sess = tf.Session()
        self.hparams = hparams
        self.model_dir = os.path.join(self.hparams.checkpoint_dir, 'model')
        self.log_dir = os.path.join(self.hparams.checkpoint_dir, 'log')
        
        np.random.seed(0)
        self.data_loader = dataset.DataSet(hparams)
        np.random.seed()
        self.data_loader.reset()
   
    def save_model(self, step=None):
        model_save_name = os.path.join(self.model_dir, 'model.ckpt')
        if not tf.gfile.IsDirectory(self.model_dir):
            tf.gfile.MakeDirs(self.model_dir)
        self.saver.save(self.sess, model_save_name, global_step = step)
        tf.logging.info('Saved child model')
    
    def extract_model_spec(self):
        checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
        if checkpoint_path is not None:
            self.saver.restore(self.sess, checkpoint_path)
            tf.logging.info('Loaded child model checkpoint from %s', checkpoint_path)
        else:
            self.save_model(step=0)
    
    @contextlib.contextmanager
    def _new_session(self, m):
        self.sess = tf.Session('',
                                   config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.sess.run(m.init)
        self.extract_model_spec()
        try:
            yield
        finally:
            tf.Session.reset('')
            self.sess = None
     
    def _run_training_loop(self, m, curr_epoch):
        start_time = time.time()
        while True:
            try:
                with self._new_session(m):
                    train_accuracy = training_base.run_epoch_training(self.sess, m, self.data_loader, curr_epoch)
                    tf.logging.info('Saving model after epoch')
                    self.save_model(step=curr_epoch)
                    break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info('Retryable error caught: %s. Retrying.', e)
        tf.logging.info('Finished epoch: {}'.format(curr_epoch))
        tf.logging.info('Epoch time(min): {}'.format((time.time() - start_time) / 60.0))
        return train_accuracy
    
    def _calc_starting_epoch(self, m):
        hparams = self.hparams
        batch_size = hparams.batch_size
        steps_per_epoch = int(hparams.train_size / batch_size)
        with self._new_session(m):
            curr_step = self.sess.run(m.global_step)
        total_steps = steps_per_epoch * hparams.num_epochs
        epochs_left = (total_steps - curr_step) // steps_per_epoch
        starting_epoch = hparams.num_epochs - epochs_left
        return starting_epoch
    
    
    def run_model(self):            
        hparams = self.hparams
        
        with tf.Graph().as_default(), tf.device('/cpu:0' if FLAGS.use_cpu  else '/gpu:0'):
            with tf.variable_scope('model', use_resource = False):
                m = Model(self.hparams)
                m.build('train')
                self._saver = m.saver
            starting_epoch = self._calc_starting_epoch(m)
            training_acc = None
            
            for curr_epoch in range(starting_epoch, hparams.num_epochs):
                training_acc = self._run_training_loop(m, curr_epoch)
            
            with tf.variable_scope('model', reuse=True, use_resource=False):
                meval = Model(self.hparams)
                meval.build('eval')
            
            with self._new_session(meval):
                test_acc = training_base.eval_child_model(self.sess, meval, self.data_loader)
          
        tf.logging.info('Train Acc: {}       Test Acc: {}'.format(training_acc,  test_acc))
            

    @property
    def saver(self):
        return self._saver

    @property
    def session(self):
        return self.sess
    
def main(_):
    hparams = tf.contrib.training.HParams(
        train_size=50000,
        validation_size=0,
        eval_test=1,
        dataset='cifar100',
        num_classes=100,
        data_path='/tmp/data',
        checkpoint_dir='/tmp/training',
        batch_size=128,
        gradient_clipping_by_global_norm=5.0,
        model_name='wrn',
        num_epochs=200,
        wrn_size=160,
        lr=0.1,
        weight_decay_rate=5e-4)
    cifar_trainer = Trainer(hparams)
    cifar_trainer.run_model()
        

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()