
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def  setup_loss(logits, labels):
    predictions = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)
    return predictions, loss



def decay_weights(loss, weight_decay_rate):
    losses =[]
    for var in tf.trainable_variables():
        losses.append(tf.nn.l2_loss(var))
    loss += tf.multiply(weight_decay_rate,  tf.add_n(losses))
    return loss


def eval_child_model(sess, model, data_loader):
    images = data_loader.test_images
    labels = data_loader.test_labels
    assert len(images) == len(labels)
    tf.logging.info('model.batch_size is {}'.format(model.batch_size))
    assert len(images) % model.batch_size == 0
    test_batches = int(len(images) / model.batch_size)
    for i in range(test_batches):
        test_images = images[i * model.batch_size:(i + 1) * model.batch_size]
        test_labels = labels[i * model.batch_size:(i + 1) * model.batch_size]
        _ = sess.run(model.eval_op,
                    feed_dict = {model.images:test_images,model.labels:test_labels,})
    return sess.run(model.accuracy)


def cosine_lr(learning_rate, epoch, iteration, batches_per_epoch, total_epochs):
    t_total = total_epochs * batches_per_epoch
    t_cur = float(epoch * batches_per_epoch + iteration)
    return 0.5 * learning_rate * (1+ np.cos(np.pi * t_cur / t_total))



def get_lr(curr_epoch, hparams, iteration=None):
    assert iteration is not None
    batches_per_epoch = int(hparams.train_size / hparams.batch_size)
    lr = cosine_lr(hparams.lr, curr_epoch, iteration, batches_per_epoch,hparams.num_epochs)
    return lr


def run_epoch_training(sess, model, data_loader, curr_epoch):
    steps_per_epoch = int(model.hparams.train_size / model.hparams.batch_size)
    tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
    curr_step = sess.run(model.global_step)
    print('curr_step:  ',curr_step)
    assert curr_step % steps_per_epoch == 0
    
    curr_lr = get_lr(curr_epoch,model.hparams, iteration=0)
    tf.logging.info('lf of {} for epoch {}'.format(curr_lr, curr_epoch))
    
    for step in range(steps_per_epoch):
        curr_lr = get_lr(curr_epoch, model.hparams, iteration=(step + 1))
        model.lr_rate_ph.load(curr_lr,session =sess)
        if step % 20 == 0:
            tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))
            
        train_images, train_labels = data_loader.next_batch()
        _, step, _= sess.run([model.train_op, model.global_step, model.eval_op],
                            feed_dict={model.images: train_images, model.labels: train_labels,})
        
    train_accuracy = sess.run(model.accuracy)           
    tf.logging.info('Train accuracy: {}'.format(train_accuracy))
    return train_accuracy
    