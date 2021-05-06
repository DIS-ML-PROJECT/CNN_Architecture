from models import loss_utils
from utils.analysis import evaluate
from utils.run import load, run_batches

import time
from typing import Mapping, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


class BaseTrainer(object):
    def __init__(self, train_batch, train_eval_batch, val_batch,
                 train_model, train_eval_model, val_model,
                 train_preds, train_eval_preds, val_preds,
                 sess, steps_per_epoch, ls_bands, nl_band, learning_rate, lr_decay,
                 log_dir, save_ckpt_prefix, init_ckpt_dir, imagenet_weights_path,
                 hs_weight_init, exclude_final_layer, image_summaries,
                 loss_fn, loss_type, results_cols):
        '''
        Args
        - train_batch / train_eval_batch / val_batch: dict of tf.Tensor
            images: tf.Tensor, shape [N, H, W, C]
            labels: tf.Tensor, shape [N] or [N, label_dim]
            locs: tf.Tensor, shape [N, 2]
        - train_model / train_eval_model / val_model: BaseModel
        - train_preds / train_eval_preds / val_preds: tf.Tensor, shape [N]
        - sess: tf.Session
        - steps_per_epoch: numeric
        - ls_bands: one of [None, 'rgb', 'ms']
        - nl_band: one of [None, 'merge', 'split']
        - learning_rate: float
        - lr_decay: float
        - log_dir: str, path to log directory
        - save_ckpt_prefix: str
        - init_ckpt_dir: str, path to directory with saved checkpoint
        - imagenet_weights_path: str
        - hs_weight_init: str
        - exclude_final_layer: bool or None
        - image_summaries: bool, whether to add image summaries
        - loss_fn: function, has signature loss_fn(labels, preds, weights)
        - loss_type: str, one of ['loss_mse', 'loss_xent']
        - results_cols: list of str, columns matching the return values of self.evaluate_preds()
        '''
        self.sess = sess
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.save_ckpt_prefix = save_ckpt_prefix

        self.train_images = train_batch['images']
        self.train_labels = train_batch['labels']
        self.train_locs = train_batch['locs']

        self.train_weights = train_batch.get('weights', None)
        self.train_eval_weights = train_eval_batch.get('weights', None)
        self.val_weights = val_batch.get('weights', None)

        self.train_eval_labels = train_eval_batch['labels']
        self.val_labels = val_batch['labels']

        self.train_preds = train_preds
        self.train_eval_preds = train_eval_preds
        self.val_preds = val_preds

        # ====================
        #      OPTIMIZER
        # ====================
        self.loss_type = loss_type
        with tf.variable_scope('train'):  # use 'train' scope to distinguish from 'val' losses
            self.train_loss_total, self.train_loss_nonreg, _, train_loss_summaries = \
                loss_fn(self.train_labels, self.train_preds, self.train_weights)

        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name='lr_placeholder')
        optimizer = tf.train.AdamOptimizer(self.lr_ph)

        # Add updates for batch normalizaiton
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.train_loss_total)

        # ====================
        #      SUMMARIES
        # ====================
        # gather the summaries to run after every step
        step_summaries = [train_loss_summaries]
        if image_summaries:
            if ls_bands in ['rgb', 'ms']:
                train_rgb_imgs = tf.reverse(self.train_images[:, :, :, 0:3], axis=[3])
                img_sums = add_image_summaries(
                    images=train_rgb_imgs, labels=self.train_labels,
                    preds=self.train_preds, locs=self.train_locs, k=1)
                step_summaries.append(img_sums)
            elif nl_band is not None:
                # add the DMSP and VIIRS bands together (one of them is all-0 anyways)
                # to create a greyscale image
                train_nl_imgs = tf.reduce_sum(self.train_images, axis=3, keepdims=True)
                img_sums = add_image_summaries(
                    images=train_nl_imgs, labels=self.train_labels,
                    preds=self.train_preds, locs=self.train_locs, k=1)
                step_summaries.append(img_sums)

        # gather the summaries to run after every epoch
        first_layer_sums = train_model.get_first_layer_summaries(ls_bands, nl_band)
        with tf.name_scope('train/'):
            lr_sum = tf.summary.scalar('learning_rate', self.lr_ph)
        epoch_summaries = [first_layer_sums, lr_sum]

        self.step_summaries_op = tf.summary.merge(step_summaries)
        self.epoch_summaries_op = tf.summary.merge(epoch_summaries)
        self.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ====================
        #  EVALUATION METRICS
        # ====================
        self.train_eval_summaries = self.create_eval_summaries('train')
        self.val_eval_summaries = self.create_eval_summaries('val')

        # ====================
        #   INITIALIZE VARS
        # ====================
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())

        # Restore from checkpoint if it exists. Otherwise, initialize weights from saved numpy arrays
        self._init_vars(
            model=train_model,
            ckpt_dir=init_ckpt_dir,
            imagenet_weights_path=imagenet_weights_path,
            hs_weight_init=hs_weight_init,
            exclude_final_layer=exclude_final_layer)

        # set var_list=tf.trainable_variables() to save only trainable variables
        MAX_MODELS_TO_KEEP = 1
        self.saver = tf.train.Saver(var_list=None, max_to_keep=MAX_MODELS_TO_KEEP)

        # variables to update during training
        self.step = 0
        self.epoch = 0
        self.results = pd.DataFrame(columns=['epoch', 'split'] + results_cols)
        self.results.set_index(['epoch', 'split'], inplace=True)

    def train_epoch(self, print_every: int = 1):
        '''Run 1 epoch of training.
        Note: assumes train dataset iterator doesn't need initialization,
            or is already initialized.
        Args
        - print_every: int, prints batch loss every this many steps
        '''
        preds_all = []
        labels_all = []
        feed_dict = {self.lr_ph: self.learning_rate * (self.lr_decay ** self.epoch)}
        step_str = 'Step {:05d}. Epoch {:02d}. {}: {:0.4f}, loss_tot: {:0.4f}, time: {:0.3f}s'

        if self.train_weights is None:
            weights_all = None
            required_ops = (self.train_preds, self.train_labels, self.train_op)  # type: Tuple[tf.Tensor, ...]
        else:
            weights_all = []
            required_ops = (self.train_preds, self.train_labels, self.train_weights, self.train_op)

        try:
            while True:
                curr_epoch = int(self.step * 1.0 / self.steps_per_epoch)
                if curr_epoch > self.epoch:
                    self.epoch = curr_epoch
                    break

                if self.step % print_every == 0:
                    start_time = time.time()
                    loss_total, loss_nonreg, summary, result = self.sess.run([
                        self.train_loss_total, self.train_loss_nonreg, self.step_summaries_op,
                        required_ops], feed_dict=feed_dict)
                    duration = time.time() - start_time
                    print(step_str.format(self.step, self.epoch, self.loss_type, loss_nonreg, loss_total, duration))
                    self.summary_writer.add_summary(summary, self.step)
                else:
                    result = self.sess.run(required_ops, feed_dict=feed_dict)

                self.step += 1

                if self.train_weights is None:
                    preds, labels, _ = result
                    preds_all.append(preds)
                    labels_all.append(labels)
                else:
                    preds, labels, weights, _ = result
                    preds_all.append(preds)
                    labels_all.append(labels)
                    weights_all.append(weights)

        except tf.errors.OutOfRangeError:
            pass

        labels_all = np.concatenate(labels_all)
        preds_all = np.concatenate(preds_all)
        if self.train_weights is not None:
            weights_all = np.concatenate(weights_all)

        # evaluate the training predictions / labels
        self.evaluate_preds(labels=labels_all, preds=preds_all, weights=weights_all,
                            split='train')

        # force flush all summaries to disk
        summary = self.sess.run(self.epoch_summaries_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.epoch)
        self.summary_writer.flush()

    def _eval_split(self, labels, preds, split, eval_summaries, weights=None,
                    init_iter=None, feed_dict=None, max_nbatches=None):
        '''
        Args
        - labels: tf.Tensor, shape [batch_size] or [batch_size, label_dim]
        - preds: tf.Tensor, shape [batch_size] or [batch_size, num_classes]
        - split: str
        - eval_summaries: dict, keys are str
            - other str => tf.placeholder for summaries
            - 'summary_op' => tf.Summary, merge of evaluation summaries
        - weights: tf.Tensor, shape [batch_size], or None
        - init_iter: tf.Operation, dataset iterator initializer
            set to None if no iterator initialization is necessary
        - feed_dict: dict, used for populating placeholders needed
            to initialize the dataset iterator
        - max_nbatches: int, maximum number of batches of the training dataset to run
            set to None to run until reaching a tf.errors.OutOfRangeError
        '''
        print(f'Evaluating model on {split} set...')
        if init_iter is not None:
            self.sess.run(init_iter, feed_dict=feed_dict)
        start_time = time.time()
        tensors_dict_ops = {'preds': preds, 'labels': labels}
        if weights is not None:
            tensors_dict_ops['weights'] = weights
        all_tensors = run_batches(
            sess=self.sess,
            tensors_dict_ops=tensors_dict_ops,
            max_nbatches=max_nbatches,
            verbose=True)
        speed = len(all_tensors['preds']) / (time.time() - start_time)
        print(f'... Finished {split} set. Completed at {speed:.3f} images / s')
        self.results.loc[(self.epoch, split), :] = self.evaluate_preds(
            labels=all_tensors['labels'],
            preds=all_tensors['preds'],
            split=split,
            weights=all_tensors.get('weights', None),
            eval_summaries=eval_summaries)

    def eval_train(self, init_iter=None, feed_dict=None, max_nbatches=None):
        '''Run trained model on training dataset.
        Args: see self._eval_split()
        '''
        self._eval_split(
            labels=self.train_eval_labels,
            preds=self.train_eval_preds,
            split='train_eval',
            eval_summaries=self.train_eval_summaries,
            weights=self.train_eval_weights,
            init_iter=init_iter,
            feed_dict=feed_dict,
            max_nbatches=max_nbatches)

    def _init_vars(self, model, ckpt_dir, imagenet_weights_path=None, hs_weight_init=None,
                   exclude_final_layer=None):
        '''Initialize the variables in the current tf.Graph.
        Tries in order:
        1. Restore weights from a saved checkpoint
        2. Load pre-trained weights from ImageNet
        3. Initialize weights randomly using the default variable initializers
        Args
        - model: instance of a model
        - ckpt_dir: str, path to checkpoint(s) for this specific model
        - imagenet_weights_path: str, path to pre-trained ImageNet weights for this model
        - hs_weight_init: str, one of [None, 'random', 'same', 'samescaled']
        - exclude_final_layer: bool, or None if no checkpoint provided
        '''
        # 1. Restore weights from a saved checkpoint
        if ckpt_dir not in [None, '']:
            var_list = tf.trainable_variables()
            if exclude_final_layer:
                exclude_list = model.get_final_layer_weights()
                print('Excluding variables:', exclude_list)
                var_list = [var for var in var_list if var not in exclude_list]
            saver = tf.train.Saver(var_list=var_list)
            load_successful = load(self.sess, saver, ckpt_dir)
            if load_successful:
                return
            else:
                print('No checkpoint file found in', ckpt_dir)

        # 2. Load pre-trained ImageNet weights from numpy file
        if imagenet_weights_path not in [None, '']:
            print('Initializing variables from pre-trained ImageNet weights.')
            model.init_from_numpy(imagenet_weights_path, self.sess, hs_weight_init=hs_weight_init)
            return

        # 3. Initialize weights randomly using the default variable initializers (ie. do nothing here)
        # comment out the next line to actually use default variable initialization
        # raise Exception('Did not find checkpoint nor pre-trained ImageNet weights.')
        print('No pre-trained weights given. Using default variable initialization.')

    def save_ckpt(self):
        '''Saves the current model to a checkpoint, and returns the checkpoint path'''
        return self.saver.save(
            sess=self.sess,
            save_path=self.save_ckpt_prefix,
            global_step=self.epoch)

    def log_results(self, csv_path: str):
        '''
        Args
        - csv_path: str, path to save results log
        '''
        print('saving csv log to:', csv_path)
        self.results.to_csv(csv_path)

    def create_eval_summaries(self, scope):
        raise NotImplementedError

    def evaluate_preds(self, labels, preds, split, weights=None, eval_summaries=None):
        raise NotImplementedError

    def eval_val(self, init_iter=None, feed_dict=None, max_nbatches=None):
        raise NotImplementedError


class RegressionTrainer(BaseTrainer):
    def __init__(self, train_batch, train_eval_batch, val_batch,
                 train_model, train_eval_model, val_model,
                 train_preds, train_eval_preds, val_preds,
                 sess, steps_per_epoch, ls_bands, nl_band, learning_rate, lr_decay,
                 log_dir, save_ckpt_prefix, init_ckpt_dir, imagenet_weights_path,
                 hs_weight_init, exclude_final_layer, image_summaries=True):
        '''
        See BaseTrainer for args descriptions
        '''
        super(RegressionTrainer, self).__init__(
            train_batch, train_eval_batch, val_batch,
            train_model, train_eval_model, val_model,
            train_preds, train_eval_preds, val_preds,
            sess, steps_per_epoch, ls_bands, nl_band, learning_rate, lr_decay,
            log_dir, save_ckpt_prefix, init_ckpt_dir, imagenet_weights_path,
            hs_weight_init, exclude_final_layer, image_summaries,
            loss_fn=loss_utils.loss_mse,
            loss_type='loss_mse',
            results_cols=['r2', 'R2', 'mse', 'rank'])

    def eval_val(self, init_iter=None, feed_dict=None, max_nbatches=None):
        '''Run trained model on validation dataset. Saves model checkpoint if
        validation mse is lower than the best seen so far.
        Args
        - init_iter: tf.Operation, validation dataset iterator initializer
            set to None if no iterator initialization is necessary
        - feed_dict: dict, used for populating placeholders needed
            to initialize the dataset iterator
        - max_nbatches: int, maximum number of batches of the validation dataset to run
            set to None to run until reaching a tf.errors.OutOfRangeError
        '''
        self._eval_split(
            labels=self.val_labels,
            preds=self.val_preds,
            split='val',
            eval_summaries=self.val_eval_summaries,
            weights=self.val_weights,
            init_iter=init_iter,
            feed_dict=feed_dict,
            max_nbatches=max_nbatches)

        # if first run or new best val mse
        val_mse = self.results.loc[(self.epoch, 'val'), 'mse']
        val_mses = self.results.loc[(slice(None), 'val'), 'mse']
        if (len(val_mses) == 1) or (val_mse == val_mses.min()):
            saved_ckpt_path = self.save_ckpt()
            print('New best MSE on val! Saved checkpoint to', saved_ckpt_path)

    def evaluate_preds(self, labels, preds, split, weights=None, eval_summaries=None):
        '''Helper method to calculate r^2, R^2, mse, and rank.
        Args
        - labels: np.array, shape [N] or [N, label_dim]
            - if shape [N, label_dim], only the 0-th column is used
        - preds: np.array, same shape as labels
        - split: str
        - weights: np.array of shape [N], or None
        - eval_summaries: dict, keys are str
            'r2_placeholder' => tf.placeholder
            'R2_placeholder' => tf.placeholder
            'mse_placeholder' => tf.placeholder
            'summary_op' => tf.Summary, merge of r2, R2, and mse summaries
        Returns: r2, R2, mse, rank
        '''
        assert labels.shape == preds.shape
        labels_eval = labels
        preds_eval = preds
        if len(labels.shape) > 1:
            labels_eval = labels[:, 0]
            preds_eval = preds[:, 0]

        r2, R2, mse, rank = evaluate(labels_eval, preds_eval, weights=weights, do_print=False)

        num_examples = len(labels_eval)
        s = 'Epoch {:02d}, {} ({} examples) r^2: {:0.3f}, R^2: {:0.3f}, mse: {:0.3f}, rank: {:0.3f}'
        print(s.format(self.epoch, split, num_examples, r2, R2, mse, rank))

        if eval_summaries is not None:
            summary_str = self.sess.run(eval_summaries['summary_op'], feed_dict={
                eval_summaries['r2_placeholder']: r2,
                eval_summaries['R2_placeholder']: R2,
                eval_summaries['mse_placeholder']: mse
            })
            self.summary_writer.add_summary(summary_str, self.epoch)
        return r2, R2, mse, rank

    def create_eval_summaries(self, scope: str) -> Mapping:
        '''
        Args
        - scope: str
        Returns metrics: dict, keys are str
            'r2_placeholder' => tf.placeholder
            'R2_placeholder' => tf.placeholder
            'mse_placeholder' => tf.placeholder
            'summary_op' => tf.Summary, merge of r2, R2, and mse summaries
        '''
        metrics = {}
        # not sure why, but we need the '/' in order to reuse the same 'train/' name for the scope
        with tf.name_scope(scope + '/'):
            metrics['r2_placeholder'] = tf.placeholder(tf.float32, shape=[], name='r2_placeholder')
            metrics['R2_placeholder'] = tf.placeholder(tf.float32, shape=[], name='R2_placeholder')
            metrics['mse_placeholder'] = tf.placeholder(tf.float32, shape=[], name='mse_placeholder')
            metrics['summary_op'] = tf.summary.merge([
                tf.summary.scalar('r2', metrics['r2_placeholder']),
                tf.summary.scalar('R2', metrics['R2_placeholder']),
                tf.summary.scalar('mse', metrics['mse_placeholder'])
            ])
        return metrics