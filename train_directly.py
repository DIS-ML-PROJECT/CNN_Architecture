#from models.vggf_model import VGGF
#from models.simple_cnn import SimpleCNN
from models.resnet_model import Hyperspectral_Resnet
#from models.resnet_combo import ResnetCombo
from batchers import dataset_constants, batcher
from utils.run import get_full_experiment_name, make_log_and_ckpt_dirs
from utils.trainer import RegressionTrainer

import os
import pickle
from pprint import pprint
import time

import numpy as np
import tensorflow as tf

# edit if necessary
ROOT_DIR = './'


def run_training(sess, ooc, batcher_type, dataset, keep_frac, model_name, model_params, batch_size,
                 ls_bands, nl_band, label_name, augment, learning_rate, lr_decay,
                 max_epochs, print_every, eval_every, num_threads, cache, log_dir, save_ckpt_dir,
                 init_ckpt_dir, imagenet_weights_path, hs_weight_init, exclude_final_layer):
    '''
    Args
    - sess: tf.Session
        A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.
    - ooc: bool, whether to use out-of-country split
    - batcher_type: str, type of batcher, one of ['base', 'urban', 'rural']
    - dataset: str, options depends on batcher_type
    - keep_frac: float
        amount of dataset to keep and randomly select
    - model_name: str, one of ['resnet', 'vggf', 'simplecnn', 'resnetcombo']
        in this case hard coded to resnet
    - model_params: dict
        model parameters (includes e.g. batch size, dataset, ckpt_dir, .... )
        these are saved to params.txt in the new_experiment directory
    - batch_size: int
    - ls_bands: one of [None, 'rgb', 'ms']
        which bands are supposed to be used
    - nl_band: one of [None, 'merge', 'split']
        are nl band supposed to be used, if yes are they supposed to be splitted?
    - label_name: str, name of the label in the TFRecord file
    - augment: bool
        whether or not to perform a augmentation on the images
    - learning_rate: float
    - lr_decay: float
    - max_epochs: int
    - print_every: int
    - eval_every: int
    - num_threads: int
    - cache: list of str
    - log_dir: str, path to directory to save logs for TensorBoard, must already exist
    - save_ckpt_dir: str, path to checkpoint dir for saving weights
        - intermediate dirs must already exist
    - init_ckpt_dir: str, path to checkpoint dir from which to load existing weights
        - set to empty string '' to use ImageNet or random initialization
    - imagenet_weights_path: str, path to pre-trained weights from ImageNet
        - set to empty string '' to use saved ckpt or random initialization
    - hs_weight_init: str, one of [None, 'random', 'same', 'samescaled']
    - exclude_final_layer: bool, or None
    '''

    # set model class
    # check if one of the supported models was chosen

    if model_name == 'resnet':
        model_class = Hyperspectral_Resnet
    #
    # # import model classes if needed
    # elif model_name == 'vggf':
    #     model_class = VGGF
    # elif model_name == 'simplecnn':
    #     model_class = SimpleCNN
    # elif model_name == 'resnetcombo':
    #     model_class = ResnetCombo
    else:
        raise ValueError('Unknown model_name. Was not one of ["resnet", "vggf", "simplecnn", "resnetcombo"]'
                         'or model was not imported.')
    # check if paths exist
    assert os.path.exists(log_dir)
    assert os.path.exists(os.path.dirname(save_ckpt_dir))

    # batcher definition
    # out-of-country split for dhs
    if ooc:

        # sustainlab: temporary hack: hard-coding '2009-17' base dataset for all DHS OOC
        base_dataset = '2009-17'

        # get tfrecord paths for train, val and all data
        train_tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(dataset, 'train'))
        val_tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(dataset, 'val'))
        all_tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(dataset, 'all'))

        # get dataset sizes
        sizes = {
            'base': dataset_constants.SIZES[dataset],
            'urban': dataset_constants.URBAN_SIZES[dataset],
            'rural': dataset_constants.RURAL_SIZES[dataset]
        }[batcher_type]

        # check size of tfrecord paths
        assert len(train_tfrecord_paths) == sizes['train']
        assert len(val_tfrecord_paths) == sizes['val']

    # in-country split
    else:
        if batcher_type != 'base':
            raise ValueError('incountry w/ non-base batcher is not supported')


        base_dataset = '2009-17'
        all_tfrecord_paths = np.asarray(batcher.get_tfrecord_paths('2009-17', 'all'))

        # get incountry folds
        with open(os.path.join(ROOT_DIR, 'data/dhs_incountry_folds.pkl'), 'rb') as f:
            incountry_folds = pickle.load(f)

        # check size of incountry folds
        assert len(all_tfrecord_paths) == dataset_constants.SIZES[dataset]['all']

        # define indices, tfrecord paths for train and val data
        fold = dataset[-1]
        train_indices = incountry_folds[fold]['train']
        val_indices = incountry_folds[fold]['val']

        train_tfrecord_paths = all_tfrecord_paths[train_indices]
        val_tfrecord_paths = all_tfrecord_paths[val_indices]

    # get size of train and val data
    num_train = len(train_tfrecord_paths)
    num_val = len(val_tfrecord_paths)

    # keep_frac affects sizes of both training and validation sets
    if keep_frac < 1.0:
        if batcher_type != 'base':
            raise ValueError('keep_frac < 1.0 w/ non-base batcher is not supported')

        # apply keep_frac on set sizes
        num_train = int(num_train * keep_frac)
        num_val = int(num_val * keep_frac)

        # get randomized tfrecord paths
        train_tfrecord_paths = np.random.choice(train_tfrecord_paths, size=num_train, replace=False)
        val_tfrecord_paths = np.random.choice(val_tfrecord_paths, size=num_val, replace=False)

    print('num_train:', num_train)
    print('num_val:', num_val)

    # calculate steps per epoch for train and val
    train_steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_val / batch_size))

    def get_batcher(tfrecord_paths, shuffle, augment, epochs, cache):

        # define BatcherClass
        BatcherClass = {
            'base': batcher.Batcher,
            'urban': batcher.UrbanBatcher,
            'rural': batcher.RuralBatcher,
        }[batcher_type]

        return BatcherClass(
            tfrecord_files=tfrecord_paths,
            dataset=base_dataset,
            batch_size=batch_size,
            label_name=label_name,
            num_threads=num_threads,
            epochs=epochs,
            ls_bands=ls_bands,
            nl_band=nl_band,
            shuffle=shuffle,
            augment=augment,
            negatives='zero',
            normalize=True,
            cache=cache
        )

    # set placeholders for tensors
    train_tfrecord_paths_ph = tf.placeholder(tf.string, shape=[None])
    val_tfrecord_paths_ph = tf.placeholder(tf.string, shape=[None])

    # get train batch
    with tf.name_scope('train_batcher'):
        train_batcher = get_batcher(
            train_tfrecord_paths_ph,
            shuffle=True,
            augment=augment,
            epochs=max_epochs,
            cache='train' in cache
        )
        train_init_iter, train_batch = train_batcher.get_batch()

    # get train, eval batch
    with tf.name_scope('train_eval_batcher'):
        train_eval_batcher = get_batcher(
            train_tfrecord_paths_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,  # sustainlab: may need extra epoch at the end of training
            cache='train_eval' in cache
        )
        train_eval_init_iter, train_eval_batch = train_eval_batcher.get_batch()

    # get val batch
    with tf.name_scope('val_batcher'):
        val_batcher = get_batcher(
            val_tfrecord_paths_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,  # sustainlab: may need extra epoch at the end of training
            cache='val' in cache
        )
        val_init_iter, val_batch = val_batcher.get_batch()

    # build model
    print('Building mode...', flush=True)
    model_params['num_outpus'] = 1

    # train model
    with tf.variable_scope(tf.get_variable_scope()) as model_scope:  # edited mm
        train_model = model_class(train_batch['images'], is_training=True, **model_params)
        train_preds = train_model.outputs
        if model_params['num_outputs'] == 1:
            train_preds = tf.reshape(train_preds, shape=[-1], name='train_preds')

    # train, eval model
    with tf.variable_scope(model_scope, reuse=True):  # edited mm
        train_eval_model = model_class(train_eval_batch['images'], is_training=False, **model_params)
        train_eval_preds = train_eval_model.outputs
        if model_params['num_outputs'] == 1:
            train_eval_preds = tf.reshape(train_eval_preds, shape=[-1], name='train_eval_preds')

    # val model
    with tf.variable_scope(model_scope, reuse=True):  # edited mm
        val_model = model_class(val_batch['images'], is_training=False, **model_params)
        val_preds = val_model.outputs
        if model_params['num_outputs'] == 1:
            val_preds = tf.reshape(val_preds, shape=[-1], name='val_preds')

    #class for training and evaluation of regression models
    trainer = RegressionTrainer(
        train_batch, train_eval_batch, val_batch,
        train_model, train_eval_model, val_model,
        train_preds, train_eval_preds, val_preds,
        sess, train_steps_per_epoch, ls_bands, nl_band, learning_rate, lr_decay,
        log_dir, save_ckpt_dir, init_ckpt_dir, imagenet_weights_path,
        hs_weight_init, exclude_final_layer, image_summaries=False)

    # sustainlab: initialize the training dataset iterator
    sess.run([train_init_iter, train_eval_init_iter, val_init_iter], feed_dict={
        train_tfrecord_paths_ph: train_tfrecord_paths,
        val_tfrecord_paths_ph: val_tfrecord_paths
    })

    # train epochs
    for epoch in range(max_epochs):
        if epoch % eval_every == 0:
            trainer.eval_train(max_nbatches=train_steps_per_epoch)
            trainer.eval_val(max_nbatches=val_steps_per_epoch)
        trainer.train_epoch(print_every)

    # eval train
    trainer.eval_train(max_nbatches=train_steps_per_epoch)
    trainer.eval_val(max_nbatches=val_steps_per_epoch)

    # save log results
    csv_log_path = os.path.join(log_dir, 'results.csv')
    trainer.log_results(csv_log_path)


def run_training_wrapper(**params):
    '''
    params is a dict with keys matching the FLAGS defined below
    '''

    # print starting time
    start = time.time()
    print('Current time:', start)

    # print all of the flags
    pprint(params)

    # parameters that might be 'None'
    none_params = ['ls_bands', 'nl_band', 'exclude_final_layer', 'hs_weight_init',
                   'imagenet_weights_path', 'init_ckpt_dir']
    for p in none_params:
        if params[p] == 'None':
            params[p] = None

    # reset any existing graph
    tf.reset_default_graph()  # edited mm

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.set_random_seed(seed)  # edited mm

    # create the log and checkpoint directories if needed
    # Returns a str
    # '{experiment_name}_b{batch_size}_fc{fc_str}_conv{conv_str}'
    # where fc_str and conv_str are the numbers past the decimal for the fc / conv regularization parameters.
    # Optionally appends a tag to the end.
    full_experiment_name = get_full_experiment_name(
        params['experiment_name'], params['batch_size'],
        params['fc_reg'], params['conv_reg'], params['lr']
    )
    # Creates 2 new directories:
    # 1. log_dir: {log_dir_base}/{full_experiment_name}
    # 2. ckpt_dir: {ckpt_dir_base}/{full_experiment_name}

    log_dir, ckpt_prefix = make_log_and_ckpt_dirs(
        params['log_dir'], params['ckpt_dir'], full_experiment_name
    )
    print(f'Checkpoint prefix: {ckpt_prefix}')
    params_filepath = os.path.join(log_dir, 'params.txt')

    # check for previous run
    assert not os.path.exists(params_filepath), f'Stopping. Found previous run at: {params_filepath}'
    with open(params_filepath, 'w') as f:
        pprint(params, stream=f)
        pprint(f'Checkpoint prefix: {ckpt_prefix}', stream=f)

    # create session
    # sustainlab: - MUST set os.environ['CUDA_VISIBLE_DEVICES'] before creating tf.Session object
    if params['gpu'] is None:  # sustainlab: restrict to CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])

    # configure session
    config = tf.ConfigProto()  # edited mm
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)  # edited mm

    # set model parameters
    model_params = {
        'fc_reg': params['fc_reg'],
        'conv_reg': params['conv_reg'],
        'use_dilated_conv_in_first_layer': False,
    }

    if params['model_name'] == 'resnet':
        model_params['num_layers'] = params['num_layers']

    run_training(
        sess=sess,
        ooc=params['ooc'],
        batcher_type=params['batcher_type'],
        dataset=params['dataset'],
        keep_frac=params['keep_frac'],
        model_name=params['model_name'],
        model_params=model_params,
        batch_size=params['batch_size'],
        ls_bands=params['ls_bands'],
        nl_band=params['nl_band'],
        label_name=params['label_name'],
        augment=params['augment'],
        learning_rate=params['lr'],
        lr_decay=params['lr_decay'],
        max_epochs=params['max_epochs'],
        print_every=params['print_every'],
        eval_every=params['eval_every'],
        num_threads=params['num_threads'],
        cache=params['cache'],
        log_dir=log_dir,
        save_ckpt_dir=ckpt_prefix,
        init_ckpt_dir=params['init_ckpt_dir'],
        imagenet_weights_path=params['imagenet_weights_path'],
        hs_weight_init=params['hs_weight_init'],
        exclude_final_layer=params['exclude_final_layer']
    )
    sess.close()

    # print ending time
    end = time.time()
    print('End time:', end)
    print('Time elapsed (sec.):', end - start)


def main(_):
    params = {
        key: flags.FLAGS.__getattr__(key)
        for key in dir(flags.FLAGS)
    }
    run_training_wrapper(**params)


if __name__ == '__main__':
    flags = tf.app.flags

    # paths
    flags.DEFINE_string('experiment_name', 'new_experiment', 'name of the experiment being run')
    flags.DEFINE_string('ckpt_dir', os.path.join(ROOT_DIR, 'ckpts/'), 'checkpoint directory')
    flags.DEFINE_string('log_dir', os.path.join(ROOT_DIR, 'logs/'), 'log directory')

    # initialization
    flags.DEFINE_string('init_ckpt_dir', None,
                        'path to checkpoint prefix from which to initialize weights (default None)')
    flags.DEFINE_string('imagenet_weights_path', None, 'path to ImageNet weights for initialization (default None)')
    flags.DEFINE_string('hs_weight_init', None,
                        'method for initializing weights of non-RGB bands in 1st conv layer, one of [None (default), "random", "same", "samescaled"]')
    flags.DEFINE_boolean('exclude_final_layer', None,
                         'whether to use checkpoint to initialize final layer (default None)')

    # learning parameters
    flags.DEFINE_string('label_name', 'wealthpooled', 'name of label to use from the TFRecord files')
    flags.DEFINE_integer('batch_size', 64, 'batch size')
    flags.DEFINE_boolean('augment', True, 'whether to use data augmentation')
    flags.DEFINE_float('fc_reg', 1e-3, 'Regularization penalty factor for fully connected layers')
    flags.DEFINE_float('conv_reg', 1e-3, 'Regularization penalty factor for convolution layers')
    flags.DEFINE_float('lr', 1e-3, 'Learning rate for optimizer')
    flags.DEFINE_float('lr_decay', 1.0, 'Decay rate of the learning rate (default 1.0 for no decay)')

    # high-level model control
    flags.DEFINE_string('model_name', 'resnet',
                        'name of the model to be used, one of ["resnet" (default), "vggf", "simplecnn", "resnetcombo"]')

    # resnet-only params
    flags.DEFINE_integer('num_layers', 18, 'Number of ResNet layers, one of [18 (default), 34, 50]')

    # data params
    flags.DEFINE_string('batcher_type', 'base', 'batcher, one of ["base" (default), "urban", "rural"]')
    flags.DEFINE_string('dataset', '2009-17', 'dataset to use, options depend on batcher_type (default "2009-17")')
    flags.DEFINE_boolean('ooc', True, 'whether to use out-of-country split (default True)')
    flags.DEFINE_float('keep_frac', 1.0, 'fraction of training data to use (default 1.0)')
    flags.DEFINE_string('ls_bands', None, 'Landsat bands to use, one of [None (default), "rgb", "ms"]')
    flags.DEFINE_string('nl_band', None, 'nightlights band, one of [None (default), "merge", "split"]')

    # system
    flags.DEFINE_integer('gpu', None, 'which GPU to use (default None)')
    flags.DEFINE_integer('num_threads', 1, 'number of threads for batcher (default 1)')
    flags.DEFINE_list('cache', [],
                      'comma-separated list (no spaces) of datasets to cache in memory, choose from [None, "train", "train_eval", "val"]')

    # Misc
    flags.DEFINE_integer('max_epochs', 150, 'maximum number of epochs for training (default 50)')
    flags.DEFINE_integer('eval_every', 1,
                         'evaluate the model on the validation set after every so many epochs of training')
    flags.DEFINE_integer('print_every', 10, 'print training statistics after every so many steps')
    flags.DEFINE_integer('seed', 123, 'seed for random initialization and shuffling')

    tf.run()  # edited mm
