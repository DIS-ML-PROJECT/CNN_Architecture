from models.base_model import BaseModel
from models import hyperspectral_resnet

class Hyperspectral_Resnet(BaseModel):
    def __init__(self, inputs, num_outputs, is_training,
                 fc_reg=0.0003, conv_reg=0.0003,
                 use_dilated_conv_in_first_layer=False,
                 num_layers=50, blocks_to_save=None):
        '''
        Args
        - inputs: tf.Tensor, shape [batch_size, H, W, C], type float32
        - num_outputs: int, number of output classes
            set to None if we are only extracting features
        - is_training: bool, or tf.placeholder of type tf.bool
        - fc_reg: float, regularization for weights in fully-connected layer
        - conv_reg: float, regularization for weights in conv layers
        - use_dilated_conv_in_first_layer: bool
        - num_layers: int, one of [18, 34, 50]
        - blocks_to_save: list of int, the blocks in the resnet from which to save features
            set to None to not save the outputs of earlier blocks in the resnet
            NOTE: this is a list of BLOCK numbers, not LAYER numbers
        '''
        super(Hyperspectral_Resnet, self).__init__(
            inputs=inputs,
            num_outputs=num_outputs,
            is_training=is_training,
            fc_reg=fc_reg,
            conv_reg=conv_reg)

        # determine bottleneck or not
        if num_layers in [18, 34]:
            self.bottleneck = False
        elif num_layers in [50]:
            self.bottleneck = True
        else:
            raise ValueError('Invalid num_layers passed to model')

        # set num_blocks
        if num_layers == 18:
            num_blocks = [2, 2, 2, 2]
        elif num_layers in [34, 50]:
            num_blocks = [3, 4, 6, 3]
        else:
            raise ValueError('Invalid num_layers passed to model')

        self.block_features = None
        if blocks_to_save is not None:
            self.block_features = {block_index: None for block_index in blocks_to_save}

        # outputs: tf.Tensor, shape [batch_size, num_outputs], type float32
        # features_layer: tf.Tensor, shape [batch_size, num_features], type float32
        self.outputs, self.features_layer = hyperspectral_resnet.inference(
            inputs,
            is_training=is_training,
            num_classes=num_outputs,
            num_blocks=num_blocks,
            bottleneck=self.bottleneck,
            use_dilated_conv_in_first_layer=use_dilated_conv_in_first_layer,
            blocks_to_save=self.block_features,
            conv_reg=conv_reg,
            fc_reg=fc_reg)