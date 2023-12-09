from enum import Enum
# import tensorflow.compat.v1 as tf
# from tensorflow.keras.layers import Layer, Conv3D, Conv3DTranspose
# from tensorflow_core.python.keras.utils import conv_utils
import torch
import torch.nn as nn

# def get_channel_axis():
#     return 1

class SequentialLayer(nn.Module):
    def __init__(self, layers, *args, **kwargs):
        super(SequentialLayer, self).__init__(*args, **kwargs)
        self._layers = layers

    # def call(self, tensor, **kwargs):
    def forward(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor

class ResidualLayer(nn.Module):
    def __init__(self, layers, residual_mode='add'): #, *args, **kwargs
        super(ResidualLayer, self).__init__() #*args, **kwargs
        assert residual_mode in ('add', 'concat')
        self._layers = layers
        self.residual_mode = residual_mode

    # def call(self, tensor):
    def forward(self, tensor):
        tensor = self._layers[0](tensor)
        tensor = self._layers[1](tensor)
        tensor1 = tensor.clone()
        for layer in self._layers[2:]:
            tensor = layer(tensor)
        if self.residual_mode == 'add':
            return tensor1 + tensor
        else:
            return torch.cat((tensor,tensor1),dim=1)

class AnalysisBlock(ResidualLayer):
    def __init__(self, in_channels, filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation=nn.ReLU(), *args, **kwargs):
        # data_format = conv_utils.normalize_data_format(data_format)
        # params = {'padding': 'same', 'use_bias': True, 'activation': activation,
        #           'filters': filters, 'kernel_size': kernel_size}
        params = {'out_channels':filters,'kernel_size': kernel_size,'bias':True,'padding':1} #无论strides等于1还是2，padding都是1
        layers = nn.ModuleList([nn.Conv3d(in_channels=in_channels,stride=strides, **params),
                  activation,
                  nn.Conv3d(in_channels=filters,**params),
                  activation,
                #   Conv3D(**params),
                #   Conv3D(**params)
                  nn.Conv3d(in_channels=filters,**params),
                  activation])
        super(AnalysisBlock, self).__init__(layers, *args, **kwargs)


class SynthesisBlock(ResidualLayer):
    def __init__(self, in_channels, filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation=nn.ReLU(), *args, **kwargs):
        # data_format = conv_utils.normalize_data_format(data_format)
        # params = {'padding': 'same', 'data_format': data_format, 'use_bias': True, 'activation': activation,
        #           'filters': filters, 'kernel_size': kernel_size}
        params = {'out_channels':filters,'kernel_size': kernel_size,'bias':True,'padding':1} #无论strides等于1还是2，padding都是1
        layers = nn.ModuleList([#Conv3DTranspose(strides=strides, **params),
                  nn.ConvTranspose3d(in_channels=in_channels, stride=strides, output_padding=1, **params), #stride=2,output_padding=1;stride=1,output_padding=0
                  activation,
                #   Conv3DTranspose(**params),
                  nn.ConvTranspose3d(in_channels=filters, output_padding=0, **params),
                  activation,
                #   Conv3DTranspose(**params)
                  nn.ConvTranspose3d(in_channels=filters, output_padding=0, **params),
                  activation
                  ])
        super(SynthesisBlock, self).__init__(layers, *args, **kwargs)

# class AnalysisTransformV2(SequentialLayer):
#     def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, residual_mode='add',
#                  *args, **kwargs):
#         data_format = conv_utils.normalize_data_format(data_format)
#         params = {'kernel_size': kernel_size, 'activation': activation, 'data_format': data_format,
#                   'residual_mode': residual_mode}
#         layers = [AnalysisBlock(filters // 2, **params),
#                   AnalysisBlock(filters, **params),
#                   AnalysisBlock(filters, **params),
#                   Conv3D(filters, kernel_size, padding="same", use_bias=False, activation=None,
#                          data_format=data_format)]
#         super(AnalysisTransformV2, self).__init__(layers, *args, **kwargs)


# class SynthesisTransformV2(SequentialLayer):
#     def __init__(self, filters, data_format=None, kernel_size=(3, 3, 3), activation=tf.nn.relu, residual_mode='add',
#                  *args, **kwargs):
#         data_format = conv_utils.normalize_data_format(data_format)
#         params = {'kernel_size': kernel_size, 'activation': activation, 'data_format': data_format,
#                   'residual_mode': residual_mode}
#         layers = [SynthesisBlock(filters, **params),
#                   SynthesisBlock(filters, **params),
#                   SynthesisBlock(filters // 2, **params),
#                   Conv3DTranspose(1, kernel_size, padding="same", use_bias=True, activation=activation,
#                                   data_format=data_format)]
#         super(SynthesisTransformV2, self).__init__(layers, *args, **kwargs)


class AnalysisTransformProgressiveV2(SequentialLayer):
    def __init__(self, in_channels, filters, kernel_size=(3, 3, 3), activation=nn.ReLU(), residual_mode='add', *args, **kwargs):
        # data_format = conv_utils.normalize_data_format(data_format)
        params = {'kernel_size': kernel_size, 'activation': activation, 'residual_mode': residual_mode}
        layers = nn.ModuleList([AnalysisBlock(in_channels=in_channels, filters=filters // 4, **params),
                  AnalysisBlock(in_channels=filters // 4,filters=filters // 2, **params),
                  AnalysisBlock(in_channels=filters // 2,filters=filters, **params),
                #   Conv3D(filters, kernel_size, padding="same", use_bias=False, activation=None, data_format=data_format)
                  nn.Conv3d(in_channels=filters,out_channels=filters,kernel_size=kernel_size,bias=False, padding=1)
                  ])
        super(AnalysisTransformProgressiveV2, self).__init__(layers, *args, **kwargs)


class SynthesisTransformProgressiveV2(SequentialLayer):
    def __init__(self, in_channels, filters, kernel_size=(3, 3, 3), activation=nn.ReLU(), residual_mode='add', *args, **kwargs):
        # data_format = conv_utils.normalize_data_format(data_format)
        # params = {'kernel_size': kernel_size, 'activation': activation, 'data_format': data_format, 'residual_mode': residual_mode}
        params = {'kernel_size': kernel_size, 'activation': activation, 'residual_mode': residual_mode}
        layers = nn.ModuleList([SynthesisBlock(in_channels,filters, **params),
                  SynthesisBlock(filters,filters // 2, **params),
                  SynthesisBlock(filters // 2,filters // 4, **params),
                #   Conv3DTranspose(1, kernel_size, padding="same", use_bias=True, activation=activation, data_format=data_format)
                  nn.ConvTranspose3d(in_channels=filters // 4, out_channels=1, kernel_size=kernel_size, stride=1, padding=1, output_padding=0, bias=True),
                  activation
                  ])
        super(SynthesisTransformProgressiveV2, self).__init__(layers, *args, **kwargs)


class HyperAnalysisTransform(SequentialLayer):
    def __init__(self, in_channels, filters, kernel_size=(3, 3, 3), activation=nn.ReLU(), *args, **kwargs):
        # data_format = conv_utils.normalize_data_format(data_format)
        # params = {'padding': 'same', 'data_format': data_format, 'filters': filters, 'kernel_size': kernel_size}
        params = {'out_channels':filters,'kernel_size': kernel_size,'padding':1}
        layers = nn.ModuleList([#Conv3D(use_bias=True, activation=activation, **params),
                  nn.Conv3d(in_channels=in_channels, bias=True, **params),
                  activation,
                #   Conv3D(use_bias=True, activation=activation, strides=(2, 2, 2), **params),
                  nn.Conv3d(in_channels=filters, bias=True, stride=(2, 2, 2), **params),
                  activation,
                #   Conv3D(use_bias=False, activation=None, **params)
                  nn.Conv3d(in_channels=filters, bias=False, **params)
                  ])
        super(HyperAnalysisTransform, self).__init__(layers, *args, **kwargs)


class HyperSynthesisTransform(SequentialLayer):
    def __init__(self, in_channels,filters, kernel_size=(3, 3, 3), activation=nn.ReLU(), *args, **kwargs):
        # data_format = conv_utils.normalize_data_format(data_format)
        # params = {'padding': 'same', 'data_format': data_format, 'activation': activation, 'use_bias': True,'filters': filters, 'kernel_size': kernel_size}
        params = {'out_channels':filters,'kernel_size': kernel_size,'bias':True,'padding':1} #无论strides等于1还是2，padding都是1
        layers = nn.ModuleList([#Conv3DTranspose(**params),
                  nn.ConvTranspose3d(in_channels=in_channels, output_padding=0, **params),
                  activation,
                #   Conv3DTranspose(strides=(2, 2, 2), **params),
                  nn.ConvTranspose3d(in_channels=filters, stride=(2, 2, 2), output_padding=1, **params),
                  activation,
                #   Conv3DTranspose(**params)
                  nn.ConvTranspose3d(in_channels=filters, output_padding=0, **params),
                  activation
                  ])
        super(HyperSynthesisTransform, self).__init__(layers, *args, **kwargs)


# class TransformType(Enum):
#     # AnalysisTransformV1 = AnalysisTransformV1
#     # AnalysisTransformV2 = AnalysisTransformV2
#     AnalysisTransformProgressiveV2 = AnalysisTransformProgressiveV2
#     # SynthesisTransformV1 = SynthesisTransformV1
#     # SynthesisTransformV2 = SynthesisTransformV2
#     SynthesisTransformProgressiveV2 = SynthesisTransformProgressiveV2
#     HyperAnalysisTransform = HyperAnalysisTransform
#     HyperSynthesisTransform = HyperSynthesisTransform
