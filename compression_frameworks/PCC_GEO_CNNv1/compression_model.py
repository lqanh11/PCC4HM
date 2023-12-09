import torch
import torch.nn as nn
import torch.nn.functional as F
from entropy_model import EntropyBottleneck

# def analysis_transform(tensor, num_filters, data_format): #data_format='channels_first' (batch, channels, depth, height, width) tensor.shape: (?, 1, 256, 256, 256)
#     with tf.variable_scope("analysis"):
#         with tf.variable_scope("layer_0"):
#             layer = tf.layers.Conv3D(
#                     num_filters, (9, 9, 9), strides=(2, 2, 2), padding="same",
#                     use_bias=True, activation=tf.nn.relu, data_format=data_format)
#             tensor = layer(tensor) #output tensor:(?, 32, 128, 128, 128)

#         with tf.variable_scope("layer_1"):
#             layer = tf.layers.Conv3D(
#                     num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
#                     use_bias=True, activation=tf.nn.relu, data_format=data_format)
#             tensor = layer(tensor) #output tensor:(?, 32, 64, 64, 64)
            
#         with tf.variable_scope("layer_2"):
#             layer = tf.layers.Conv3D(
#                     num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
#                     use_bias=False, activation=None, data_format=data_format)
#             tensor = layer(tensor) #output tensor:(?, 32, 32, 32, 32)

#     return tensor

class analysis_transform(nn.Module): #conv3d: input(N,Cin,D,H,W) output(N,Cout,Dout,Hout,Wout)
    def __init__(self,num_filters):
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels=1, out_channels=num_filters,kernel_size=9, stride=2, padding=4, bias=True)
        self.conv_2 = nn.Conv3d(in_channels=num_filters, out_channels=num_filters,kernel_size=5, stride=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(in_channels=num_filters, out_channels=num_filters,kernel_size=5, stride=2, padding=2, bias=False)

    def forward(self, x):
        output = self.conv_1(x)
        output = F.relu(output)
        output = self.conv_2(output)
        output = F.relu(output)
        output = self.conv_3(output)
        return output

# def synthesis_transform(tensor, num_filters, data_format): #tensor(?, 32, 32, 32, 32) num_filters：32
#     with tf.variable_scope("synthesis"):
#         with tf.variable_scope("layer_0"):
#             layer = tf.layers.Conv3DTranspose(
#                     num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
#                     use_bias=True, activation=tf.nn.relu, data_format=data_format)
#             tensor = layer(tensor) #output tensor:(?, 32, 64, 64, 64)

#         with tf.variable_scope("layer_1"):
#             layer = tf.layers.Conv3DTranspose(
#                     num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
#                     use_bias=True, activation=tf.nn.relu, data_format=data_format)
#             tensor = layer(tensor) #output tensor:(?, 32, 128, 128, 128)

#         with tf.variable_scope("layer_2"):
#             layer = tf.layers.Conv3DTranspose(
#                     1, (9, 9, 9), strides=(2, 2, 2), padding="same",
#                     use_bias=True, activation=tf.nn.relu, data_format=data_format)
#             tensor = layer(tensor) #output tensor:(?, 1, 256, 256, 256)

#     return tensor

class synthesis_transform(nn.Module):
    def __init__(self,num_filters):
        super().__init__()
        self.up_1 = nn.ConvTranspose3d(in_channels=32, out_channels=num_filters, kernel_size=5, stride=2, padding=2,output_padding=1, bias=True)
        self.up_2 = nn.ConvTranspose3d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=2, padding=2,output_padding=1, bias=True)
        self.up_3 = nn.ConvTranspose3d(in_channels=num_filters, out_channels=1, kernel_size=9, stride=2, padding=4,output_padding=1, bias=True)
    def forward(self, x):
        output = self.up_1(x)
        output = F.relu(output)
        output = self.up_2(output)
        output = F.relu(output)
        output = self.up_3(output)
        output = F.relu(output)
        return output

class PCCModel(torch.nn.Module):
    def __init__(self,num_filters):
        super().__init__()
        self.analysis_transform = analysis_transform(num_filters)
        self.synthesis_transform = synthesis_transform(num_filters)
        self.entropy_bottleneck = EntropyBottleneck(channels=num_filters)

    def forward(self, x, training=True):
        y = self.analysis_transform(x)
        y_tilde, likelihoods,quantiles_loss = self.entropy_bottleneck(y, quantize_mode="noise" if training else "symbols")
        x_tilde = self.synthesis_transform(y_tilde)
        
        return {'likelihoods':likelihoods,
                'x_tilde':x_tilde,
                'quantiles_loss':quantiles_loss}

if __name__ == '__main__':
    model = PCCModel(num_filters=32).to('cuda')
    input = torch.ones((1, 1, 128,128,128), device='cuda')
    # print(input)
    out = model(input)
    print(torch.max(out['x_tilde']))
    print(out['x_tilde'].shape)
    
    print(out['likelihoods'].shape)