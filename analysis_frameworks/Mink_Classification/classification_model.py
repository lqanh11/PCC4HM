
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.data

import MinkowskiEngine as ME
from entropy_model import EntropyBottleneck


class MinkowskiFCNN(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        ME.MinkowskiNetwork.__init__(self, D)

        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = self.get_conv_block(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv4 = self.get_conv_block(
            channels[3],
            channels[4],
            kernel_size=kernel_size,
            stride=2,
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
            ),
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

        # No, Dropout, last 256 linear, AVG_POOLING 92%

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)

        x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        return self.final(ME.cat(x1, x2)).F


class GlobalMaxAvgPool(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, tensor):
        x = self.global_max_pool(tensor)
        y = self.global_avg_pool(tensor)
        return ME.cat(x, y)


class MinkowskiSplatFCNN(MinkowskiFCNN):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        MinkowskiFCNN.__init__(
            self, in_channel, out_channel, embedding_channel, channels, D
        )

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.splat()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.interpolate(x)
        x2 = y2.interpolate(x)
        x3 = y3.interpolate(x)
        x4 = y4.interpolate(x)

        x = ME.cat(x1, x2, x3, x4)
        y = self.conv5(x.sparse())

        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        return self.final(ME.cat(x1, x2)).F
    

class MinkoPointNet_Conv(MinkowskiFCNN):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(3, 32, 64, 96, 128),
        D=3,
    ):
        MinkowskiFCNN.__init__(
            self, in_channel, out_channel, embedding_channel, channels, D
        )

    def forward(self, x: ME.TensorField):
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.interpolate(x)
        x2 = y2.interpolate(x)
        x3 = y3.interpolate(x)
        x4 = y4.interpolate(x)

        x = ME.cat(x1, x2, x3, x4)
        y = self.conv5(x.sparse())

        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        return self.final(ME.cat(x1, x2)).F

class MinkoPointNet_Conv_2(ME.MinkowskiNetwork):
    def __init__(
            self, 
            in_channel = 3,
            out_channel = 40,
            embedding_channel=1024,
            channels=(64, 64, 64, 128),
            D =3
        ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()
    
    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )
    
    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.conv1 = self.get_conv_block(
            in_channel=in_channel,
            out_channel=channels[0],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            in_channel=channels[0],
            out_channel=channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv3 = self.get_conv_block(
            in_channel=channels[1],
            out_channel=channels[2],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv4 = self.get_conv_block(
            in_channel=channels[2],
            out_channel=channels[3],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv5 = self.get_conv_block(
            in_channel=channels[3],
            out_channel=embedding_channel,
            kernel_size=kernel_size,
            stride=1,
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel*2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, input: ME.TensorField):
        x = input.sparse()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x1 = self.global_max_pool(x)
        x2 = self.global_avg_pool(x)
        return self.final(ME.cat(x1, x2)).F

class MinkoPointNet_Conv_1_chanel(ME.MinkowskiNetwork):
    def __init__(
            self, 
            in_channel = 1,
            out_channel = 40,
            embedding_channel=1024,
            channels=(64, 64, 64, 128),
            D =3
        ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()
    
    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )
    
    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.conv1 = self.get_conv_block(
            in_channel=in_channel,
            out_channel=channels[0],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            in_channel=channels[0],
            out_channel=channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv3 = self.get_conv_block(
            in_channel=channels[1],
            out_channel=channels[2],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv4 = self.get_conv_block(
            in_channel=channels[2],
            out_channel=channels[3],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv5 = self.get_conv_block(
            in_channel=channels[3],
            out_channel=embedding_channel,
            kernel_size=kernel_size,
            stride=1,
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel*2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, input: ME.TensorField):
        x = input.sparse()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x1 = self.global_max_pool(x)
        x2 = self.global_avg_pool(x)
        return self.final(ME.cat(x1, x2)).F

class Gain(nn.Module):
    def __init__(self, shape=None, factor: float = 10.):
        super().__init__()
        self.factor = factor
        self.gain = nn.Parameter(torch.ones(shape))

    def forward(self, x: Tensor) -> Tensor:
        return self.factor * self.gain * x

class MinkoPointNet_Conv_PCC(ME.MinkowskiNetwork):
    def __init__(
            self, 
            in_channel = 3,
            out_channel = 40,
            embedding_channel=1024,
            channels=(64, 64, 64, 128),
            D =3
        ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.entropy_bottleneck = EntropyBottleneck(1024)
        # self.gain1 = Gain((1, 1024), 10.)
        # self.gain2 = Gain((1, 1024), 1/10.)

        self.weight_initialization()
    
    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )
    
    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.conv1 = self.get_conv_block(
            in_channel=in_channel,
            out_channel=channels[0],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            in_channel=channels[0],
            out_channel=channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv3 = self.get_conv_block(
            in_channel=channels[1],
            out_channel=channels[2],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv4 = self.get_conv_block(
            in_channel=channels[2],
            out_channel=channels[3],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv5 = self.get_conv_block(
            in_channel=channels[3],
            out_channel=embedding_channel,
            kernel_size=kernel_size,
            stride=1,
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel*2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood


    def forward(self, input: ME.TensorField, training=True):
        x = input.sparse()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)

        # Quantizer & Entropy Model
        x_q, likelihood = self.get_likelihood(x, 
            quantize_mode="noise" if training else "symbols")
        
        x1 = self.global_max_pool(x_q)
        x2 = self.global_avg_pool(x_q)
        out=  self.final(ME.cat(x1, x2)).F
    
        return {'logits':out,
                'prior':x_q, 
                'likelihood':likelihood, 
                }
