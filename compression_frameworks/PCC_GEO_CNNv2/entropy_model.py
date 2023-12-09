import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torchac

class RoundNoGradient(torch.autograd.Function):
    """ TODO: check. """
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x<1e-9] = 0
        except RuntimeError:
            print("ERROR! grad1[x<1e-9] = 0")
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy()<0.0)
        t = torch.Tensor(pass_through_if+0.0).to(grad1.device)

        return grad1*t


class EntropyBottleneck(nn.Module):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""
    
    def __init__(self, channels, init_scale=10, filters=(3,3,3), tail_mass=1e-9, likelihood_bound=1e-9): #channels：[N,C,D,H,W]中的C
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = likelihood_bound
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._tail_mass = float(tail_mass)
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # Create variables.
        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):
            #可训练参数初始化，并且将参数绑定到当前类（module）的parameter列表中
            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix) #源数据是数时，一般用fill_
            # self.matrix = torch.nn.functional.softplus(self.matrix)
            self._matrices.append(self.matrix)
            #
            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias) #源数据也是tensor时，一般用copy_
            self._biases.append(self.bias)
            #       
            if i < len(self._filters):
                self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factors.append(self.factor)
        
        target = np.log(2 / self._tail_mass - 1)
        self.target = torch.tensor([-target, 0, target], dtype=torch.float32, requires_grad=False) #常量，不算梯度
        self.quantiles = Parameter(torch.FloatTensor(channels, 1, 3))
        init_quantiles = torch.tensor([[[-self._init_scale, 0, self._init_scale]]], dtype=torch.float32)
        init_quantiles = init_quantiles.repeat(channels, 1,1)
        self.quantiles.data.copy_(init_quantiles)

    def _logits_cumulative(self, inputs, stop_gradient=False):
        """Evaluate logits of the cumulative densities.
        
        Arguments:
        inputs: The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch)`.

        Returns:
        A tensor of the same shape as inputs, containing the logits of the
        cumulatice densities evaluated at the the given inputs.
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            # matrix = self._matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(matrix, logits)
            bias = self._biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factors):
                factor = torch.tanh(self._factors[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * torch.tanh(logits)
        
        return logits

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            optimize_integer_offset = True
            if optimize_integer_offset:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                shape = inputs.size()
                inputs = inputs.view(shape[0], 1, -1)
                _medians = self.quantiles[:, :, 1:2]
                values = RoundNoGradient.apply(inputs - _medians) + _medians #(channels, 1, 1)
                values = values.view(shape)
                values = values.permute(1, 0, 2, 3, 4)
            return values

    def _likelihood(self, inputs):
        """Estimate the likelihood.
        inputs shape: [points, channels] (1, 8, 8, 8, 8)(N,C,D,H,W)
        """
        # reshape to (channels, 1, ...)
        inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()# [channels, N,D,H,W]
        shape = inputs.size()# [channels, N,D,H,W]
        inputs = inputs.view(shape[0], 1, -1)# [channels, 1, ...]
        inputs = inputs.to(self.matrix.device)
        # Evaluate densities.
        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach() #.detach():返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        # reshape to (points, channels)
        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0, 2, 3, 4)

        return likelihood

    def forward(self, inputs, quantize_mode="noise"):
        """Pass a tensor through the bottleneck.
        """
        if quantize_mode is None: outputs = inputs
        else: outputs = self._quantize(inputs, mode=quantize_mode)
        likelihood = self._likelihood(outputs)
        likelihood = Low_bound.apply(likelihood)
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True) #只训练self.quantiles，不算self._matrices等参数梯度

        quantiles_loss = torch.sum(abs(logits - self.target.to(self.matrix.device)))
        return outputs, likelihood, quantiles_loss

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1) #每一列都是前面列的累加和
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0
    
    def get_cdf(self):
        quantiles = self.quantiles #(channels, 1, 3)
        _medians = quantiles[:, :, 1:2] #(channels, 1, 1)
        optimize_integer_offset = True
        if not optimize_integer_offset:
            _medians = _medians.round()
        # Largest distance observed between lower tail quantile and median,
        # or between median and upper tail quantile.
        minima = (_medians - quantiles[:, :, 0:1]).max()
        maxima = (quantiles[:, :, 2:3] - _medians).max()
        minmax = torch.maximum(minima, maxima)
        minmax = minmax.ceil()
        minmax = torch.maximum(minmax, torch.tensor([1.0]).to(self.matrix.device))
        minmax = minmax.item()
        # Sample the density up to `minmax` around the median.
        samples = torch.arange(-minmax, minmax + 1).float().to(self.matrix.device) #N
        samples = samples + _medians #output samples:[channels, 1, N]
        lower = self._logits_cumulative(samples - 0.5, stop_gradient=True)
        upper = self._logits_cumulative(samples + 0.5, stop_gradient=True)
        sign = -torch.sign(torch.add(lower, upper))
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        pmf = torch.cat((
            torch.add(pmf[:, 0, :1], torch.sigmoid(lower[:, 0, :1])),
            pmf[:, 0, 1:-1],
            torch.add(pmf[:, 0, -1:], torch.sigmoid(-upper[:, 0, -1:])) ), -1)
        pmf = torch.clamp(pmf, min=self._likelihood_bound) #[channels, N]
        ndim = 5
        channel_axis = 1
        slices = ndim * [None] + [slice(None)]
        slices[channel_axis] = slice(None)
        slices = tuple(slices)
        # get cdf
        cdf = self._pmf_to_cdf(pmf) #cdf：累计概率，从0到1 cdf.shape: (C,N+1) 增加了一个0
        # print("cdf:",cdf)
        return cdf,slices,_medians

    @torch.no_grad() #以下数据不需要计算梯度，也不会进行反向传播 只在推理时调用，train时调用的是forward
    def compress(self, inputs):#inputs:[N,C,D,H,W]
        cdf,slices,_medians = self.get_cdf()
        _quantized_cdf = cdf
        ori_cdf = cdf.clone()
        cdf = _quantized_cdf[slices[1:]] #cdf.shape[C, 1, 1, 1, N+1]
        num_levels = cdf.shape[-1] - 1

        medians = torch.squeeze(_medians) #output medians.shape: (32,)
        offsets = (num_levels // 2) + 0.5 - medians #(32,)
        values = inputs + offsets[slices[:-1]]
        values = torch.maximum(values, torch.tensor([0.5]).to(self.matrix.device))
        values = torch.minimum(values, torch.tensor([num_levels - 0.5]).to(self.matrix.device))
        values = values.to(torch.int16) #values.shape: torch.Size([1, 32, 32, 32, 32])

        values = values.permute(0, 2, 3, 4, 1) #[N,C,D,H,W]->[N,D,H,W,channels] channels=32
        values = values.reshape(-1,values.shape[-1]) #[BatchSizexHxWxD, C] values.shape: torch.Size([32768, 32])
        out_cdf = ori_cdf.unsqueeze(0).repeat(values.shape[0], 1, 1).detach().cpu() #扩展了维度out_cdf.shape: torch.Size([13849, 8, 37])，而且在13849的维度上是一样的 out_cdf.shape: torch.Size([32768, 32, 86])
        strings = torchac.encode_float_cdf(out_cdf, values.cpu(), check_input_bounds=True) #根据累积分布函数进行算术编解码

        return strings

    @torch.no_grad()
    def decompress(self, strings, shape): #shape:(N,C,D,H,W)
        cdf,slices,_medians = self.get_cdf()
        _quantized_cdf = cdf
        ori_cdf = cdf.clone()
        cdf = _quantized_cdf[slices[1:]] #cdf.shape[C, 1, 1, 1, N+1]
        num_levels = cdf.shape[-1] - 1

        channels = int(shape[1])
        out_cdf = ori_cdf.unsqueeze(0).repeat(torch.prod(torch.tensor(shape)).item()//channels, 1, 1).cpu() #out_cdf.shape:[BatchSizexHxWxD, C, N+1]，跟上面compress一样 out_cdf.shape: torch.Size([32768, 32, 84])
        values = torchac.decode_float_cdf(out_cdf, strings) #values.shape: torch.Size([32768, 32])
        outputs = values.float()
        outputs = torch.reshape(outputs, (shape[0],shape[2],shape[3],shape[4],-1)) #[N,D,H,W,channels] outputs.shape: torch.Size([1, 32, 32, 32, 32])
        outputs = outputs.permute(0, 4, 1, 2, 3) #[N,D,H,W,channels]->[N,C,D,H,W]
        medians = torch.squeeze(_medians) #output medians.shape: (32,)
        offsets = (num_levels // 2) - medians.cpu() #offsets.shape: torch.Size([32])
        outputs = outputs - offsets[slices[:-1]] #offsets[slices[:-1]].shape: torch.Size([1, 32, 1, 1, 1])
        print("outputs.shape:",outputs.shape) #outputs.shape: torch.Size([1, 32, 32, 32, 32])

        return outputs #在cpu

#验过，没问题
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(108)
    training = False
    y = np.random.rand(2, 8, 8, 8, 8).astype("float32") #0-1均匀分布
    y = np.round(y * 20 - 10)
    y_gpu = torch.from_numpy(y).to(device)
    print("y_gpu[0,0,0,0]:",y_gpu[0,0,0,0])
    entropy_bottleneck = EntropyBottleneck(channels=8)
    y_strings, y_min_v, y_max_v = entropy_bottleneck.compress(y_gpu) #encode
    print("y_min_v:",y_min_v)
    print("y_max_v:",y_max_v)

    #decode
    y_decoded = entropy_bottleneck.decompress(y_strings, y_min_v.item(), y_max_v.item(), y_gpu.shape)
    compare = torch.eq(torch.from_numpy(y).int(),y_decoded.int())
    compare = compare.float()
    print("compare=False:",torch.nonzero(compare<0.1),len(torch.nonzero(compare<0.1))) #len(torch.nonzero(compare<0.1))=0
    print("y_decoded[0,0,0,0]:",y_decoded[0,0,0,0])
