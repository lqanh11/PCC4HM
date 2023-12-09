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
    
    def __init__(self, channels, init_scale=8, filters=(3,3,3)):
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        #self._channels = channels
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # Create variables.
        #self._matrices = nn.ParameterList([]) #不支持多卡
        #self._biases = nn.ParameterList([])
        #self._factors = nn.ParameterList([])
        #self._matrices = nn.ModuleList([]) #用python的list[]会出不能同步到不同GPU的问题
        #self._biases = nn.ModuleList([]) #不支持append Parameter
        #self._factors = nn.ModuleList([])

        for i in range(len(self._filters) + 1):
            #可训练参数初始化，并且将参数绑定到当前类（module）的parameter列表中
            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix) #源数据是数时，一般用fill_
            #print(i," self.matrix:",self.matrix)
            #self._matrices.append(self.matrix)
            self.register_parameter('matrix_{}'.format(i), self.matrix)
            #
            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias) #源数据也是tensor时，一般用copy_
            #self._biases.append(self.bias)
            self.register_parameter('bias_{}'.format(i), self.bias)
            #       
            self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            self.factor.data.fill_(0.0)
            #self._factors.append(self.factor)
            self.register_parameter('factor_{}'.format(i), self.factor)

    def _logits_cumulative(self, inputs):
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
            #matrix = torch.nn.functional.softplus(self._matrices[i])
            matrix = torch.nn.functional.softplus(eval('self.matrix_{}'.format(i))) #通过字符串访问变量
            #print(i," self._matrices[i]:",self._matrices[i])
            logits = torch.matmul(matrix, logits)
            #logits += self._biases[i]
            logits += eval('self.bias_{}'.format(i))
            #factor = torch.tanh(self._factors[i])
            factor = torch.tanh(eval('self.factor_{}'.format(i)))
            logits += factor * torch.tanh(logits)
        
        return logits

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

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

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1) #每一列都是前面列的累加和，shape不变
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device) #(8,1)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1) #(8,N+1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    @torch.no_grad() #以下数据不需要计算梯度，也不会进行反向传播 只在推理时调用，train时调用的是forward
    def compress(self, inputs):#inputs:[N,C,D,H,W]
        inputs = inputs.permute(0, 2, 3, 4, 1) #[N,C,D,H,W]->[N,D,H,W,channels]
        # quantize
        values = self._quantize(inputs, mode="symbols") #四舍五入，[2, 8, 8, 8, 8]
        # get symbols
        min_v = values.min().detach().float() #-17
        max_v = values.max().detach().float() #18
        symbols = torch.arange(min_v, max_v+1)
        symbols = symbols.reshape(1,1,-1).repeat(values.shape[-1],1,1).float() #(channels=8,1,num_symbols)
        symbols = symbols.to(self.matrix.device)

        # get normalized values
        values_norm = values - min_v
        min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
        values_norm = values_norm.to(torch.int16)

        # get pmf
        lower = self._logits_cumulative(symbols - 0.5)
        upper = self._logits_cumulative(symbols + 0.5)
        sign = -torch.sign(torch.add(lower, upper))
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        pmf = torch.clamp(likelihood, min=self._likelihood_bound) #(channels=8,1,num_symbols)
        pmf = pmf.reshape(values.shape[-1],-1) #(8,N)

        # get cdf
        cdf = self._pmf_to_cdf(pmf) #cdf：累计概率，从0到1 cdf.shape: (8,N+1) 增加了一个0
        print("cdf:",cdf)
        '''
        cdf示例：后面非严格递增，都为1：[0.0000e+00, 1.0000e-09, 2.0000e-09, 3.0000e-09, 4.0000e-09, 6.1672e-09,
         1.7219e-08, 7.3574e-08, 3.6079e-07, 1.8232e-06, 9.2533e-06, 4.6831e-05,
         2.3501e-04, 1.1572e-03, 5.4723e-03, 2.3821e-02, 9.0058e-02, 2.9703e-01,
         7.0280e-01, 9.0978e-01, 9.7609e-01, 9.9450e-01, 9.9884e-01, 9.9976e-01,
         9.9995e-01, 9.9999e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00]'''
        # arithmetic encoding 到这里
        values_norm = values_norm.reshape(-1,values.shape[-1]) #[BatchSizexHxWxD, C]
        out_cdf = cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu() #扩展了维度out_cdf.shape: torch.Size([13849, 8, 37])，而且在13849的维度上是一样的
        strings = torchac.encode_float_cdf(out_cdf, values_norm.cpu(), check_input_bounds=True) #根据累积分布函数进行算术编解码

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape): #shape:(N,C,D,H,W)
        # get symbols
        symbols = torch.arange(min_v, max_v+1)
        channels = int(shape[1])
        symbols = symbols.reshape(1,1,-1).repeat(channels,1,1).float() #(channels=8,1,num_symbols)
        symbols = symbols.to(self.matrix.device)

        # get pmf
        lower = self._logits_cumulative(symbols - 0.5)
        upper = self._logits_cumulative(symbols + 0.5)
        sign = -torch.sign(torch.add(lower, upper))
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        pmf = torch.clamp(likelihood, min=self._likelihood_bound) #(channels=8,1,num_symbols)
        pmf = pmf.reshape(channels,-1) #(8,N)
        # get cdf
        cdf = self._pmf_to_cdf(pmf)
        # arithmetic decoding
        out_cdf = cdf.unsqueeze(0).repeat(torch.prod(torch.tensor(shape)).item()//channels, 1, 1).cpu() #out_cdf.shape:[BatchSizexHxWxD, C, N+1]，跟上面compress一样
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += min_v #values.shape: torch.Size([BatchSizexHxWxD, C])这里的value与compress中self._quantize得到的value值一样
        values = torch.reshape(values, (shape[0],shape[2],shape[3],shape[4],-1)) #[N,D,H,W,channels]
        values = values.permute(0, 4, 1, 2, 3) #[N,D,H,W,channels]->[N,C,D,H,W]
        return values #在cpu

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
