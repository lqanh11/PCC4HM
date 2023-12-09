import torch
import torch.nn as nn
#from torch.nn.parameter import Parameter
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


class GaussianConditional(nn.Module):
    """Symmetric conditional entropy model.
    Argument:
        likelihood_bound;
        range_coder_precision;
    """
    
    def __init__(self,scale_lower_bound=1e-5):
        super().__init__()
        self._likelihood_bound = 1e-9
        self.scale_lower_bound = scale_lower_bound

    def _standardized_cumulative(self, inputs):
        const = -(2 ** -0.5)
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(const * inputs)

    def _likelihood(self, inputs, scale):
        """ Estimate the likelihoods conditioned on assumed distribution.
        Arguments:
        inputs;(quantized values); scale;
        Return:
        likelihood.
        """
        values = inputs
        upper = self._standardized_cumulative((.5 - values) / scale)
        lower = self._standardized_cumulative((-.5 - values) / scale)
        likelihood = upper - lower
        return likelihood

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

    def forward(self, inputs, scale, quantize_mode="noise"):
        """Pass a tensor through the bottleneck.
        Arguments:
        input tensor, scale.

        Returns:
        output quantized tensor.
        likelihoods.
        """
        if quantize_mode is None: outputs = inputs
        else: outputs = self._quantize(inputs, mode=quantize_mode)
        scale = torch.clamp(scale, min=self.scale_lower_bound) #这个函数对梯度没影响
        likelihood = self._likelihood(outputs, scale)
        likelihood = Low_bound.apply(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf): # pmf:[-1, N]
        cdf = pmf.cumsum(dim=-1) #每一列都是前面列的累加和
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device) #[-1,1]
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    def _get_cdf(self, scale, min_v, max_v, datashape):
        """Get quantized cdf for compress/decompress.
        Arguments:
        inputs: integer tensor min_v, max_v. 
                float32 tensor loc, scale. [BatchSizexHxWxD*C]
        Return: 
        cdf with shape [-1, channels, symbols]
        """
        # shape of cdf shound be # [-1, N]
        a = torch.arange(min_v, max_v+1)
        a = a.reshape(1,-1)
        #channels = datashape[1]
        a = a.repeat(torch.prod(torch.tensor(datashape)).int(), 1) #复制很多份
        a = a.float().to(scale.device) # [-1, N]
        
        scale = scale.unsqueeze(-1) #[-1,1]
        likelihood = self._likelihood(a, scale) #与a同维度
        pmf = torch.clamp(likelihood, min=self._likelihood_bound)
        cdf = self._pmf_to_cdf(pmf) #[-1,N+1]
        return cdf

    @torch.no_grad() #以下数据不需要计算梯度，也不会进行反向传播 只在推理时调用，train时调用的是forward
    def compress(self, inputs, scale):
        """Compress inputs and store their binary representations into strings.
        Arguments:
        inputs: `Tensor` with values to be compressed. Must have shape 
        [batchsize,C,D,H,W]
        locs & scales: same shape like inputs.
        Returns:
        compressed: String `Tensor` vector containing the compressed
            representation of each batch element of `inputs`.
        """
        datashape = inputs.shape
        #channels = datashape[1]
        scale = torch.reshape(scale, (-1,))
        inputs = torch.reshape(inputs, (-1,)) #[BatchSizexHxWxD*C]

        # quantize
        values = self._quantize(inputs, mode="symbols") #y.F，四舍五入
        # get cdf
        min_v = values.min().detach().float() #-17
        max_v = values.max().detach().float() #18
        cdf = self._get_cdf(scale, min_v, max_v, datashape) #[BatchSizexHxWxD*C, N+1]
        #print("cdf[0]:",cdf[0])
        # range encode.
        values_norm = values - min_v
        values_norm = values_norm.to(torch.int16)
        strings = torchac.encode_float_cdf(cdf.cpu(), values_norm.cpu(), check_input_bounds=True)
        min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, scale, min_v, max_v, datashape):
        """Decompress values from their compressed string representations.
        Arguments:
        strings: A string `Tensor` vector containing the compressed data.
        shape: A `Tensor` vector of int32 type. Contains the shape of the tensor to be
            decompressed. [batch size, length, width, height, channels]
        loc & scale: parameters of distributions.
        min_v & max_v: minimum & maximum values.
        Return: outputs [BatchSize, H, W, D, C]
        """
        # reshape.
        #channels = datashape[1]
        scale = torch.reshape(scale, (-1,))

        # get cdf.
        cdf = self._get_cdf(scale, min_v, max_v, datashape) #[BatchSizexHxWxD*C, N+1]
        values = torchac.decode_float_cdf(cdf.cpu(), strings)
        values = values.float()
        values += min_v
        values = torch.reshape(values,datashape)

        return values #在cpu

#验过，没问题
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(108)
    training = False
    y = np.random.randn(2, 16, 16, 16, 16).astype("float32")*10 #标准正态分布
    conditional_entropy_model = SymmetricConditional()
    loc = np.random.randn(2, 16, 16, 16, 16).astype("float32")
    scale = np.random.rand(2, 16, 16, 16, 16).astype("float32") #服从“0~1”均匀分布
    scale = torch.from_numpy(scale).to(device)
    y = torch.from_numpy(y).to(device)
    loc = torch.from_numpy(loc).to(device)
    #y.shape(1, 16, 16, 16, 16)
    #loc.shape: (1, 16, 16, 16, 16)
    #scale.shape: (1, 16, 16, 16, 16)
    scale = torch.abs(scale)
    scale = torch.clamp(scale, min=1e-9)
    y_tilde, likelihoods = conditional_entropy_model(y, loc, scale, quantize_mode="noise" if training else "symbols")
    print("y_tilde.shape:",y_tilde.shape)
    print("likelihoods.shape:",likelihoods.shape)
    print("y_tilde[0,0,0,0]:",y_tilde[0,0,0,0])
    print("likelihoods[0,0,0,0]:",likelihoods[0,0,0,0])
    strings, min_v, max_v = conditional_entropy_model.compress(y,loc,scale) #encode
    #decode
    y_decoded = conditional_entropy_model.decompress(strings, loc, scale, min_v.item(), max_v.item(), y.shape)
    compare = torch.eq(y_tilde.cpu().int(),y_decoded.int())
    compare = compare.float()
    print("compare=False:",torch.nonzero(compare<0.1),len(torch.nonzero(compare<0.1))) #len(torch.nonzero(compare<0.1))=0
    print("y_decoded[0,0,0,0]:",y_decoded[0,0,0,0])
