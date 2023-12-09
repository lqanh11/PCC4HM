import torch
import models.model_voxception as model
from models.entropy_model import EntropyBottleneck
from models.conditional_entropy_model import SymmetricConditional

class PCCModel(torch.nn.Module):
    def __init__(self, lower_bound):
        super().__init__()
        self.analysis_transform = model.AnalysisTransform()
        self.synthesis_transform = model.SynthesisTransform()
        self.hyper_encoder = model.HyperEncoder()
        self.hyper_decoder = model.HyperDecoder()
        self.entropy_bottleneck = EntropyBottleneck(channels=8)
        self.conditional_entropy_model = SymmetricConditional()
        self.lower_bound = lower_bound

    def forward(self, x, training=True):
        y = self.analysis_transform(x)
        z = self.hyper_encoder(y)
        z_tilde, likelihoods_hyper = self.entropy_bottleneck(z, quantize_mode="noise" if training else "symbols")
        loc, scale = self.hyper_decoder(z_tilde)
        scale = torch.clamp(scale, min=self.lower_bound) #这个函数对梯度没影响
        y_tilde, likelihoods = self.conditional_entropy_model(y, loc, scale, quantize_mode="noise" if training else "symbols")
        x_tilde = self.synthesis_transform(y_tilde)
        
        return {'likelihoods':likelihoods,
                'likelihoods_hyper':likelihoods_hyper,
                'x_tilde':x_tilde}


if __name__ == '__main__':
    model = PCCModel()
    print(model)

