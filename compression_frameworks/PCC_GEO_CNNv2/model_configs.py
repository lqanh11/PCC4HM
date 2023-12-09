# from enum import Enum
import torch
import model_transforms
import numpy as np
# from model_transforms import TransformType
# from model_types import ModelType
from entropy_model import EntropyBottleneck
from GaussianConditional import GaussianConditional

# class ModelConfig:
#     def __init__(self, model_type: ModelType, model_params):
#         self.model_type = model_type
#         self.model_params = model_params

#     def build(self):
#         print("self.model_type:",self.model_type) #ModelType.v2
#         return self.model_type.value(**self.model_params) # **表示参数作为字典传入 .value是取Enum key的值

class C3PCompressionModelV2(torch.nn.Module):
    def __init__(self, num_filters=64,scale_lower_bound=1e-5,analysis_transform_type=model_transforms.AnalysisTransformProgressiveV2,
        synthesis_transform_type = model_transforms.SynthesisTransformProgressiveV2,
        hyper_analysis_transform_type = model_transforms.HyperAnalysisTransform,
        hyper_synthesis_transform_type = model_transforms.HyperSynthesisTransform,
        scales_min=0.11, scales_max=256, scales_levels=64, n_thresholds=2 ** 8):#,*args, **kwargs
        super().__init__()
        self.num_filters = num_filters
        self.analysis_transform = analysis_transform_type(in_channels=1,filters=num_filters)
        self.synthesis_transform = synthesis_transform_type(in_channels=64,filters=num_filters)
        self.hyper_analysis_transform = hyper_analysis_transform_type(in_channels=64,filters=num_filters)
        self.hyper_synthesis_transform = hyper_synthesis_transform_type(in_channels=64,filters=num_filters)
        # self.scale_table = np.exp(np.linspace(np.log(scales_min), np.log(scales_max), scales_levels)) #self.scale_table: [1.1e-01 1.24404103e-01...2.56e+02]
        self.entropy_bottleneck = EntropyBottleneck(channels=num_filters)
        self.conditional_bottleneck = GaussianConditional(scale_lower_bound=scale_lower_bound)
    
    def forward(self, x, training=True): #x.shape: (1, 1, 64, 64, 64)
        y = self.analysis_transform(x) #y.shape: (1, 64, 8, 8, 8)
        z = self.hyper_analysis_transform(y) #z.shape: (1, 64, 4, 4, 4)
        z_tilde, z_likelihoods,quantiles_loss = self.entropy_bottleneck(z, quantize_mode="noise" if training else "symbols")
        sigma_tilde = self.hyper_synthesis_transform(z_tilde) #sigma_tilde.shape: (1, 64, 8, 8, 8)
        y_tilde, y_likelihoods = self.conditional_bottleneck(y, scale=sigma_tilde, quantize_mode="noise" if training else "symbols")
        x_tilde = self.synthesis_transform(y_tilde)
        
        return {'y_likelihoods':y_likelihoods,
                'z_likelihoods':z_likelihoods,
                'x_tilde':x_tilde,
                'quantiles_loss':quantiles_loss}

# class ModelConfigType(Enum):
#     c1 = ModelConfig(ModelType.v1, {
#         'num_filters': 32,
#         'analysis_transform_type': TransformType.AnalysisTransformV1,
#         'synthesis_transform_type': TransformType.SynthesisTransformV1
#     })
#     c2 = ModelConfig(ModelType.v2, {
#         'num_filters': 32,
#         'analysis_transform_type': TransformType.AnalysisTransformV1,
#         'synthesis_transform_type': TransformType.SynthesisTransformV1,
#         'hyper_analysis_transform_type': TransformType.HyperAnalysisTransform,
#         'hyper_synthesis_transform_type': TransformType.HyperSynthesisTransform
#     })
#     c3 = ModelConfig(ModelType.v2, {
#         'num_filters': 32,
#         'analysis_transform_type': TransformType.AnalysisTransformV2,
#         'synthesis_transform_type': TransformType.SynthesisTransformV2,
#         'hyper_analysis_transform_type': TransformType.HyperAnalysisTransform,
#         'hyper_synthesis_transform_type': TransformType.HyperSynthesisTransform
#     })
#     c3p = ModelConfig(ModelType.v2, {
#         'num_filters': 64,
#         'analysis_transform_type': TransformType.AnalysisTransformProgressiveV2,
#         'synthesis_transform_type': TransformType.SynthesisTransformProgressiveV2,
#         'hyper_analysis_transform_type': TransformType.HyperAnalysisTransform,
#         'hyper_synthesis_transform_type': TransformType.HyperSynthesisTransform
#     })

#     @staticmethod
#     def keys():
#         return ModelConfigType.__members__.keys()

#     def build(self):
#         return self.value.build()
