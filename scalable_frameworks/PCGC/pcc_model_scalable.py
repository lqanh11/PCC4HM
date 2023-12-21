import torch
import torch.nn.functional as F
import MinkowskiEngine as ME

from autoencoder import Encoder, Decoder, Adapter, TransposeAdapter, LatentSpaceTransform
from entropy_model import EntropyBottleneck
from classification_model import MinkowskiPointNet, MinkowskiFCNN, MinkoPointNet_Conv_2


class PCCModel_Scalable_ForBest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)
        self.entropy_bottleneck_b = EntropyBottleneck(4)
        self.entropy_bottleneck_e = EntropyBottleneck(4)
        self.adapter = Adapter(channels=[8,4])
        self.transpose_adapter = TransposeAdapter(channels=[4,8])
        # self.latentspace_transform = LatentSpaceTransform(channels=[8,8])
        self.classifier = MinkoPointNet_Conv_2(in_channel=4, out_channel=10, embedding_channel=1024)
        
        self.analysis_residual = Adapter(channels=[8,4])
        self.systhesis_residual = TransposeAdapter(channels=[4,8])

    def get_likelihood_o(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def get_likelihood_b(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_b(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood
    
    def get_likelihood_e(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_e(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, x_fix_pts, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model - Original
        y_q, likelihood = self.get_likelihood_o(y, 
            quantize_mode="noise" if training else "symbols")
        
        # Quantizer & Entropy Model - Scalable Coding - Base
        z = self.adapter(y)
        z_q, likelihood_b = self.get_likelihood_b(z, 
            quantize_mode="noise" if training else "symbols")
        
        # Classification
        # num_points = [[len(C) for C in x.decomposed_coordinates] \
        #     for x in [x_fix_pts]]

        # pred_fix_pts = self.latentspace_transform(z_q, 
        #                                     num_points=num_points[0], 
        #                                     x_fix_pts=x_fix_pts, 
        #                                     training=training)
        
        # points_norm = F.normalize(pred_fix_pts.C[:,1:4].to(torch.float32))
        

        # input_classifier = ME.TensorField(
        #     coordinates=pred_fix_pts.C,
        #     features=points_norm,
        #     device=pred_fix_pts.device
        # )
        logits = self.classifier(z_q)

        # Transpose adapter
        nums_list_for_e = [[len(C) for C in y.decomposed_coordinates] \
            for y in [y]]
        
        y_b = self.transpose_adapter(z_q, nums_list_for_e, [y], training)

        y_r_features =  y.F - y_b.F


        y_r = ME.SparseTensor(
            features=y_r_features, 
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, 
            device=y.device)
        # y_r = ME.SparseTensor(
        #     coordinates=y.C,
        #     features=y_r_features,
        #     device=pred_fix_pts.device
        # )

        z_r = self.analysis_residual(y_r)

        # Quantizer & Entropy Model - Scalable Coding - Residual
        z_r_q, likelihood_e = self.get_likelihood_e(z_r, 
            quantize_mode="noise" if training else "symbols")
        
        y_r_hat = self.systhesis_residual(z_r_q, nums_list_for_e, [y], training)
        
        y_scalable_features = y_r_hat.F + y_b.F

        y_scalable = ME.SparseTensor(
            features=y_scalable_features, 
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, 
            device=y.device)
        
        # Decoder
        out_cls_list, out = self.decoder(y_scalable, nums_list, ground_truth_list, training)

        return {'out':out,
                'logits':logits,
                'out_cls_list':out_cls_list,
                'prior_original':[y_q],
                'prior_scalable':[y_scalable],
                'pred_LST':[x_fix_pts],
                'groundtruth_LST':[x_fix_pts],  
                'likelihood':likelihood, 
                'likelihood_b':likelihood_b,
                'likelihood_e':likelihood_e,  
                'ground_truth_list':ground_truth_list,
                'nums_list': nums_list
                }

class PCCModel_Scalable_BCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)
        self.entropy_bottleneck_b = EntropyBottleneck(4)
        self.entropy_bottleneck_e = EntropyBottleneck(8)
        self.adapter = Adapter(channels=[8,4])
        self.transpose_adapter = TransposeAdapter(channels=[4,8])
        self.latentspace_transform = LatentSpaceTransform(channels=[4,8])
        self.classifier = MinkowskiFCNN(in_channel=3, out_channel=10, embedding_channel=1024)
        

    def get_likelihood_o(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def get_likelihood_b(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_b(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood
    
    def get_likelihood_e(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_e(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, x_fix_pts, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model - Original
        y_q, likelihood = self.get_likelihood_o(y, 
            quantize_mode="noise" if training else "symbols")
        
        # Quantizer & Entropy Model - Scalable Coding - Base
        z = self.adapter(y)
        z_q, likelihood_b = self.get_likelihood_b(z, 
            quantize_mode="noise" if training else "symbols")
        
        # Classification
        num_points = [[len(C) for C in x.decomposed_coordinates] \
            for x in [x_fix_pts]]

        pred_fix_pts = self.latentspace_transform(z_q, 
                                            num_points=num_points[0], 
                                            x_fix_pts=x_fix_pts, 
                                            training=training)
        
        points_norm = F.normalize(pred_fix_pts.C[:,1:4].to(torch.float32))
        input_classifier = ME.TensorField(
            coordinates=pred_fix_pts.C,
            features=points_norm,
            device=pred_fix_pts.device
        )
        logits = self.classifier(input_classifier)

        # Transpose adapter
        nums_list_for_e = [[len(C) for C in y.decomposed_coordinates] \
            for y in [y]]
        
        y_b = self.transpose_adapter(z_q, nums_list_for_e, [y], training)

        y_r_features =  y.F - y_b.F

        y_r = ME.SparseTensor(
            coordinates=y.C,
            features=y_r_features,
            device=pred_fix_pts.device
        )
        # Quantizer & Entropy Model - Scalable Coding - Residual
        y_r_q, likelihood_e = self.get_likelihood_e(y_r, 
            quantize_mode="noise" if training else "symbols")
        
        y_scalable_features = y_r_q.F + y_b.F

        y_scalable = ME.SparseTensor(
            features=y_scalable_features, 
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, 
            device=y.device)
        
        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out':out,
                'logits':logits,
                'out_cls_list':out_cls_list,
                'prior_original':[y_q],
                'prior_scalable':[y_scalable],
                'pred_LST':[pred_fix_pts],
                'groundtruth_LST':[x_fix_pts],  
                'likelihood':likelihood, 
                'likelihood_b':likelihood_b,
                'likelihood_e':likelihood_e,  
                'ground_truth_list':ground_truth_list,
                'nums_list': nums_list
                }

class PCCModel_Scalable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)
        self.entropy_bottleneck_b = EntropyBottleneck(4)
        self.entropy_bottleneck_e = EntropyBottleneck(8)
        self.adapter = Adapter(channels=[8,4])
        self.transpose_adapter = TransposeAdapter(channels=[4,8])
        self.latentspace_transform = LatentSpaceTransform(channels=[4,8])
        self.classifier = MinkoPointNet_Conv_2(in_channel=3, out_channel=10, embedding_channel=1024)
        

    def get_likelihood_o(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def get_likelihood_b(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_b(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood
    
    def get_likelihood_e(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_e(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model - Original
        y_q, likelihood = self.get_likelihood_o(y, 
            quantize_mode="noise" if training else "symbols")
        
        # Quantizer & Entropy Model - Scalable Coding - Base
        z = self.adapter(y)
        z_q, likelihood_b = self.get_likelihood_b(z, 
            quantize_mode="noise" if training else "symbols")
        
        # Classification
        points = self.latentspace_transform(z_q, num_points=[1024] * len(nums_list[2]))
        points_norm = F.normalize(points[:,1:4].to(torch.float32))
        input_classifier = ME.TensorField(
            coordinates=points,
            features=points_norm,
            device=points.device
        )
        logits = self.classifier(input_classifier)

        # Transpose adapter
        nums_list_for_e = [[len(C) for C in y.decomposed_coordinates] \
            for y in [y]]
        
        y_b = self.transpose_adapter(z_q, nums_list_for_e, [y], training)

        y_r_features =  y.F - y_b.F

        y_r = ME.SparseTensor(
            coordinates=y.C,
            features=y_r_features,
            device=points.device
        )
        # Quantizer & Entropy Model - Scalable Coding - Residual
        y_r_q, likelihood_e = self.get_likelihood_e(y_r, 
            quantize_mode="noise" if training else "symbols")
        
        y_scalable_features = y_r_q.F + y_b.F

        y_scalable = ME.SparseTensor(
            features=y_scalable_features, 
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, 
            device=y.device)
        
        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out':out,
                'logits':logits,
                'out_cls_list':out_cls_list,
                'prior_original':[y_q],
                'prior_scalable':[y_scalable],  
                'likelihood':likelihood, 
                'likelihood_b':likelihood_b,
                'likelihood_e':likelihood_e,  
                'ground_truth_list':ground_truth_list,
                'nums_list': nums_list
                }

class PCCModel_Classification_Split(torch.nn.Module):
    def __init__(self, split_channel=3):
        super().__init__()
        self.split_channel = split_channel

        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)
        self.entropy_bottleneck_b = EntropyBottleneck(channels=self.split_channel)
        self.entropy_bottleneck_e = EntropyBottleneck(channels= 8 - self.split_channel)
        self.classifier = MinkoPointNet_Conv_2(in_channel=self.split_channel, out_channel=10, embedding_channel=1024)

    def get_likelihood_o(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)
        
        return data_Q, likelihood

    def get_likelihood_b(self, data, split, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_b(data.F[:,:split],
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)
        
        return data_Q, likelihood
    
    def get_likelihood_e(self, data, split, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck_e(data.F[:,split:],
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, x_fix_pts, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model - Original
        y_q, likelihood = self.get_likelihood_o(y, 
            quantize_mode="noise" if training else "symbols")

        # Quantizer & Entropy Model
        split = self.split_channel
        y_b_q, likelihood_b = self.get_likelihood_b(y, split,
            quantize_mode="noise" if training else "symbols")
        
        y_e_q, likelihood_e = self.get_likelihood_e(y, split,
            quantize_mode="noise" if training else "symbols")
        
        # classification
        logits = self.classifier(y_b_q)

        y_hat = ME.SparseTensor(
            features=torch.cat([y_b_q.F, y_e_q.F], dim=1), 
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, 
            device=y.device)

        # Decoder
        out_cls_list, out = self.decoder(y_hat, nums_list, ground_truth_list, training)

        return {'out':out,
                'logits': logits,
                'out_cls_list':out_cls_list,
                'prior':y_hat, 
                'likelihood':likelihood, 
                'likelihood_b':likelihood_b,
                'likelihood_e':likelihood_e,   
                'ground_truth_list':ground_truth_list,
                'nums_list': nums_list
                }

class PCCModel_Classification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)
        self.classifier = MinkoPointNet_Conv_2(in_channel=8, out_channel=10, embedding_channel=1024)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, x_fix_pts, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y, 
            quantize_mode="noise" if training else "symbols")
        
        # classification
        logits = self.classifier(y_q)

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out':out,
                'logits': logits,
                'out_cls_list':out_cls_list,
                'prior':y_q, 
                'likelihood':likelihood, 
                'ground_truth_list':ground_truth_list,
                'nums_list': nums_list
                }

class PCCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x] 
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y, 
            quantize_mode="noise" if training else "symbols")

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out':out,
                'out_cls_list':out_cls_list,
                'prior':y_q, 
                'likelihood':likelihood, 
                'ground_truth_list':ground_truth_list,
                'nums_list': nums_list
                }

