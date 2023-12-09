# Copyright (c) Nanjing University, Vision Lab.
# Last update:
# 2021.9.9

import os
import argparse
import numpy as np
import torch
import time
import importlib 
import subprocess
import math
#import models.model_voxception as model
#from models.entropy_model import EntropyBottleneck
#from models.conditional_entropy_model import SymmetricConditional

################### Compression Network (with conditional entropy model) ###################
def parallel_run(function,x,batch_parallel):
  for i in range(math.ceil(len(x)/batch_parallel)):
    end_idx = (i+1)*batch_parallel if len(x)-i*batch_parallel>batch_parallel else len(x)
    x0 = x[i*batch_parallel:end_idx]
    y_a = function(x0)
    if(i == 0):
      ys = y_a
    else:
      ys = torch.cat([ys, y_a], dim=0)
  return ys


@torch.no_grad()
def compress_hyper(cubes, model,device,batch_parallel):
  """Compress cubes to bitstream.
  Input: cubes with shape [batch size, length, width, height, channel(1)].
  Output: compressed bitstream.
  """

  print('===== Compress =====')

  x_cpu = torch.from_numpy(cubes.astype('float32')).permute(0,4,1,2,3) # (8, 64, 64, 64, 1)->(8, 1, 64, 64, 64)
  x = x_cpu.to(device)

  start = time.time()
  ys = parallel_run(model.analysis_transform,x,batch_parallel)

  print("Analysis Transform: {}s".format(round(time.time()-start, 4)))
  start = time.time()
  zs = model.hyper_encoder(ys) 
  print("Hyper Encoder: {}s".format(round(time.time()-start, 4)))

  z_hats, _ = model.entropy_bottleneck(zs, quantize_mode="symbols") 
  print("Quantize hyperprior.")

  start = time.time()
  locs, scales = model.hyper_decoder(z_hats)
  lower_bound = 1e-9
  scales = torch.clamp(scales, min=lower_bound)
  print("Hyper Decoder: {}s".format(round(time.time()-start, 4)))
  locs = torch.round(locs * 1e3) / 1e3
  scales = torch.round(scales * 1e3) / 1e3

  start = time.time()
  z_strings, z_min_v, z_max_v = model.entropy_bottleneck.compress(zs)
  z_shape = torch.tensor(zs.shape)
  print("Entropy Encode (Hyper): {}s".format(round(time.time()-start, 4)))

  start = time.time()
  y_shape = torch.tensor(ys.shape)
  y_strings, y_min_vs, y_max_vs = model.conditional_entropy_model.compress(ys, locs, scales)
  print("Entropy Encode: {}s".format(round(time.time()-start, 4)))

  return y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape

@torch.no_grad()
def decompress_hyper(y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape, model,device,batch_parallel):
  """Decompress bitstream to cubes.
  Input: compressed bitstream. latent representations (y) and hyper prior (z).
  Output: cubes with shape [batch size, length, width, height, channel(1)]
  """
  print('===== Decompress =====')

  start = time.time()
  zs = model.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape) #device随model，zs在CPU
  print("Entropy Decoder (Hyper): {}s".format(round(time.time()-start, 4)))

  start = time.time()
  zs_gpu = zs.to(device)
  locs, scales = model.hyper_decoder(zs_gpu)
  lower_bound = 1e-9
  scales = torch.clamp(scales, min=lower_bound)
  print("Hyper Decoder: {}s".format(round(time.time()-start, 4)))
  locs = torch.round(locs * 1e3) / 1e3
  scales = torch.round(scales * 1e3) / 1e3

  start = time.time()
  y_shape = tuple(y_shape)
  ys = model.conditional_entropy_model.decompress(y_strings, locs, scales, y_min_vs, y_max_vs, y_shape)
  print("Entropy Decoder: {}s".format(round(time.time()-start, 4)))
  print("ys.shape:",ys.shape)

  start = time.time()
  ys_gpu = ys.to(device)
  xs = parallel_run(model.synthesis_transform,ys_gpu,batch_parallel)
  #xs = model.synthesis_transform(ys_gpu)
  print("Synthesis Transform: {}s".format(round(time.time()-start, 4)))
  xs = xs.permute(0, 2, 3, 4, 1) #[N,C,D,H,W]->[N,D,H,W,channels]
  return xs.cpu()
