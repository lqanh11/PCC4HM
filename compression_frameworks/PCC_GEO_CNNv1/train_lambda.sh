#!/bin/bash

python train.py --lmbda=1e-5\
                --init_ckpt='./models/1e-05/epoch_25_30874.pth' &&\

python train.py --lmbda=5e-5\
                --init_ckpt='./models/5e-05/epoch_20_6319.pth' &&\

python train.py --lmbda=1e-4\
                --init_ckpt='./models/0.0001/epoch_28_34927.pth' &&\

python train.py --lmbda=5e-4\
                --init_ckpt='./models/0.0005/epoch_1_2110.pth' &&\

python train.py --lmbda=1e-3\
                --init_ckpt='./models/0.001/epoch_24_30503.pth' &&\

python train.py --lmbda=5e-3\
                --init_ckpt='./models/0.005/epoch_29_36758.pth'




# for lambda_value in {1e-5,5e-5,1e-4,5e-4,1e-3,5e-3}
# 0.00001, 0.00005, 0.0001, 0.0005, 0.001
# 1e-5, 5e-5, 1e-4, 5e-4, 1e-3

# {5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6}
# for lambda_value in 5
    # do
        # python train.py --lmbda=$lambda_value\                        
    # done
# done