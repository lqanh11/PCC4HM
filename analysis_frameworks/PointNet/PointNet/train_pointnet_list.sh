# 512, 1024, 2048, 4096, 8192, 16384, 32768

# # python train_classification.py --model pointnet_cls --num_point 512 --log_dir pointnet_cls_512 &&\
# python train_classification.py --model pointnet_cls --num_point 1024 --log_dir pointnet_cls_1024 &&\
# python train_classification.py --model pointnet_cls --num_point 2048 --log_dir pointnet_cls_2048 &&\
# python train_classification.py --model pointnet_cls --num_point 4096 --log_dir pointnet_cls_4096 &&\
# python train_classification.py --model pointnet_cls --num_point 8192 --log_dir pointnet_cls_8192 &&\
# python train_classification.py --model pointnet_cls --num_point 16384 --log_dir pointnet_cls_16384 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 1024 --resolution 64 --log_dir pointnet2_cls_ssg_r64_n1024 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 1024 --resolution 128 --log_dir pointnet2_cls_ssg_r128_n1024 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 1024 --resolution 256 --log_dir pointnet2_cls_ssg_r256_n1024 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 1024 --resolution 512 --log_dir pointnet2_cls_ssg_r512_n1024 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 1024 --resolution 1024 --log_dir pointnet2_cls_ssg_r1024_n1024 
