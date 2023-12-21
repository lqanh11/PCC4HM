# 512, 1024, 2048, 4096, 8192, 16384, 32768

python train_classification.py --model pointnet2_cls_ssg --num_point 512 --log_dir pointnet2_cls_ssg_512 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 1024 --log_dir pointnet2_cls_ssg_1024 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 2048 --log_dir pointnet2_cls_ssg_2048 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 4096 --log_dir pointnet2_cls_ssg_4096 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 8192 --log_dir pointnet2_cls_ssg_8192 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 16384 --log_dir pointnet2_cls_ssg_16384 &&\
python train_classification.py --model pointnet2_cls_ssg --num_point 32768 --log_dir pointnet2_cls_ssg_32768 