LIST_CKPT=(
#   "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000/ckpts/epoch_10.pth"
#   "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000/ckpts/epoch_10.pth"
#   "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth"        
)

LIST_ALPHA=(
    # "encFIXa025_baseTRANc_mlp_cls_only"
    # "encFIXa05_baseTRANmlp_cls_only"
    # "encFIXa1_baseTRANmlp_cls_only"
    "encFIXa2_baseTRANmlp_cls_only"
    "encFIXa4_baseTRANmlp_cls_only"
    "encFIXa6_baseTRANmlp_cls_only"
    "encFIXa10_baseTRANmlp_cls_only"
)

# Get the length of one of the arrays (assuming both arrays have the same length)
length_ckpt=${#LIST_CKPT[@]}

for ((i=0; i<$length_ckpt; i++)); do
    ckptdir=${LIST_CKPT[i]}
    prefix_text=${LIST_ALPHA[i]}

    echo "$prefix_text , $ckptdir"

    # python train_cls_MLP.py --alpha 16000 --resolution 64 --prefix $prefix_text --init_ckpt_original $ckptdir

    for i in {16000,640,160,80,40,1}
    do
        python cls_MLP_train.py --alpha $i --resolution 128 --prefix $prefix_text --init_ckpt_original $ckptdir
    done

    # # for res in {64,128,256}
    # for res in {256}
    # do
    #     for i in {16000,640,160,80,40,1}
    #     do
    #         python train_cls_MLP.py --alpha $i --resolution $res --prefix $prefix_text --init_ckpt_original $ckptdir
    #     done
    # done
done