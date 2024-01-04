LIST_CKPT=(
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000/ckpts/epoch_10.pth"
  # "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth"        
)

LIST_ALPHA=(
    "20231230_encFIXa025_baseTRANc4_keepcoords"
    "20231230_encFIXa05_baseTRANc4_keepcoords"
    "20231230_encFIXa1_baseTRANc4_keepcoords"
    "20231230_encFIXa2_baseTRANc4_keepcoords"
    "20231230_encFIXa4_baseTRANc4_keepcoords"
    "20231230_encFIXa6_baseTRANc4_keepcoords"
    # "20231230_encFIXa10_baseTRANc4_keepcoords"
)

# Get the length of one of the arrays (assuming both arrays have the same length)
length_ckpt=${#LIST_CKPT[@]}

for ((i=0; i<$length_ckpt; i++)); do
    ckptdir=${LIST_CKPT[i]}
    prefix_text=${LIST_ALPHA[i]}

    echo "$prefix_text , $ckptdir"
    python train_cls_only.py --alpha 320 --resolution 128 --prefix $prefix_text --init_ckpt_original $ckptdir
    # for i in {16000,320,160,40,1}
    # do
    #     python train_cls_only.py --alpha $i --resolution 128 --prefix $prefix_text --init_ckpt_original $ckptdir
    # done
done