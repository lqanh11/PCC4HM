LIST_CKPT_ORIGINAL=(
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000/ckpts/epoch_10.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth"        
)

LIST_CKPT_BASE=(
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/KeepCoordinates/20231230_encFIXa025_baseTRANc4_keepcoords_resolution128_alpha320.0_000/best_model/epoch_195.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/KeepCoordinates/20231230_encFIXa05_baseTRANc4_keepcoords_resolution128_alpha320.0_000/best_model/epoch_228.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/KeepCoordinates/20231230_encFIXa1_baseTRANc4_keepcoords_resolution128_alpha320.0_000/best_model/epoch_194.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/KeepCoordinates/20231230_encFIXa2_baseTRANc4_keepcoords_resolution128_alpha320.0_000/best_model/epoch_197.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/KeepCoordinates/20231230_encFIXa4_baseTRANc4_keepcoords_resolution128_alpha320.0_000/best_model/epoch_234.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/KeepCoordinates/20231230_encFIXa6_baseTRANc4_keepcoords_resolution128_alpha320.0_000/best_model/epoch_154.pth"
  "/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/KeepCoordinates/20231230_encFIXa10_baseTRANc4_keepcoords_resolution128_alpha320.0_000/best_model/epoch_261.pth"
)

LIST_ALPHA=(
    "20240104_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords"
    "20240104_encFIXa05_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords"
    "20240104_encFIXa1_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords"
    "20240104_encFIXa2_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords"
    "20240104_encFIXa4_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords"
    "20240104_encFIXa6_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords"
    "20240104_encFIXa10_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords"
)

# Get the length of one of the arrays (assuming both arrays have the same length)
length_ckpt=${#LIST_CKPT_ORIGINAL[@]}

for ((i=0; i<$length_ckpt; i++)); do
    ckptdir_original=${LIST_CKPT_ORIGINAL[i]}
    ckptdir_base=${LIST_CKPT_BASE[i]}

    prefix_text=${LIST_ALPHA[i]}

    echo "$prefix_text , $ckptdir_original"
    python train_scalable.py --alpha 1\
                             --resolution 128\
                             --prefix $prefix_text\
                             --init_ckpt_original $ckptdir_original\
                             --init_ckpt_base $ckptdir_base
    # for i in {16000,320,160,40,1}
    # do
    #     python train_cls_only.py --alpha $i --resolution 128 --prefix $prefix_text --init_ckpt_original $ckptdir
    # done
done