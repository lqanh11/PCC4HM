LIST_CKPT=(
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.25_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.5_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.0_000/ckpts/epoch_10.pth"        
)

# LIST_CKPT=(
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth"        
# )

LIST_ALPHA=(
    # "encFIXa025_baseTRANc_mlp"
    # "encFIXa05_baseTRANc_mlp"
    "encFIXa1_baseTRANc_mlp"
    # "encFIXa2_baseTRANc_mlp"
    # "encFIXa4_baseTRANc_mlp"
    # "encFIXa6_baseTRANc_mlp"
    # "encFIXa10_baseTRANc_mlp"
)
length_ckpt=${#LIST_CKPT[@]}

echo $length_ckpt

for ((i=0; i<$length_ckpt; i++)); do
    ckptdir=${LIST_CKPT[i]}
    prefix_text=${LIST_ALPHA[i]}

    echo "$prefix_text , $ckptdir"
    # for alpha_value in {20000,16000,12000,8000,4000,1000,600,200,160,120,80,40,10,1}
    for alpha_value in {1000,600,200,160}
    do
        python scalable_MLP_base_train_early.py --alpha $alpha_value --resolution 256 --prefix $prefix_text --init_ckpt_original $ckptdir
    done
done

LIST_CKPT=(
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000/ckpts/epoch_10.pth"
#   "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth"        
)

LIST_ALPHA=(
    # "encFIXa025_baseTRANc_mlp"
    # "encFIXa05_baseTRANc_mlp"
    # "encFIXa1_baseTRANc_mlp"
    "encFIXa2_baseTRANc_mlp"
    "encFIXa4_baseTRANc_mlp"
    "encFIXa6_baseTRANc_mlp"
    "encFIXa10_baseTRANc_mlp"
)
length_ckpt=${#LIST_CKPT[@]}

echo $length_ckpt

for ((i=0; i<$length_ckpt; i++)); do
    ckptdir=${LIST_CKPT[i]}
    prefix_text=${LIST_ALPHA[i]}

    echo "$prefix_text , $ckptdir"
    # for alpha_value in {20000,16000,12000,8000,4000,1000,600,200,160,120,80,40,10,1}
    for alpha_value in {20000,16000,12000,8000,4000,1000,600,200,160}
    do
        python scalable_MLP_base_train_early.py --alpha $alpha_value --resolution 256 --prefix $prefix_text --init_ckpt_original $ckptdir
    done
done


