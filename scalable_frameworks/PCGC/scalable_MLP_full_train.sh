LIST_CKPT_ORIGINAL=(
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.25_000/ckpts/epoch_10.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.5_000/ckpts/epoch_10.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha1.0_000/ckpts/epoch_10.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha2.0_000/ckpts/epoch_10.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha4.0_000/ckpts/epoch_10.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha6.0_000/ckpts/epoch_10.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha10.0_000/ckpts/epoch_10.pth"        
)

LIST_CKPT_BASE=(
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/64/encFIXa025_baseTRANc_mlp_resolution64_alpha4000.0_000/best_model/best_model.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/64/encFIXa05_baseTRANc_mlp_resolution64_alpha4000.0_000/best_model/best_model.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/64/encFIXa1_baseTRANc_mlp_resolution64_alpha4000.0_000/best_model/best_model.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/64/encFIXa2_baseTRANc_mlp_resolution64_alpha4000.0_000/best_model/best_model.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/64/encFIXa4_baseTRANc_mlp_resolution64_alpha4000.0_000/best_model/best_model.pth"
  # "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/64/encFIXa6_baseTRANc_mlp_resolution64_alpha4000.0_000/best_model/best_model.pth"
  "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/64/encFIXa10_baseTRANc_mlp_resolution64_alpha4000.0_000/best_model/best_model.pth"
)
RESOLUTION=64

LIST_ALPHA=(
    # "encFIXa025_baseTRANc4_MLP_scalable_full"
    # "encFIXa05_baseTRANc4_MLP_scalable_full"
    # "encFIXa1_baseTRANc4_MLP_scalable_full"
    # "encFIXa2_baseTRANc4_MLP_scalable_full"
    # "encFIXa4_baseTRANc4_MLP_scalable_full"
    # "encFIXa6_baseTRANc4_MLP_scalable_full"
    "encFIXa10_baseTRANc4_MLP_scalable_full"
)

# Get the length of one of the arrays (assuming both arrays have the same length)
length_ckpt=${#LIST_CKPT_ORIGINAL[@]}

for ((i=0; i<$length_ckpt; i++)); do
    ckptdir_original=${LIST_CKPT_ORIGINAL[i]}
    ckptdir_base=${LIST_CKPT_BASE[i]}

    prefix_text=${LIST_ALPHA[i]}

    echo "$prefix_text , $ckptdir_original"
    python scalable_MLP_full_train.py \
                             --resolution $RESOLUTION\
                             --prefix $prefix_text\
                             --init_ckpt_original $ckptdir_original\
                             --init_ckpt_base $ckptdir_base
    # for i in {16000,320,160,40,1}
    # do
    #     python train_cls_only.py --alpha $i --resolution 128 --prefix $prefix_text --init_ckpt_original $ckptdir
    # done
done

