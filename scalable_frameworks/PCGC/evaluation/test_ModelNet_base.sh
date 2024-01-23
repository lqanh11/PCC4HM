ALPHA_LIST_=(
  # "025"
  # "05"
  # "1"
  # "2"
  # "4"
  # "6"
  "10"
)
RATE_LIST=(
  # "r1"
  # "r2"
  # "r3"
  # "r4"
  # "r5"
  # "r6"
  "r7"
)

length_alpha=${#ALPHA_LIST_[@]}
for ((index=0; index<$length_alpha; index++)); do
    TEXT=${ALPHA_LIST_[index]}
    RATE=${RATE_LIST[index]}

    LIST_PREFIX=(
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha160.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha200.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha600.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha1000.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha4000.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha8000.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha12000.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha16000.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution256_alpha20000.0_000"
    )
    RESOLUTION=256

    # LIST_PREFIX=(
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha160.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha200.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha600.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha1000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha4000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha8000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha12000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha16000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha20000.0_000"
    # )
    # RESOLUTION=128

    # LIST_PREFIX=(
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha160.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha200.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha600.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha1000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha4000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha8000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha12000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha16000.0_000"
    #     "encFIXa$TEXT""_baseTRANc_mlp_resolution64_alpha20000.0_000"
    # )
    # RESOLUTION=64
    

    # Get the length of one of the arrays (assuming both arrays have the same length)
    length_prefix=${#LIST_PREFIX[@]}

    for ((i=0; i<$length_prefix; i++)); do
        prefix_text=${LIST_PREFIX[i]}

        echo $prefix_text
        python ModelNet10_base_test.py --resolution $RESOLUTION\
                                        --rate $RATE\
                                        --prefix $prefix_text\
                                        --logdir "/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/Proposed_Codec/"$RESOLUTION
                    
    done
done