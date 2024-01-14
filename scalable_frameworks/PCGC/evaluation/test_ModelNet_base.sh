ALPHA_LIST_=(
  "025"
  "05"
  "1"
  "2"
  "4"
  "6"
  "10"
)
RATE_LIST=(
  "r1"
  "r2"
  "r3"
  "r4"
  "r5"
  "r6"
  "r7"
)

length_alpha=${#ALPHA_LIST_[@]}
for ((index=0; index<$length_alpha; index++)); do
    TEXT=${ALPHA_LIST_[index]}
    RATE=${RATE_LIST[index]}

    LIST_PREFIX=(
        "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha16000.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha640.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha160.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha80.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha40.0_000"
        "encFIXa$TEXT""_baseTRANc_mlp_resolution128_alpha1.0_000"
    )
    RESOLUTION=128
    

    # Get the length of one of the arrays (assuming both arrays have the same length)
    length_prefix=${#LIST_PREFIX[@]}

    for ((i=0; i<$length_prefix; i++)); do
        prefix_text=${LIST_PREFIX[i]}

        echo $prefix_text
        python ModelNet10_base_test.py --resolution $RESOLUTION\
                                        --rate $RATE\
                                        --prefix $prefix_text
                    
    done
done