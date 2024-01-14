
RATE_LIST=(
  "r1"
  "r2"
  "r3"
  "r4"
  "r5"
  "r6"
  "r7"
)
# LIST_PREFIX=(
#     "2024-01-14_11-03_encFIXa025_baseTRANc4_MLP_scalable_full_resolution64_alpha0.5_000"
#     "2024-01-14_11-24_encFIXa05_baseTRANc4_MLP_scalable_full_resolution64_alpha0.5_000"
#     "2024-01-14_11-48_encFIXa1_baseTRANc4_MLP_scalable_full_resolution64_alpha0.5_000"
#     "2024-01-14_12-09_encFIXa2_baseTRANc4_MLP_scalable_full_resolution64_alpha0.5_000"
#     "2024-01-14_12-40_encFIXa4_baseTRANc4_MLP_scalable_full_resolution64_alpha0.5_000"
#     "2024-01-14_13-07_encFIXa6_baseTRANc4_MLP_scalable_full_resolution64_alpha0.5_000"
#     "2024-01-14_13-47_encFIXa10_baseTRANc4_MLP_scalable_full_resolution64_alpha0.5_000"
# )
# RESOLUTION=64

LIST_PREFIX=(
    "2024-01-13_12-39_encFIXa025_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000"
    "2024-01-13_13-58_encFIXa05_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000"
    "2024-01-13_15-04_encFIXa1_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000"
    "2024-01-13_16-36_encFIXa2_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000"
    "2024-01-13_19-01_encFIXa4_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000"
    "2024-01-14_01-31_encFIXa6_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000"
    "2024-01-14_03-19_encFIXa10_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000"
)
RESOLUTION=128
    

# Get the length of one of the arrays (assuming both arrays have the same length)
length_prefix=${#LIST_PREFIX[@]}

for ((i=0; i<$length_prefix; i++)); do
    prefix_text=${LIST_PREFIX[i]}
    RATE=${RATE_LIST[i]}

    echo $prefix_text
    python ModelNet10_full_test.py --resolution $RESOLUTION\
                                    --rate $RATE\
                                    --prefix $prefix_text
                
done
