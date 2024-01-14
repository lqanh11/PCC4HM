# LIST_PREFIX=(
#     "modelnet_dense_full_reconstruction_with_pretrained_resolution256_alpha0.25_000"
#     "modelnet_dense_full_reconstruction_with_pretrained_resolution256_alpha0.5_000"
#     "modelnet_dense_full_reconstruction_with_pretrained_resolution256_alpha1.0_000"
#     "modelnet_dense_full_reconstruction_with_pretrained_resolution256_alpha2.0_000"
#     "modelnet_dense_full_reconstruction_with_pretrained_resolution256_alpha4.0_000"
#     "modelnet_dense_full_reconstruction_with_pretrained_resolution256_alpha6.0_000"
#     "modelnet_dense_full_reconstruction_with_pretrained_resolution256_alpha10.0_000"
# )
# RESOLUTION=512

# LIST_RATE=(
#   "r1"
#   "r2"
#   "r3"
#   "r4"
#   "r5"
#   "r6"
#   "r7"
# )

# # Get the length of one of the arrays (assuming both arrays have the same length)
# length_prefix=${#LIST_PREFIX[@]}

# for ((i=0; i<$length_prefix; i++)); do
#     prefix_text=${LIST_PREFIX[i]}
#     rate=${LIST_RATE[i]}

#     echo "$prefix_text , $ckptdir_original"
#     python ModelNet10_test.py --resolution $RESOLUTION\
#                               --rate $rate\
#                               --prefix $prefix_text
                
# done

LIST_PREFIX=(
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000"
)
RESOLUTION=256

LIST_RATE=(
  "r1"
  "r2"
  "r3"
  "r4"
  "r5"
  "r6"
  "r7"
)

# Get the length of one of the arrays (assuming both arrays have the same length)
length_prefix=${#LIST_PREFIX[@]}

for ((i=0; i<$length_prefix; i++)); do
    prefix_text=${LIST_PREFIX[i]}
    rate=${LIST_RATE[i]}

    echo "$prefix_text , $ckptdir_original"
    python ModelNet10_test.py --resolution $RESOLUTION\
                              --rate $rate\
                              --prefix $prefix_text
                
done

LIST_PREFIX=(
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000"
)
RESOLUTION=128

LIST_RATE=(
  "r1"
  "r2"
  "r3"
  "r4"
  "r5"
  "r6"
  "r7"
)

# Get the length of one of the arrays (assuming both arrays have the same length)
length_prefix=${#LIST_PREFIX[@]}

for ((i=0; i<$length_prefix; i++)); do
    prefix_text=${LIST_PREFIX[i]}
    rate=${LIST_RATE[i]}

    echo "$prefix_text , $ckptdir_original"
    python ModelNet10_test.py --resolution $RESOLUTION\
                              --rate $rate\
                              --prefix $prefix_text
                
done

LIST_PREFIX=(
    "modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.25_000"
    "modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha0.5_000"
    "modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha1.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha2.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha4.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha6.0_000"
    "modelnet_dense_full_reconstruction_with_pretrained_resolution64_alpha10.0_000"
)
RESOLUTION=64

LIST_RATE=(
  "r1"
  "r2"
  "r3"
  "r4"
  "r5"
  "r6"
  "r7"
)

# Get the length of one of the arrays (assuming both arrays have the same length)
length_prefix=${#LIST_PREFIX[@]}

for ((i=0; i<$length_prefix; i++)); do
    prefix_text=${LIST_PREFIX[i]}
    rate=${LIST_RATE[i]}

    echo "$prefix_text , $ckptdir_original"
    python ModelNet10_test.py --resolution $RESOLUTION\
                              --rate $rate\
                              --prefix $prefix_text
                
done