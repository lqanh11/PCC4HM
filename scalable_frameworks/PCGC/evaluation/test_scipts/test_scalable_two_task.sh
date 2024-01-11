LIST_PREFIX=(
    "2024-01-05_16-41_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution64_alpha1.0_000"
    "2024-01-05_17-34_encFIXa05_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution64_alpha1.0_000"
    "2024-01-05_18-12_encFIXa1_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution64_alpha1.0_000"
    "2024-01-05_18-48_encFIXa2_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution64_alpha1.0_000"
    "2024-01-05_19-27_encFIXa4_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution64_alpha1.0_000"
    "2024-01-05_19-58_encFIXa6_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution64_alpha1.0_000"
    "2024-01-05_20-25_encFIXa10_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution64_alpha1.0_000"
)

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
    python test_scalable_two_task.py --resolution 64\
                                     --rate $rate\
                                     --prefix $prefix_text
                
done


######################################################


LIST_PREFIX=(
    "2024-01-04_22-17_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000"
    "2024-01-04_23-27_encFIXa05_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000"
    "2024-01-05_00-48_encFIXa1_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000"
    "2024-01-05_02-29_encFIXa2_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000"
    "2024-01-05_03-57_encFIXa4_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000"
    "2024-01-05_06-22_encFIXa6_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000"
    "2024-01-05_09-30_encFIXa10_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000"
)

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
    python test_scalable_two_task.py --resolution 128\
                                     --rate $rate\
                                     --prefix $prefix_text
                
done

###########################################################################################

LIST_PREFIX=(
    "2024-01-06_01-02_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000"
    "2024-01-06_04-27_encFIXa05_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000"
    "2024-01-06_09-31_encFIXa1_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000"
    "2024-01-06_16-38_encFIXa2_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000"
    "2024-01-06_22-09_encFIXa4_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000"
    "2024-01-07_04-02_encFIXa6_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000"
    "2024-01-07_07-58_encFIXa10_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000"
)

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
    python test_scalable_two_task.py --resolution 256\
                                     --rate $rate\
                                     --prefix $prefix_text
                
done