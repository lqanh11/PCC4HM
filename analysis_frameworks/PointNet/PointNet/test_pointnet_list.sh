LIST_RATE=(
  "0.0125" "0.03125" "0.0625" "0.125" "0.25" "0.375" "0.5" "0.625" "0.75" "0.875" "0.99"
)
length_rate=${#LIST_RATE[@]}

# # Loop through the arrays in parallel
# for ((i=0; i<$length_rate; i++)); do
#     rate=${LIST_RATE[i]}
#     python test_classification_voxelized.py --model pointnet2_cls_ssg\
#                                             --num_point 1024\
#                                             --resolution 64\
#                                             --rate $rate\
#                                             --use_uniform_sample\
#                                             --log_dir pointnet2_cls_ssg_r64_n1024_fps
# done

# Loop through the arrays in parallel
for ((i=0; i<$length_rate; i++)); do
    rate=${LIST_RATE[i]}
    python test_classification_voxelized.py --model pointnet2_cls_ssg\
                                            --num_point 1024\
                                            --resolution 256\
                                            --rate $rate\
                                            --use_uniform_sample\
                                            --log_dir pointnet2_cls_ssg_r128_n1024_fps
done

# python test_classification_voxelized.py --model pointnet2_cls_ssg --num_point 1024 --resolution 128 --use_uniform_sample --log_dir pointnet2_cls_ssg_r128_n1024_fps &&\
# python test_classification_voxelized.py --model pointnet2_cls_ssg --num_point 1024 --resolution 256 --use_uniform_sample --log_dir pointnet2_cls_ssg_r256_n1024_fps &&\

# python test_classification_voxelized.py --model pointnet_cls --num_point 1024 --resolution 64 --use_uniform_sample --log_dir pointnet_cls_r64_n1024_fps &&\
# python test_classification_voxelized.py --model pointnet_cls --num_point 1024 --resolution 128 --use_uniform_sample --log_dir pointnet_cls_r128_n1024_fps &&\
# python test_classification_voxelized.py --model pointnet_cls --num_point 1024 --resolution 256 --use_uniform_sample --log_dir pointnet_cls_r256_n1024_fps 


# #!/bin/bash


# ########### Pretrained Model ##################
# ## load text file
# # Check if a file path is provided as an argument
# if [ $# -eq 0 ]; then
#     echo "Usage: $0 <text_file>"
#     exit 1
# fi

# # Read each line from the text file and store it in an array
# file_path=$1
# if [ -f "$file_path" ]; then
#     # Read the lines into an array
#     mapfile -t LIST_TEST_FILES < "$file_path"
# else
#     echo "File not found: $file_path"
# fi


# LIST_RESOLUTIONS=(
#   # "1024"
#   # "512"
#   "256"
#   # "128"
#   "64"
# )


# LIST_RATE=(
#   "0.125" "0.25" "0.375" "0.5" "0.625" "0.75" "0.875" "0.99"
# )

# # Get the length of one of the arrays (assuming both arrays have the same length)
# length_files=${#LIST_TEST_FILES[@]}
# length_resolution=${#LIST_RESOLUTIONS[@]}

# # Loop through the arrays in parallel
# for ((i=0; i<$length_resolution; i++)); do
#     resolution=${LIST_RESOLUTIONS[i]}
#     for ((j=0; j<$length_files; j++)); do
#       file_path=${LIST_TEST_FILES[j]}
      
#       echo "file_path: $file_path, resolution: $resolution"
#       python test_ModelNet_file.py --filedir=$file_path\
#                                    --outdir="./output/ModelNet/lossy"\
#                                    --res=$resolution\
#                                    --ckptdir_list ${LIST_RATE[@]}
#     done
# done