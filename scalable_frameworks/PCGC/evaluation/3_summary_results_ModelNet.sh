#!/bin/bash


########### Pretrained Model ##################

# Define arrays for the two variables
# LIST_TEST_FILES=(
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/longdress_vox10_1300.ply" 
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/loot_vox10_1200.ply" 
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/redandblack_vox10_1550.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/soldier_vox10_0690.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/andrew_vox9_frame0000.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/david_vox9_frame0000.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/phil_vox9_frame0139.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/sarah_vox9_frame0023.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/basketball_player_vox11_00000200.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/dancer_vox11_00000001.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/exercise_vox11_00000001.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/model_vox11_00000001.ply"
#   )

# LIST_CKPT=(
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/finetuning_models/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000/ckpts/epoch_10.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/finetuning_models/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000/ckpts/epoch_10.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/finetuning_models/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000/ckpts/epoch_10.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/finetuning_models/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000/ckpts/epoch_10.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/finetuning_models/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000/ckpts/epoch_10.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/finetuning_models/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000/ckpts/epoch_10.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/finetuning_models/PCGCv2/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth"
# )

## load text file
# Check if a file path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <text_file>"
    exit 1
fi

# Read each line from the text file and store it in an array
file_path=$1
if [ -f "$file_path" ]; then
    # Read the lines into an array
    mapfile -t LIST_TEST_FILES < "$file_path"
else
    echo "File not found: $file_path"
fi

LIST_CKPT=(
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r1_0.025bpp.pth"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r2_0.05bpp.pth"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r3_0.10bpp.pth"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r4_0.15bpp.pth"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r5_0.25bpp.pth"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r6_0.3bpp.pth"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r7_0.4bpp.pth"
)

LIST_RESOLUTIONS=(
  # "1024"
  # "512"
  "256"
  # "128"
  # "64"
)


# Get the length of one of the arrays (assuming both arrays have the same length)
length_files=${#LIST_TEST_FILES[@]}
length_resolution=${#LIST_RESOLUTIONS[@]}

# Loop through the arrays in parallel
for ((i=0; i<$length_resolution; i++)); do
    resolution=${LIST_RESOLUTIONS[i]}
    for ((j=0; j<$length_files; j++)); do
      file_path=${LIST_TEST_FILES[j]}

      echo $file_path
      python summary_results_ModelNet.py --filedir=$file_path\
                                          --outdir="./output/ModelNet/scalable_full"\
                                          --res=$resolution\
                                          --ckptdir_list ${LIST_CKPT[@]}
    done
done



# ########### Pretrained Model ##################

# # Define arrays for the two variables
# LIST_TEST_FILES=(
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/longdress_vox10_1300.ply" 
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/loot_vox10_1200.ply" 
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/redandblack_vox10_1550.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/soldier_vox10_0690.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/andrew_vox9_frame0000.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/david_vox9_frame0000.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/phil_vox9_frame0139.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/sarah_vox9_frame0023.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/basketball_player_vox11_00000200.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/dancer_vox11_00000001.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/exercise_vox11_00000001.ply"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/model_vox11_00000001.ply"
#   )

# LIST_CKPT=(
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r1_0.025bpp.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r2_0.05bpp.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r3_0.10bpp.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r4_0.15bpp.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r5_0.25bpp.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r6_0.3bpp.pth"
#   "/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r7_0.4bpp.pth"
# )

# # Get the length of one of the arrays (assuming both arrays have the same length)
# length=${#LIST_TEST_FILES[@]}

# # Loop through the arrays in parallel
# for ((i=0; i<$length; i++)); do
#     file_path=${LIST_TEST_FILES[i]}
#     resolution=${LIST_RESOLUTIONS[i]}

#     echo $file_path
#     python summary_results.py --filedir=$file_path\
#                               --outdir="./output/MPEG/pretrained"\
#                               --ckptdir_list ${LIST_CKPT[@]}
# done
