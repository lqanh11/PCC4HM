#!/bin/bash
# Define arrays for the two variables
LIST_TEST_FILES=(
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/longdress_vox10_1300.ply" 
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/loot_vox10_1200.ply" 
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/redandblack_vox10_1550.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/soldier_vox10_0690.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/andrew_vox9_frame0000.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/david_vox9_frame0000.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/phil_vox9_frame0139.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/MVUB/sarah_vox9_frame0023.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/basketball_player_vox11_00000200.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/dancer_vox11_00000001.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/exercise_vox11_00000001.ply"
  "/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/Owlii/model_vox11_00000001.ply"
  )
  
LIST_RESOLUTIONS=(
  "1024"
  "1024"
  "1024"
  "1024"
  "512"
  "512"
  "512"
  "512"
  "2048"
  "2048"
  "2048"
  "2048"
)


# Get the length of one of the arrays (assuming both arrays have the same length)
length=${#LIST_TEST_FILES[@]}

# Loop through the arrays in parallel
for ((i=0; i<$length; i++)); do
    file_path=${LIST_TEST_FILES[i]}
    resolution=${LIST_RESOLUTIONS[i]}
    
    echo "file_path: $file_path, resolution: $resolution"
    python calculate_BD_rates_MPEG.py --filedir=$file_path\
                                --resolution=$resolution
done

