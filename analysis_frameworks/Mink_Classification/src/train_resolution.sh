DATASET_FOLDERS=(
  "/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_modelnet10_ply_hdf5_1024/voxelized_64" 
  "/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_modelnet10_ply_hdf5_1024/voxelized_128"
  "/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_modelnet10_ply_hdf5_1024/voxelized_256"
  "/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_modelnet10_ply_hdf5_1024/voxelized_512"
  "/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_modelnet10_ply_hdf5_1024/voxelized_1024"
)

MODEL_LIST=(
    "minkpointnet"
    "minksplatfcnn"
    "minkpointnet_conv_2"
)

for dataset in "${DATASET_FOLDERS[@]}"
do 
  echo $dataset
  for model_name in "${MODEL_LIST[@]}"
    do
    python classification_modelnet10_voxelize.py --data_root=$dataset\
                                                 --num_points=1024\
                                                 --network=$model_name\
                                                       
    done
done