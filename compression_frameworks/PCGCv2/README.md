# PCC4HM
Point Cloud Compression for Human and Machine Vision

## Data preprocessing
### ModelNet40

This instruction is from https://github.com/multimedialabsfu/learned-point-cloud-compression-for-classification 
Download and *repair* the [ModelNet40] dataset (it has incorrect OFF headers!) by running:

```bash
# You should download the ModelNet40 dataset and unzip it to get the ModelNet40 directory
# The download link: http://modelnet.cs.princeton.edu/ModelNet40.zip
export ModelNet40_Path=...
python datapre_processing/repair_modelnet.py --input_dir={$ModelNet40}
mv ModelNet40 format=off
```
### Generate the Point Cloud dataset

Problem: off file coordinate are too big modelnet40 (e.g., test/airplane_0645.off)
-> Need to increase the resolution 

```bash
cd data_preprocessing
python generate_dataset_ModelNet40.py
```

## Data loader
### CSV file to save file directory
```bash
cd data_split
python create_csv_file.py
```
### Dataset 
Problem: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfc in position 120: invalid start byte
Maybe from the tool to save Point Cloud and code to read Point Cloud from PLY are difference.
