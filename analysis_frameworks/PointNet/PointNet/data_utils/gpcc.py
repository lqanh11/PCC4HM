import os
import numpy as np 
import subprocess
from pyntcloud import PyntCloud

rootdir = os.path.split(__file__)[0]

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return

def read_ply_ascii_geo(filedir):
    xyz = ["x", "y", "z"]
    cloud = PyntCloud.from_file(str(filedir))
    coords = cloud.points[xyz].values
    return coords

def gpcc_encode(filedir, bin_dir, rate=None, show=False):
    """Compress point cloud:
    + losslessly --positionQuantizationScale = 1
    + lossily ---positionQuantizationScale = rate [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75]
    using MPEG G-PCCv12. 
    You can download and install TMC13 from 
    https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    if rate == None:
        subp=subprocess.Popen(rootdir+'/tmc3'+ 
                                ' --mode=0' + 
                                ' --positionQuantizationScale=1' + 
                                ' --trisoupNodeSizeLog2=0' + 
                                ' --neighbourAvailBoundaryLog2=8' + 
                                ' --intra_pred_max_node_size_log2=6' + 
                                ' --inferredDirectCodingMode=0' + 
                                ' --maxNumQtBtBeforeOt=4' +
                                ' --uncompressedDataPath='+filedir +                                            
                                ' --compressedStreamPath='+bin_dir, 
                                shell=True, stdout=subprocess.PIPE)
    else:
        subp=subprocess.Popen(rootdir+'/tmc3'+ 
                                ' --mode=0' + 
                                ' --positionQuantizationScale='+str(rate) + 
                                ' --trisoupNodeSizeLog2=0' + 
                                ' --neighbourAvailBoundaryLog2=8' + 
                                ' --intra_pred_max_node_size_log2=6' + 
                                ' --inferredDirectCodingMode=0' + 
                                ' --maxNumQtBtBeforeOt=4' +
                                ' --uncompressedDataPath='+filedir + 
                                ' --compressedStreamPath='+bin_dir, 
                                shell=True, stdout=subprocess.PIPE)

    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    
    return 

# def gpcc_encode(filedir, bin_dir, show=False):
#     """Compress point cloud losslessly using MPEG G-PCCv12. 
#     You can download and install TMC13 from 
#     https://github.com/MPEGGroup/mpeg-pcc-tmc13
#     """
#     subp=subprocess.Popen(rootdir+'/tmc3'+ 
#                             ' --mode=0' + 
#                             ' --positionQuantizationScale=1' + 
#                             ' --trisoupNodeSizeLog2=0' + 
#                             ' --neighbourAvailBoundaryLog2=8' + 
#                             ' --intra_pred_max_node_size_log2=6' + 
#                             ' --inferredDirectCodingMode=0' + 
#                             ' --maxNumQtBtBeforeOt=4' +
#                             ' --uncompressedDataPath='+filedir + 
#                             ' --compressedStreamPath='+bin_dir, 
#                             shell=True, stdout=subprocess.PIPE)
#     c=subp.stdout.readline()
#     while c:
#         if show: print(c)
#         c=subp.stdout.readline()
    
#     return 

def gpcc_decode(bin_dir, rec_dir, show=False):
    subp=subprocess.Popen(rootdir+'/tmc3'+ 
                            ' --mode=1'+ 
                            ' --compressedStreamPath='+bin_dir+ 
                            ' --reconstructedDataPath='+rec_dir+
                            ' --outputBinaryPly=0'
                          ,
                            shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)      
        c=subp.stdout.readline()
    
    return