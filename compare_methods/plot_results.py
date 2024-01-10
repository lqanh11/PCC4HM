import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

workspace_path = '/media/avitech-pc2/Student/quocanhle/Point_Cloud/'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", 
                        default='dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0117.h5')

    parser.add_argument("--outdir_pcgcv2", 
                        default="compression_frameworks/PCGCv2/evaluation/output/ModelNet/pretrained")
    
    parser.add_argument("--outdir_pcgcv2_f", 
                        default="")
    
    parser.add_argument("--outdir_gpcc", 
                        default="compression_frameworks/G-PCC/evaluation/output/ModelNet/lossy")
    
    parser.add_argument("--outdir_geov1", 
                        default="")
    
    parser.add_argument("--outdir_geov2", 
                        default="")
    
    parser.add_argument("--outdir_pcgcv1", 
                        default="")
    
    parser.add_argument("--resolution",
                        default="64")

    args = parser.parse_args()

    basename = os.path.split(args.filedir)[-1].split('.')[0]
    
    ## Rate-distortion efficiency
    # D1
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    if args.outdir_gpcc != '': 
        csv_gpcc= os.path.join(workspace_path, args.outdir_gpcc, args.resolution, basename +'.csv')
        gpcc_results = pd.read_csv(csv_gpcc)
        axs[0].plot(np.array(gpcc_results["bpp"][:]), np.array(gpcc_results["mseF,PSNR (p2point)"][:]), 
                    label="G-PCC", marker='x', color='red')
        axs[1].plot(np.array(gpcc_results["bpp"][:]), np.array(gpcc_results["mseF,PSNR (p2plane)"][:]), 
                    label="G-PCC", marker='x', color='red')

    if args.outdir_pcgcv1 != '': 
        csv_pcgcv1= os.path.join(workspace_path, args.outdir_pcgcv1, args.resolution, basename +'.csv')
        pcgcv1_results = pd.read_csv(csv_pcgcv1)
        axs[0].plot(np.array(pcgcv1_results["bpp"][:]), np.array(pcgcv1_results["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv1", marker='x', color='green')
        axs[1].plot(np.array(pcgcv1_results["bpp"][:]), np.array(pcgcv1_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv1", marker='x', color='green')

    if args.outdir_pcgcv2 != '': 
        csv_pcgcv2= os.path.join(workspace_path, args.outdir_pcgcv2, args.resolution, basename +'.csv')
        pcgcv2_results = pd.read_csv(csv_pcgcv2)
        axs[0].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv2", marker='x', color='blue')
        axs[1].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv2", marker='x', color='blue')

    if args.outdir_pcgcv2_f != '':         
        csv_pcgcv2_f= os.path.join(workspace_path, args.outdir_pcgcv2_f, args.resolution, basename +'.csv')
        pcgcv2_results_f = pd.read_csv(csv_pcgcv2_f)
        axs[0].plot(np.array(pcgcv2_results_f["bpp"][:]), np.array(pcgcv2_results_f["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv2_f", marker='x', color='orange')
        axs[1].plot(np.array(pcgcv2_results_f["bpp"][:]), np.array(pcgcv2_results_f["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv2_f", marker='x', color='orange')
    
    if args.outdir_geov1 != '':  
        csv_geov1= os.path.join(workspace_path, args.outdir_geov1, args.resolution, basename +'.csv')
        geov1_results = pd.read_csv(csv_geov1)
        axs[0].plot(np.array(geov1_results["bpp"][:]), np.array(geov1_results["mseF,PSNR (p2point)"][:]), 
                    label="PCC_GEO_CNNv1", marker='x', color='orange')
        axs[1].plot(np.array(geov1_results["bpp"][:]), np.array(geov1_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCC_GEO_CNNv1", marker='x', color='orange')
        
    if args.outdir_geov2 != '':
        csv_geov2= os.path.join(workspace_path, args.outdir_geov2, args.resolution, basename +'.csv')
        geov2_results = pd.read_csv(csv_geov2)
        axs[0].plot(np.array(geov2_results["bpp"][:]), np.array(geov2_results["mseF,PSNR (p2point)"][:]), 
                    label="PCC_GEO_CNNv2", marker='x', color='purple')
        axs[1].plot(np.array(geov2_results["bpp"][:]), np.array(geov2_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCC_GEO_CNNv2", marker='x', color='purple')

    axs[0].set_xlabel('bpp')
    axs[0].set_ylabel('D1 PSNR (dB)')
    axs[0].grid(ls='-.')
    axs[0].legend(loc='lower right')

    axs[1].set_xlabel('bpp')
    axs[1].set_ylabel('D2 PSNR (dB)')
    axs[1].grid(ls='-.')
    axs[1].legend(loc='lower right')
    
    fig.savefig(os.path.join(basename+'.jpg'))
    print(os.path.join(basename+'.jpg'))
    
    
    
    
    
    
    
    
    

#     # D2
#     fig, ax = plt.subplots(figsize=(7, 4))
#     plt.plot(np.array(gpcc_results["bpp"][:]), np.array(gpcc_results["mseF,PSNR (p2plane)"][:]), 
#             label="G-PCC", marker='x', color='red')
#     plt.plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2plane)"][:]), 
#             label="PCGCv2", marker='x', color='blue')
#     plt.plot(np.array(pcgcv2_results_f["bpp"][:]), np.array(pcgcv2_results_f["mseF,PSNR (p2plane)"][:]), 
#             label="PCGCv2_f", marker='x', color='orange')
#     plt.plot(np.array(geov2_results["bpp"][:]), np.array(geov2_results["mseF,PSNR (p2plane)"][:]), 
#             label="PCC_GEO_CNNv2", marker='x', color='purple')    
# #     plt.plot(np.array(geov1_results["bpp"][:]), np.array(geov1_results["mseF,PSNR (p2plane)"][:]), 
# #             label="PCC_GEO_CNNv1", marker='x', color='orange')
    
#     plt.plot(np.array(pcgcv1_results["bpp"][:]), np.array(pcgcv1_results["mseF,PSNR (p2plane)"][:]), 
#             label="PCGCv1", marker='x', color='green')
#     plt.title(basename)
#     plt.xlabel('bpp')
#     plt.ylabel('D2 PSNR (dB)')
#     plt.grid(ls='-.')
#     plt.legend(loc='lower right')
    
#     fig.savefig(os.path.join(basename+'_D2.jpg'))



