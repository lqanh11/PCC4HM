import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

workspace_path = '/media/avitech/Data/quocanhle/PointCloud/'
workspace_path_13 = '/media/avitech/Data/quocanhle/PointCloud/source_code/'



color_list = [
    '#e03524',
    '#f07c12',
    '#ffc200',
    '#90bc1a',
    '#21b534',
    '#0095ac',
    '#1f64ad',
    '#4040a0',
    '#903498',
              ]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--list_txt", 
                        default="scalable_frameworks/PCGC/evaluation/ModelNet_test_list.txt")

    parser.add_argument("--outdir_pcgcv2", 
                        default="compression_frameworks/PCGC/PCGCv2/evaluation/ModelNet/pretrained")
    
    parser.add_argument("--resolution",
                        default="64")

    args = parser.parse_args()

    number_of_points = []


    with open(os.path.join(workspace_path_13, args.list_txt), 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            print(line.strip())
            args.filedir = line.strip()

            basename = os.path.split(args.filedir)[-1].split('.')[0]
            
                
            if args.outdir_pcgcv2 != '': 
                csv_pcgcv2= os.path.join(workspace_path, args.outdir_pcgcv2, args.resolution, basename +'.csv')
                results = pd.read_csv(csv_pcgcv2)

                number_of_points.append(results["num_points(input)"][:].mean())

    print("Max: ", np.max(number_of_points))
    print("Min: ", np.min(number_of_points))
    print("Avg: ", np.mean(number_of_points))
    print("STD: ", np.std(number_of_points))

                
           
                
            ########################################
            ########################################
            ########################################
            ########################################
                
    
    # if args.outdir_pcgcv2 != '': 
        
    #     axs[0].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2point)"][:]), 
    #                 label="PCGCv2", marker='x', color=color_list[4])
    #     axs[1].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2plane)"][:]), 
    #                 label="PCGCv2", marker='x', color=color_list[4])

    
    
    
    
    
    

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



