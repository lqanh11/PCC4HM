import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import bjontegaard as bd

plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('text', usetex = True)

workspace_path = '/media/avitech/QuocAnh_1TB/Point_Cloud/'
workspace_path_13 = '/media/avitech/QuocAnh_1TB/Point_Cloud/source_code'

color_list = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#008080', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
              ]

markersize=4

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--list_txt", 
                        default="scalable_frameworks/PCGC/evaluation/ModelNet_test_list.txt")
    
    parser.add_argument("--outdir_scalable_2_tasks", 
                        # default="")
                        # default="scalable_frameworks/PCGC/evaluation/output/ModelNet/scalable_two_tasks")
                        default="scalable_frameworks/PCGC/evaluation/output/ModelNet/scalable_full")

    parser.add_argument("--outdir_pcgcv2", 
                        default="compression_frameworks/PCGC/PCGCv2/evaluation/ModelNet/pretrained")
                        # default="")
    
    parser.add_argument("--outdir_pcgcv2_f", 
                        default="source_code/compression_frameworks/PCGCv2/evaluation/output/ModelNet/finetuning")
                        
    
    parser.add_argument("--outdir_gpcc", 
                        default="compression_frameworks/G-PCC/evaluation/output/ModelNet/lossy")
    
    parser.add_argument("--outdir_geov1", 
                        default="")
    
    parser.add_argument("--outdir_geov2", 
                        default="compression_frameworks/GEO_CNN/v2/evaluation/output/ModelNet/pretrained")
    
    parser.add_argument("--outdir_pcgcv1", 
                        default="")
    
    parser.add_argument("--resolution",
                        default="128")
    
    parser.add_argument("--metrics",
                        default="mseF,PSNR (p2plane)")

    args = parser.parse_args()

    with open(os.path.join(workspace_path_13, args.list_txt), 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            print(line.strip())
            args.filedir = line.strip()

            basename = os.path.split(args.filedir)[-1].split('.')[0]
            
            ## Rate-distortion efficiency
            # D1
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            if args.outdir_gpcc != '': 
                csv_gpcc= os.path.join(workspace_path, args.outdir_gpcc, args.resolution, basename +'.csv')
                results = pd.read_csv(csv_gpcc)
                if idx == 0:
                    gpcc_results = results
                else:
                    gpcc_results += results
                if idx == len(lines)-1:
                    gpcc_results = gpcc_results / len(lines)

                    print(gpcc_results)
                

            if args.outdir_pcgcv1 != '': 
                csv_pcgcv1= os.path.join(workspace_path, args.outdir_pcgcv1, args.resolution, basename +'.csv')
                results = pd.read_csv(csv_pcgcv1)
                if idx == 0:
                    pcgcv1_results = results
                else:
                    pcgcv1_results += results
                if idx == len(lines)-1:
                    pcgcv1_results = pcgcv1_results / len(lines)
                
            if args.outdir_pcgcv2 != '': 
                csv_pcgcv2= os.path.join(workspace_path, args.outdir_pcgcv2, args.resolution, basename +'.csv')
                results = pd.read_csv(csv_pcgcv2)
                if idx == 0:
                    pcgcv2_results = results
                else:
                    pcgcv2_results += results
                if idx == len(lines)-1:
                    pcgcv2_results = pcgcv2_results / len(lines)
                
            if args.outdir_pcgcv2_f != '':         
                csv_pcgcv2_f= os.path.join(workspace_path, args.outdir_pcgcv2_f, args.resolution, basename +'.csv')
                results = pd.read_csv(csv_pcgcv2_f)
                results['bpp'] = results['bits'] / results['num_points(input)']
                if idx == 0:
                    pcgcv2_results_f = results
                else:
                    pcgcv2_results_f += results
                if idx == len(lines)-1:
                    pcgcv2_results_f = pcgcv2_results_f / len(lines)
                
            if args.outdir_geov1 != '':  
                csv_geov1= os.path.join(workspace_path, args.outdir_geov1, args.resolution, basename +'.csv')
                results = pd.read_csv(csv_geov1)
                if idx == 0:
                    geov1_results = results
                else:
                    geov1_results += results
                if idx == len(lines)-1:
                    geov1_results = geov1_results / len(lines)
                
            if args.outdir_geov2 != '':
                csv_geov2= os.path.join(workspace_path, args.outdir_geov2, args.resolution, basename +'.csv')
                results = pd.read_csv(csv_geov2)
                if idx == 0:
                    geov2_results = results
                else:
                    geov2_results += results
                if idx == len(lines)-1:
                    geov2_results = geov2_results / len(lines)
            
            if args.outdir_scalable_2_tasks != '':
                csv_file= os.path.join(workspace_path_13, args.outdir_scalable_2_tasks, args.resolution, basename +'.csv')
                if os.path.exists(csv_file):
                    results = pd.read_csv(csv_file)
                    if idx == 0:
                        scalable_results = results
                    else:
                        scalable_results += results
                    if idx == len(lines)-1:
                        scalable_results = scalable_results / len(lines)
                    
                
            ########################################
            ########################################
            ########################################
            ########################################
                
    if args.outdir_gpcc != '': 
        
        axs[0].plot(np.array(gpcc_results["bpp"][:]), np.array(gpcc_results["mseF,PSNR (p2point)"][:]), 
                    label="G-PCC", marker='x', color=color_list[0], markersize=markersize)
        axs[1].plot(np.array(gpcc_results["bpp"][:]), np.array(gpcc_results["mseF,PSNR (p2plane)"][:]), 
                    label="G-PCC", marker='x', color=color_list[0], markersize=markersize)
        bd_rate_gpcc = bd.bd_rate(
            rate_anchor=np.array(gpcc_results["bpp"][:]),
            dist_anchor=np.array(gpcc_results[args.metrics][:]),
            rate_test=np.array(scalable_results["bpp"][:]),
            dist_test=np.array(scalable_results[args.metrics][:]),
            method="akima",
            require_matching_points=False,
        )
        print(bd_rate_gpcc)

    if args.outdir_geov1 != '':  
        
        axs[0].plot(np.array(geov1_results["bpp"][:]), np.array(geov1_results["mseF,PSNR (p2point)"][:]), 
                    label="GEO_CNNv1", marker='x', color=color_list[1], markersize=markersize)
        axs[1].plot(np.array(geov1_results["bpp"][:]), np.array(geov1_results["mseF,PSNR (p2plane)"][:]), 
                    label="GEO_CNNv1", marker='x', color=color_list[1], markersize=markersize)
       
    if args.outdir_geov2 != '':
        
        axs[0].plot(np.array(geov2_results["bpp"][:]), np.array(geov2_results["mseF,PSNR (p2point)"][:]), 
                    label="GEO_CNNv2", marker='x', color=color_list[2], markersize=markersize)
        axs[1].plot(np.array(geov2_results["bpp"][:]), np.array(geov2_results["mseF,PSNR (p2plane)"][:]), 
                    label="GEO_CNNv2", marker='x', color=color_list[2], markersize=markersize)

        bd_rate_geocnnv2 = bd.bd_rate(
            rate_anchor=np.array(geov2_results["bpp"][:]),
            dist_anchor=np.array(geov2_results[args.metrics][:]),
            rate_test=np.array(scalable_results["bpp"][:]),
            dist_test=np.array(scalable_results[args.metrics][:]),
            method="akima",
            require_matching_points=False,
        )
        print(bd_rate_geocnnv2)

    if args.outdir_pcgcv1 != '': 
        
        axs[0].plot(np.array(pcgcv1_results["bpp"][:]), np.array(pcgcv1_results["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv1", marker='x', color=color_list[3], markersize=markersize)
        axs[1].plot(np.array(pcgcv1_results["bpp"][:]), np.array(pcgcv1_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv1", marker='x', color=color_list[3], markersize=markersize)

    if args.outdir_pcgcv2 != '': 
        
        axs[0].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv2", marker='x', color=color_list[4], markersize=markersize)
        axs[1].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv2", marker='x', color=color_list[4], markersize=markersize)
        
        # x_ref = np.sort(np.array(x_ref), axis=0)
        # y_ref = np.sort( np.array(y_ref), axis=0)
        # x_curr = np.sort( np.array(x_curr), axis=0)
        # y_curr = np.sort( np.array(y_curr),axis=0)

        bd_rate_pcgcv2 = bd.bd_rate(
            rate_anchor=np.sort(np.array(pcgcv2_results["bpp"][:])),
            dist_anchor=np.sort(np.array(pcgcv2_results[args.metrics][:])),
            rate_test=np.array(scalable_results["bpp"][:]),
            dist_test=np.array(scalable_results[args.metrics][:]),
            method="akima",
            require_matching_points=False,
        )
        print(bd_rate_pcgcv2)

    if args.outdir_pcgcv2_f != '':         
        
        axs[0].plot(np.array(pcgcv2_results_f["bpp"][:]), np.array(pcgcv2_results_f["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv2 (finetuning)", marker='x', color=color_list[5], markersize=markersize)
        axs[1].plot(np.array(pcgcv2_results_f["bpp"][:]), np.array(pcgcv2_results_f["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv2 (finetuning)", marker='x', color=color_list[5], markersize=markersize)
        
        bd_rate_pcgcv2_f = bd.bd_rate(
            rate_anchor=np.array(pcgcv2_results_f["bpp"][:]),
            dist_anchor=np.array(pcgcv2_results_f[args.metrics][:]),
            rate_test=np.array(scalable_results["bpp"][:]),
            dist_test=np.array(scalable_results[args.metrics][:]),
            method="akima",
            require_matching_points=False,
        )
        print(bd_rate_pcgcv2_f)


    if args.outdir_scalable_2_tasks != '':
        try:
            axs[0].plot(np.array(scalable_results["bpp"][:]), np.array(scalable_results["mseF,PSNR (p2point)"][:]), 
                        label="Proposed Codec", marker='x', color=color_list[6], markersize=markersize)
            axs[1].plot(np.array(scalable_results["bpp"][:]), np.array(scalable_results["mseF,PSNR (p2plane)"][:]), 
                        label="Proposed Codec", marker='x', color=color_list[6], markersize=markersize)
        
        except NameError:
            pass    

    axs[0].title.set_text(f'ModelNet10 (r = {args.resolution})') 
    axs[0].set_xlabel('bpp')
    axs[0].set_ylabel('D1 PSNR (dB)')
    axs[0].grid(ls='-.')
    axs[0].legend(loc='lower right')

    axs[1].title.set_text(f'ModelNet10 (r = {args.resolution})') 
    axs[1].set_xlabel('bpp')
    axs[1].set_ylabel('D2 PSNR (dB)')
    axs[1].grid(ls='-.')
    axs[1].legend(loc='lower right')

    save_name = f'mean_all_{args.resolution}'

    save_path = os.path.join('./output/Scalable/', 'JPG')
    os.makedirs(save_path, exist_ok=True)
    
    fig.savefig(os.path.join(save_path, save_name + '.jpg'), dpi=600)
    print(os.path.join(save_path, save_name + '.jpg'))
    
    save_path = os.path.join('./output/Scalable/', 'PDF')
    os.makedirs(save_path, exist_ok=True)
    
    fig.savefig(os.path.join(save_path, save_name + '.pdf'), dpi=600)
    print(os.path.join(save_path, save_name + '.pdf'))

    encode_times = []
    decode_times = []

    encode_times.append(gpcc_results["time(enc)"][:].mean())
    decode_times.append(gpcc_results["time(dec)"][:].mean())

    encode_times.append(geov2_results["time(enc)"][:].mean())
    decode_times.append(geov2_results["time(dec)"][:].mean())

    encode_times.append(pcgcv2_results["time(enc)"][:].mean())
    decode_times.append(pcgcv2_results["time(dec)"][:].mean())

    encode_times.append(pcgcv2_results_f["time(enc)"][:].mean())
    decode_times.append(pcgcv2_results_f["time(dec)"][:].mean())

    encode_times.append(scalable_results["time(enc)"][:].mean())
    decode_times.append(scalable_results["time(dec)"][:].mean())

    print(encode_times)
    print(decode_times)
            
    
    
    
    
    
    
    
    

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



