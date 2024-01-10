import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# font = {'family' : 'sans-serif',
#         'weight' : 'bold',
#         'size'   : 11}

# plt.rc('font', **font)

plt.rcParams["font.family"] = "Times New Roman"

workspace_path = '/media/avitech/Data/quocanhle/PointCloud/'
color_list = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#008080', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
              ]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", 
                        default='/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/testdata/8iVFB/longdress_vox10_1300.ply')

    parser.add_argument("--outdir_pcgcv2", 
                        default="compression_frameworks/PCGC/PCGCv2/evaluation/MPEG/pretrained")
    
    parser.add_argument("--outdir_pcgcv2_f", 
                        default="")
                        # default="compression_frameworks/PCGCv2/evaluation/output/MPEG/finetuning")
    
    parser.add_argument("--outdir_gpcc", 
                        default="compression_frameworks/G-PCC/evaluation/output/MPEG/lossy")
    
    parser.add_argument("--outdir_geov1", 
                        # default="")
                        default="compression_frameworks/GEO_CNN/v1/evaluation/output/MPEG/pretrained")
    
    parser.add_argument("--outdir_geov2", 
                        default="compression_frameworks/GEO_CNN/v2/evaluation/output/MPEG/pretrained")
    
    parser.add_argument("--outdir_pcgcv1", 
                        default="compression_frameworks/PCGC/PCGCv1/evaluation/output/MPEG/pretrained")
    
    parser.add_argument("--resolution",
                        default="512")

    args = parser.parse_args()

    basename = os.path.split(args.filedir)[-1].split('.')[0]
    
    ## encode decode time
    labels = []
    encode_times = []
    decode_times = []

    ## Rate-distortion efficiency
    # D1
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    

    if args.outdir_gpcc != '': 
        csv_gpcc= os.path.join(workspace_path, args.outdir_gpcc, basename +'.csv')
        gpcc_results = pd.read_csv(csv_gpcc)
        axs[0].plot(np.array(gpcc_results["bpp"][:]), np.array(gpcc_results["mseF,PSNR (p2point)"][:]), 
                    label="G-PCC", marker='x', color=color_list[0])
        axs[1].plot(np.array(gpcc_results["bpp"][:]), np.array(gpcc_results["mseF,PSNR (p2plane)"][:]), 
                    label="G-PCC", marker='x', color=color_list[0])
        
        labels.append("G-PCC")
        encode_times.append(gpcc_results["time(enc)"][:].mean())
        decode_times.append(gpcc_results["time(dec)"][:].mean())
    
    if args.outdir_geov1 != '':  
        csv_geov1= os.path.join(workspace_path, args.outdir_geov1, basename +'.csv')
        if os.path.exists(csv_geov1):
            geov1_results = pd.read_csv(csv_geov1)
            axs[0].plot(np.array(geov1_results["bpp"][:3]), np.array(geov1_results["mseF,PSNR (p2point)"][:3]), 
                        label="GEO_CNNv1", marker='x', color=color_list[1])
            axs[1].plot(np.array(geov1_results["bpp"][:3]), np.array(geov1_results["mseF,PSNR (p2plane)"][:3]), 
                        label="GEO_CNNv1", marker='x', color=color_list[1])
            
            labels.append("GEO_CNNv1")
            encode_times.append(geov1_results["time(enc)"][:].mean())
            decode_times.append(geov1_results["time(dec)"][:].mean())
        
    if args.outdir_geov2 != '':
        csv_geov2= os.path.join(workspace_path, args.outdir_geov2, basename +'.csv')
        geov2_results = pd.read_csv(csv_geov2)
        axs[0].plot(np.array(geov2_results["bpp"][:]), np.array(geov2_results["mseF,PSNR (p2point)"][:]), 
                    label="GEO_CNNv2", marker='x', color=color_list[2])
        axs[1].plot(np.array(geov2_results["bpp"][:]), np.array(geov2_results["mseF,PSNR (p2plane)"][:]), 
                    label="GEO_CNNv2", marker='x', color=color_list[2])
        
        labels.append("GEO_CNNv2")
        encode_times.append(geov2_results["time(enc)"][:].mean())
        decode_times.append(geov2_results["time(dec)"][:].mean())

    if args.outdir_pcgcv1 != '': 
        csv_pcgcv1= os.path.join(workspace_path, args.outdir_pcgcv1, basename +'.csv')
        pcgcv1_results = pd.read_csv(csv_pcgcv1)
        axs[0].plot(np.array(pcgcv1_results["bpp"][:]), np.array(pcgcv1_results["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv1", marker='x', color=color_list[3])
        axs[1].plot(np.array(pcgcv1_results["bpp"][:]), np.array(pcgcv1_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv1", marker='x', color=color_list[3])
        
        labels.append("PCGCv1")
        encode_times.append(pcgcv1_results["time(enc)"][:].mean())
        decode_times.append(pcgcv1_results["time(dec)"][:].mean())

    if args.outdir_pcgcv2 != '': 
        csv_pcgcv2= os.path.join(workspace_path, args.outdir_pcgcv2, basename +'.csv')
        pcgcv2_results = pd.read_csv(csv_pcgcv2)
        axs[0].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv2", marker='x', color=color_list[4])
        axs[1].plot(np.array(pcgcv2_results["bpp"][:]), np.array(pcgcv2_results["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv2", marker='x', color=color_list[4])

        labels.append("PCGCv2")
        encode_times.append(pcgcv2_results["time(enc)"][:].mean())
        decode_times.append(pcgcv2_results["time(dec)"][:].mean())

    if args.outdir_pcgcv2_f != '':         
        csv_pcgcv2_f= os.path.join(workspace_path, args.outdir_pcgcv2_f, basename +'.csv')
        pcgcv2_results_f = pd.read_csv(csv_pcgcv2_f)
        axs[0].plot(np.array(pcgcv2_results_f["bpp"][:]), np.array(pcgcv2_results_f["mseF,PSNR (p2point)"][:]), 
                    label="PCGCv2_f", marker='x', color=color_list[5])
        axs[1].plot(np.array(pcgcv2_results_f["bpp"][:]), np.array(pcgcv2_results_f["mseF,PSNR (p2plane)"][:]), 
                    label="PCGCv2_f", marker='x', color=color_list[5])
        
        labels.append("PCGCv2 (finetuning)")
        encode_times.append(pcgcv2_results_f["time(enc)"][:].mean())
        decode_times.append(pcgcv2_results_f["time(dec)"][:].mean())

    axs[0].title.set_text(basename) 
    axs[0].set_xlabel('bpp')
    axs[0].set_ylabel('D1 PSNR (dB)')
    axs[0].grid(ls='-.')
    axs[0].legend(loc='lower right')

    axs[1].title.set_text(basename) 
    axs[1].set_xlabel('bpp')
    axs[1].set_ylabel('D2 PSNR (dB)')
    axs[1].grid(ls='-.')
    axs[1].legend(loc='lower right')
    
    # fig.savefig(os.path.join(basename+'.jpg'), dpi=600)
    # print(os.path.join(basename+'.jpg'))

    save_name = basename

    save_path = os.path.join('./output/MPEG/', 'JPG')
    os.makedirs(save_path, exist_ok=True)
    
    fig.savefig(os.path.join(save_path, save_name + '.jpg'), dpi=600)
    print(os.path.join(save_path, save_name + '.jpg'))
    
    save_path = os.path.join('./output/MPEG/', 'PDF')
    os.makedirs(save_path, exist_ok=True)
    
    fig.savefig(os.path.join(save_path, save_name + '.pdf'), dpi=600)
    print(os.path.join(save_path, save_name + '.pdf'))


    # # Set up the positions for the bars
    # x = np.arange(len(labels))
    # width = 0.35  # Width of the bars

    # encode_times_limited_list = np.clip(encode_times, 0, 5)
    # decode_times_limited_list = np.clip(decode_times, 0, 5)

    # fig, ax = plt.subplots()

    # rects1 = ax.bar(x - width/2, encode_times_limited_list, width, label='Encode Time')
    # rects2 = ax.bar(x + width/2, decode_times_limited_list, width, label='Decode Time')    

    # # Add labels, title, and legend
    # ax.set_xlabel('Compression Frameworks')
    # ax.set_ylabel('Time (s)')
    # ax.set_title('Encode and Decode Times')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()
    # # Display the bar chart
    # fig.savefig(os.path.join(basename+'_time.jpg'))
    # print(os.path.join(basename+'_time.jpg'))
    
    
    
    
    
    
    

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



