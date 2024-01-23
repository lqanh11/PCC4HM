import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from BD_Rates.AIPCCReportingTemplate.test import compare

workspace_path = '/media/avitech7/QuocAnh_1TB/Point_Cloud/'
xlabel='bpp'
ylabel1='mseF,PSNR (p2point)'
ylabel2="mseF,PSNR (p2plane)"

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

    save_TXT = open('BD_rates.txt', 'a+')

    BD_rates_D1 = np.zeros(5)
    BD_rates_D2 = np.zeros(5)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    

    if args.outdir_gpcc != '': 
        csv_gpcc= os.path.join(workspace_path, args.outdir_gpcc, basename +'.csv')
        gpcc_results = pd.read_csv(csv_gpcc)
       
        labels.append("G-PCC")
        encode_times.append(gpcc_results["time(enc)"][:].mean())
        decode_times.append(gpcc_results["time(dec)"][:].mean())
    
    if args.outdir_geov1 != '':  
        csv_geov1= os.path.join(workspace_path, args.outdir_geov1, basename +'.csv')
        if os.path.exists(csv_geov1):
            geov1_results = pd.read_csv(csv_geov1)

            d1_bd = compare(gpcc_results, geov1_results, xlabel, ylabel1, basename)
            BD_rates_D1[0] = d1_bd
            d2_bd = compare(gpcc_results, geov1_results, xlabel, ylabel2, basename)
            BD_rates_D2[0] = d2_bd
            
            labels.append("GEO_CNNv1")
            encode_times.append(geov1_results["time(enc)"][:].mean())
            decode_times.append(geov1_results["time(dec)"][:].mean())
        
    if args.outdir_geov2 != '':
        csv_geov2= os.path.join(workspace_path, args.outdir_geov2, basename +'.csv')
        geov2_results = pd.read_csv(csv_geov2)

        d1_bd = compare(gpcc_results, geov2_results, xlabel, ylabel1, basename)
        BD_rates_D1[1] = d1_bd
        d2_bd = compare(gpcc_results, geov2_results, xlabel, ylabel2, basename)
        BD_rates_D2[1] = d2_bd
        
        labels.append("GEO_CNNv2")
        encode_times.append(geov2_results["time(enc)"][:].mean())
        decode_times.append(geov2_results["time(dec)"][:].mean())

    if args.outdir_pcgcv1 != '': 
        csv_pcgcv1= os.path.join(workspace_path, args.outdir_pcgcv1, basename +'.csv')
        pcgcv1_results = pd.read_csv(csv_pcgcv1)

        d1_bd = compare(gpcc_results, pcgcv1_results, xlabel, ylabel1, basename)
        BD_rates_D1[2] = d1_bd
        d2_bd = compare(gpcc_results, pcgcv1_results, xlabel, ylabel2, basename)
        BD_rates_D2[2] = d2_bd
        
        labels.append("PCGCv1")
        encode_times.append(pcgcv1_results["time(enc)"][:].mean())
        decode_times.append(pcgcv1_results["time(dec)"][:].mean())

        

    if args.outdir_pcgcv2 != '': 
        csv_pcgcv2= os.path.join(workspace_path, args.outdir_pcgcv2, basename +'.csv')
        pcgcv2_results = pd.read_csv(csv_pcgcv2)
        

        d1_bd = compare(gpcc_results, pcgcv2_results, xlabel, ylabel1, basename)
        BD_rates_D1[3] = d1_bd
        d2_bd = compare(gpcc_results, pcgcv2_results, xlabel, ylabel2, basename)
        BD_rates_D2[3] = d2_bd

        labels.append("PCGCv2")
        encode_times.append(pcgcv2_results["time(enc)"][:].mean())
        decode_times.append(pcgcv2_results["time(dec)"][:].mean())

    if args.outdir_pcgcv2_f != '':         
        csv_pcgcv2_f= os.path.join(workspace_path, args.outdir_pcgcv2_f, basename +'.csv')
        pcgcv2_results_f = pd.read_csv(csv_pcgcv2_f)

        d1_bd = compare(gpcc_results, pcgcv2_results_f, xlabel, ylabel1, basename)
        BD_rates_D1[4] = d1_bd
        d2_bd = compare(gpcc_results, pcgcv2_results_f, xlabel, ylabel2, basename)
        BD_rates_D2[4] = d2_bd
        
        labels.append("PCGCv2 (finetuning)")
        encode_times.append(pcgcv2_results_f["time(enc)"][:].mean())
        decode_times.append(pcgcv2_results_f["time(dec)"][:].mean())

    print(basename)
    print(BD_rates_D1)
    print(BD_rates_D2)

    save_TXT.write(f'{basename}\t{BD_rates_D1[0]}\t{BD_rates_D1[1]}\t{BD_rates_D1[2]}\t{BD_rates_D1[3]}')
    save_TXT.write(f'\t{BD_rates_D2[0]}\t{BD_rates_D2[1]}\t{BD_rates_D2[2]}\t{BD_rates_D2[3]}\n')
    save_TXT.close()