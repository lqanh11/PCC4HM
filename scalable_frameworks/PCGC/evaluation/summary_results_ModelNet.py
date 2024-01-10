import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/media/avitech-pc2/Student/quocanhle/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0512.h5')
    parser.add_argument("--outdir", default='./output/ModelNet/scalable_two_tasks')
    parser.add_argument("--res", type=int, default=64, help='resolution')
    parser.add_argument('--ckptdir_list',
                    type=str, nargs='*', default=[
                                                    '/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r1_0.025bpp.pth',
                                                    '/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r2_0.05bpp.pth',
                                                    '/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r3_0.10bpp.pth',
                                                    '/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r4_0.15bpp.pth',
                                                    '/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r5_0.25bpp.pth',
                                                    '/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r6_0.3bpp.pth',
                                                    '/media/avitech-pc2/Student/quocanhle/Point_Cloud/compression_frameworks/pretrained_models/PCGCv2/ckpts/r7_0.4bpp.pth'
                                                    ], help="CKPT list")
    
    args = parser.parse_args()

    output_resolution = os.path.join(args.outdir, f'{args.res}')
    save_original_path = os.path.join(output_resolution, 'original')

    pc_filedir = os.path.join(save_original_path, os.path.split(args.filedir)[-1].split('.')[0] + '.ply')

    basename = os.path.split(pc_filedir)[-1].split('.')[0]

    for idx, ckptdir in enumerate(args.ckptdir_list):
        
        # postfix: rate index
        postfix_idx = 'r'+str(idx+1)
        postfix_file = ''
        # initialize output filename
        save_results_path = os.path.join(output_resolution, postfix_idx)

        basename = os.path.split(pc_filedir)[-1].split('.')[0]
        filename = os.path.join(save_results_path, basename)
        
        ## Coding results
        csv_coding_name = os.path.join(save_results_path, basename +'.csv')
        coding_results = pd.read_csv(csv_coding_name)
        
        ## Distortion results
        csv_distortion_name = os.path.join(save_results_path, basename +'_distortion.csv')
        distortion_results = pd.read_csv(csv_distortion_name)

        # print(csv_coding_name, csv_distortion_name)

        results = pd.concat([coding_results, distortion_results], axis=1)

        if idx == 0:
            all_results = results
        else:
            all_results = pd.concat([all_results, results], axis=0)

    
    csv_all_results = os.path.join(output_resolution, basename +'.csv')
    all_results.to_csv(csv_all_results, index=False)
    print('Wrile results to: \t', csv_all_results)
    

    # plot RD-curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color='blue')

    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(output_resolution, basename+'.jpg'))
    print(os.path.join(output_resolution, basename+'.jpg'))