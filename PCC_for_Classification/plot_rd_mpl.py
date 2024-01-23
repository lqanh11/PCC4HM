import sys
import os
import re

sys.path.append('/media/avitech/QuocAnh_1TB/Point_Cloud/source_code/compression_frameworks/learned-point-cloud-compression-for-classification')


import bjontegaard as bd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from compressai_trainer.utils.compressai.results import compressai_results_dataframe

sns.set_theme(
    # context="paper",
    style="whitegrid",
    font="Times New Roman",
    font_scale=1,
)

EPSILON = 1e-8

DATASET = "ModelNet10"
x = "bits"
y = "acc_top1"

resolution = 256
REF_CODEC_NAME = "G-PCC + PointNet++ [P=1024]"
CODECS = [
    
    f"MMSP2023_methods_res{resolution}.json",
    f"GPCC_PointNetPP_res{resolution}.json",
    f"Proposed_Cocdec_r1_res{resolution}.json",
    f"Proposed_Cocdec_r2_res{resolution}.json",
    f"Proposed_Cocdec_r3_res{resolution}.json",
    f"Proposed_Cocdec_r4_res{resolution}.json",
    f"Proposed_Cocdec_r5_res{resolution}.json",
    f"Proposed_Cocdec_r6_res{resolution}.json",
    f"Proposed_Cocdec_r7_res{resolution}.json",
]

COLORS = {
    "pointnetpp": "#4f4d4d",
    "pointnet": "#6e6a6a",
    "minkpointnet": "#969292",
}

MINKPOINTNET_RESULTS = {
    '64':  0.9107930  * 100,
    '128': 0.9174008 * 100,
    '256': 0.9096916 * 100
}
POINTNET_RESULTS = {
    '64': 0.927632 * 100,
    '128': 0.91557 * 100,
    '256': 0.926535 * 100
}
POINTNETPP_RESULTS = {
    '64': 0.934211 * 100,
    '128': 0.932018 * 100,
    '256': 0.930921 * 100
}

def remove_duplicates_and_get_indices(arr):
    unique_elements, indices = np.unique(arr, return_index=True)
    return unique_elements, indices

def preprocess_for_bd(x_ref, y_ref, x_curr, y_curr):
    x_ref = np.sort(np.array(x_ref), axis=0)
    y_ref = np.sort( np.array(y_ref), axis=0)
    x_curr = np.sort( np.array(x_curr), axis=0)
    y_curr = np.sort( np.array(y_curr),axis=0)

    y_ref, indices = remove_duplicates_and_get_indices(y_ref)
    x_ref = x_ref[indices]

    y_curr, indices = remove_duplicates_and_get_indices(y_curr)
    x_curr = x_curr[indices]

    # Ensure strictly monotonically increasing.
    for arr in [x_ref, x_curr, y_ref, y_curr]:
        # print(arr)
        assert (arr[:-1] < arr[1:]).all()

    x_min = min(x_ref[0], x_curr[0])
    x_max = max(x_ref[-1], x_curr[-1])

    # if x_ref[0] > x_min:
    #     x_ref = [x_min, *x_ref]
    #     y_ref = [INTERPOLATE_WITH_RD_ORIGIN, *y_ref]
    #
    # if x_curr[0] > x_min:
    #     x_curr = [x_min, *x_curr]
    #     y_curr = [INTERPOLATE_WITH_RD_ORIGIN, *y_curr]

    if "avoid_x_zero":
        x_ref[x_ref == 0] = x_ref[1] - EPSILON
        x_curr[x_curr == 0] = x_curr[1] - EPSILON

    if x_ref[-1] < x_max:
        x_ref = [*x_ref, x_max]
        y_ref = [*y_ref, y_ref[-1] + EPSILON]

    if x_curr[-1] < x_max:
        x_curr = [*x_curr, x_max]
        y_curr = [*y_curr, y_curr[-1] + EPSILON]

    return x_ref, y_ref, x_curr, y_curr

save_txt = open(f'BD_rate_cls_{resolution}.txt', 'w+')

def compute_stats(name, df_curr, df_ref):
    # codec_type = df_curr["codec_type"].unique()[0]
    # df_curr = df_curr[df_curr["bit_loss"] < BD_MAX_BITRATES[codec_type]["curr"]]
    # df_ref = df_ref[df_ref["bit_loss"] < BD_MAX_BITRATES[codec_type]["ref"]]

    x_ref, y_ref, x_curr, y_curr = preprocess_for_bd(
        df_ref[x], df_ref[y], df_curr[x], df_curr[y]
    )

    bd_rate = bd.bd_rate(
        rate_anchor=x_ref,
        dist_anchor=y_ref,
        rate_test=x_curr,
        dist_test=y_curr,
        method="akima",
        require_matching_points=False,
    )

    bd_dist = bd.bd_psnr(
        rate_anchor=x_ref,
        dist_anchor=y_ref,
        rate_test=x_curr,
        dist_test=y_curr,
        method="akima",
        require_matching_points=False,
    )

    max_y = df_curr[y].max()

    print(f"{name:<40} & {max_y*100:6.1f} & {bd_rate:6.1f} \\\\")

    save_txt.write(f"{name:<40}\t{max_y*100:6.1f}\t{bd_rate:6.1f}\n")


def plot_baseline(ax, resolution):
    _, x_max = ax.get_xlim()
    ax.plot(
        [0, x_max],
        [POINTNETPP_RESULTS[str(resolution)], POINTNETPP_RESULTS[str(resolution)]],
        label=f"PointNet++ [official] [{POINTNETPP_RESULTS[str(resolution)]:.{2}f}%]",
        color=COLORS["pointnetpp"],
        linestyle="--",
    )
    ax.plot(
        [0, x_max],
        [POINTNET_RESULTS[str(resolution)], POINTNET_RESULTS[str(resolution)]],
        label=f"PointNet [official] [{POINTNET_RESULTS[str(resolution)]:.{2}f}%]",
        color=COLORS["pointnet"],
        linestyle=":",
    )

    ax.plot(
        [0, x_max],
        [MINKPOINTNET_RESULTS[str(resolution)], MINKPOINTNET_RESULTS[str(resolution)]],
        label=f"MinkPointNet [{MINKPOINTNET_RESULTS[str(resolution)]:.{2}f}%]",
        color=COLORS["minkpointnet"],
        linestyle="-.",
    )
    

def read_dataframe():
    df = pd.concat(
        [
            compressai_results_dataframe(
                filename=name,
                base_path="./",
            )
            for name in CODECS
        ]
    )

    def name_to_codec_type(name):
        name_pattern = r"^Proposed codec \[(?P<codec_type>[\w-]+), P=(?P<points>\d+)\]$"
        m = re.match(name_pattern, name)
        if m:
            return m.group("codec_type")
        return "input-compression"

    df["codec_type"] = df["name"].apply(name_to_codec_type)

    return df

df = read_dataframe()
# print(df.to_string())

df_ref = df[df["name"] == REF_CODEC_NAME]

for name, df_curr in df.groupby("name", sort=False):
    compute_stats(name, df_curr, df_ref)


fig, ax = plt.subplots(figsize=(0.9 * 6.4, 1.0 * 4.8))

ax.set(
    xlabel="Rate (bits)",
    ylabel="Top-1 Accuracy (%)",
    xlim=[0, 2000],
    ylim=[0, 100],
)
palette = sns.color_palette("husl", 9)

plot_baseline(ax, resolution)

codec_type = "input-compression"

mask = (df["codec_type"] == codec_type) | (df["name"] == REF_CODEC_NAME)
mask |= (df["codec_type"] == "input-compression")
df_curr = df[mask]
df_curr["acc_top1"] = df_curr["acc_top1"] * 100

sns.lineplot(ax=ax, data=df_curr, x=x, y=y, markers=True, hue="name", palette=palette)
ax.legend().set_title(None)
ax.title.set_text(f'ModelNet10 (r = {resolution})') 
ax.grid(ls='-.')
ax.legend(loc='lower right')

fig.savefig(
    f"output.jpg",
    bbox_inches="tight",
    # pad_inches=0,
)
plt.show()

save_name = f'PCC_for_classification_{resolution}'

save_path = os.path.join('./output/Scalable/', 'JPG')
os.makedirs(save_path, exist_ok=True)

fig.savefig(os.path.join(save_path, save_name + '.jpg'), dpi=600)
print(os.path.join(save_path, save_name + '.jpg'))

save_path = os.path.join('./output/Scalable/', 'PDF')
os.makedirs(save_path, exist_ok=True)

fig.savefig(os.path.join(save_path, save_name + '.pdf'), dpi=600)
print(os.path.join(save_path, save_name + '.pdf'))

