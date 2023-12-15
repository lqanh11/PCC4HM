import sys
sys.path.append('../')

import os, logging, argparse
import sklearn.metrics as metrics
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from pointnet import (
    PointNet,
    MinkowskiPointNet,
    CoordinateTransformation,
    ModelNetH5_voxelize_all,
    stack_collate_fn,
    minkowski_collate_fn,
)

from classification_model import (
    MinkowskiFCNN, 
    MinkowskiSplatFCNN, 
    MinkoPointNet_Conv, 
    MinkoPointNet_Conv_2, 
    )

from common import seed_all

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/")
parser.add_argument("--num_points", type=int, default=1024)
parser.add_argument("--resolution", type=float, default=128)
parser.add_argument(
    "--network",
    type=str,
    choices=["pointnet", 
             "minkpointnet", 
             "minkfcnn", 
             "minksplatfcnn", 
             "minkpointnet_conv", 
             'minkpointnet_conv_2'],
    default="minkpointnet",
)

parser.add_argument("--voxel_size", type=float, default=1)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--weights", type=str, default="modelnet")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--test_translation", type=float, default=0.0)


STR2NETWORK = dict(
    pointnet=PointNet,
    minkpointnet=MinkowskiPointNet,
    minkfcnn=MinkowskiFCNN,
    minksplatfcnn=MinkowskiSplatFCNN,
    minkpointnet_conv=MinkoPointNet_Conv,
    minkpointnet_conv_2=MinkoPointNet_Conv_2,
)

def create_input_batch(batch, is_minknet, device="cuda", quantization_size=0.05):
    if is_minknet:
        batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
        return ME.TensorField(
            coordinates=batch["coordinates"],
            features=batch["features"],
            device=device,
        )
    else:
        return batch["coordinates"].permute(0, 2, 1).to(device)


class CoordinateTranslation:
    def __init__(self, translation):
        self.trans = translation

    def __call__(self, coords):
        if self.trans > 0:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        return coords


def make_data_loader(phase, is_minknet, config):
    assert phase in ["train", "val", "test"]
    is_train = phase == "train"


    dataset = ModelNetH5_voxelize_all(
        phase=phase,
        transform=CoordinateTransformation(trans=config.translation)
        if is_train
        else CoordinateTranslation(config.test_translation),
        data_root=config.data_root,
        num_points=config.num_points,
        resolution=config.resolution
    )
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        shuffle=is_train,
        collate_fn=minkowski_collate_fn if is_minknet else stack_collate_fn,
        batch_size=config.batch_size,
    )


def test(net, device, config, logger, phase="val"):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    data_loader = make_data_loader(
        "test",
        is_minknet,
        config=config,
    )

    net.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            input = create_input_batch(
                batch,
                is_minknet,
                device=device,
                quantization_size=config.voxel_size,
            )
            logit = net(input)
            pred = torch.argmax(logit, 1)
            labels.append(batch["labels"].cpu().numpy())
            preds.append(pred.cpu().numpy())
            torch.cuda.empty_cache()
    return metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds))

def getlogger(logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO) #NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S') 
        handler.setFormatter(formatter)
        console = logging.StreamHandler() 
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger

if __name__ == "__main__":

    config = parser.parse_args()
    logdir = os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/Mink_classification/', 
                          'classification_modelnet10_voxelize', str(config.num_points),'eval_' + config.network + f'_voxelized_{config.resolution}')
    os.makedirs(logdir, exist_ok=True)
    logger = getlogger(logdir)

    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = STR2NETWORK[config.network](
        in_channel=3, out_channel=10, embedding_channel=1024
    ).to(device)

    ckpt_path = os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/Mink_classification/', 
                          'classification_modelnet10_voxelize', str(config.num_points),config.network + f'_voxelized_{config.resolution}',
                          config.weights + "_" + config.network + ".pth")
    logger.info('Load checkpoint from ' + ckpt_path)
    
    ckpt_state_dict = torch.load(ckpt_path)
    net.load_state_dict(ckpt_state_dict['state_dict'])
   
    accuracy = test(net, device, config, logger, phase="test")
    logger.info(f"Test accuracy: {accuracy}")