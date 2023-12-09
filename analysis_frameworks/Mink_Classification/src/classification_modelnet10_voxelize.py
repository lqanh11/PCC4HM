# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import logging, os
import argparse
import sklearn.metrics as metrics
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import MinkowskiEngine as ME

import sys
sys.path.append('../')

from pointnet import (
    PointNet,
    MinkowskiPointNet,
    CoordinateTransformation,
    ModelNet40H5_voxelize,
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
parser.add_argument("--data_root", type=str, default="/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_modelnet10_ply_hdf5_1024/voxelized_128")
parser.add_argument("--num_points", type=int, default=1024)

parser.add_argument("--voxel_size", type=float, default=1)
parser.add_argument("--max_steps", type=int, default=50000)
parser.add_argument("--val_freq", type=int, default=1000)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=1e-1, type=float)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--stat_freq", type=int, default=100)
parser.add_argument("--weights", type=str, default="modelnet")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--translation", type=float, default=0.2)
parser.add_argument("--test_translation", type=float, default=0.0)
parser.add_argument(
    "--network",
    type=str,
    choices=["pointnet", 
             "minkpointnet", 
             "minkfcnn", 
             "minksplatfcnn", 
             "minkpointnet_conv", 
             'minkpointnet_conv_2'],
    default="minkfcnn",
)

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


    dataset = ModelNet40H5_voxelize(
        phase=phase,
        transform=CoordinateTransformation(trans=config.translation)
        if is_train
        else CoordinateTranslation(config.test_translation),
        data_root=config.data_root,
        num_points=config.num_points
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


def criterion(pred, labels, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    labels = labels.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, labels, reduction="mean")

    return loss


def train(net, device, config, logger):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
    )
    logger.info(optimizer)
    logger.info(scheduler)

    train_iter = iter(make_data_loader("train", is_minknet, config))
    best_metric = 0
    net.train()
    for i in range(config.max_steps):
        optimizer.zero_grad()
        try:
            data_dict = train_iter.next()
        except StopIteration:
            train_iter = iter(make_data_loader("train", is_minknet, config))
            data_dict = train_iter.next()
        input = create_input_batch(
            data_dict, is_minknet, device=device, quantization_size=config.voxel_size
        )
        logit = net(input)
        loss = criterion(logit, data_dict["labels"].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if i % config.stat_freq == 0:
            logger.info(f"Iter: {i}, Loss: {loss.item():.3e}")

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                os.path.join(config.logdir, config.weights + "_" + config.network + ".pth"),
            )
            accuracy = test(net, device, config, logger, phase="val")
            if best_metric < accuracy:
                best_metric = accuracy
            logger.info(f"Validation accuracy: {accuracy}. Best accuracy: {best_metric}")
            net.train()

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
                          'classification_modelnet10_voxelize', str(config.num_points), config.network)
    os.makedirs(logdir, exist_ok=True)
    logger = getlogger(logdir)
    
    config.logdir = logdir

    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("===================ModelNet40 Dataset===================")
    logger.info(f"Training with translation {config.translation}")
    logger.info(f"Evaluating with translation {config.test_translation}")
    logger.info("=============================================\n\n")

    net = STR2NETWORK[config.network](
        in_channel=3, out_channel=10, embedding_channel=1024
    ).to(device)
    
    logger.info("===================Network===================")
    logger.info(net)
    logger.info("=============================================\n\n")

    train(net, device, config, logger)
    accuracy = test(net, device, config, logger, phase="test")
    logger.info(f"Test accuracy: {accuracy}")
   