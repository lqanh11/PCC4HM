import sys
sys.path.append('../PointNet/PointNet')

import os
import sys
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader_h5 import ModelNetDataLoader_h5, ModelNetDataLoader_h5_all



BASE_DIR = '../PointNet/PointNet'
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=512, help='Point Number')
    parser.add_argument('--resolution', type=int, default=64, help='Resolution for Voxelized')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def plot_confusion_matrix(y_true, y_pred, save_fig_dir):
    target_names = ['Bathtub','Bed','Chair','Desk','Dresser','Monitor','Nightstand','Sofa','Table','Toilet']
    label_names = list(range(10))

    print(classification_report(y_true, y_pred, labels=label_names, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred, labels=label_names,
                          normalize='true')
    accuracy = accuracy_score(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(cm, annot=True, cmap="Blues", cbar=True, fmt='.2',
                xticklabels=target_names, 
                yticklabels=target_names,)
    # disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    # disp.plot(cmap="Blues", values_format=".2f", ax=ax, xticks_rotation=45, colorbar=True)
    plt.xticks(rotation=45) 

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_fig_dir,'confusion_matrix.jpg'), dpi=300)
    print(os.path.join(save_fig_dir,'confusion_matrix.jpg'))


def test(model, loader, num_class=10, save_fig_dir=None):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    all_targets = []
    all_predictions = []

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        all_targets = np.append(all_targets, target.cpu().numpy())
        all_predictions = np.append(all_predictions, pred_choice.cpu().numpy())

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    plot_confusion_matrix(all_targets, all_predictions, save_fig_dir)

    return instance_acc, class_acc

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('/media/avitech/Data/quocanhle/PointCloud/logs/PointNet_cls/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    save_fig_dir = exp_dir.joinpath('figs/')
    save_fig_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    data_path = '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_modelnet10_ply_hdf5_single_file/precise_coordinates/'

    test_dataset = ModelNetDataLoader_h5(root=data_path, args=args, split='test', process_data=args.process_data)
    
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    num_class = args.num_category
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class, normal_channel=args.use_normals)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        
    except:
        log_string('No existing model, starting training from scratch...')

    classifier.eval()
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class, save_fig_dir=save_fig_dir)

        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

if __name__ == '__main__':
    args = parse_args()
    main(args)