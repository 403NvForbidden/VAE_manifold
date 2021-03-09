'''
    Configuration of meta variables
    TODO: complte the config file to aovid
'''

import argparse
import torch
from torch import cuda

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='Felix_Channelwise', choices=['BBBC', 'MNIST', '2dSprite','Felix_FC', 'Felix_Channelwise'])
args.add_argument('--in', dest="input_path", default='../DataSets/')
args.add_argument('--out', dest="output_path", default='/mnt/Linux_Storage/outputs/')
args.add_argument('--input_size', type=int, default=256)
args.add_argument('--input_channel', type=int, default=1)
args.add_argument('-z', dest='hidden_dim', type=int, default=3)
args.add_argument('-l', dest='learning_rate', type=float, default=1e-4)
args.add_argument('--batch', dest='batch_size', type=int, default=32)
args.add_argument('--epochs', type=int, default=20)
args.add_argument('--hidden_size', type=int, default=100)
args.add_argument('--save', type=bool, default=True)
args.add_argument('--dropout', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-3)
args.add_argument('--early_stopping', type=int, default=30)
args.add_argument('--scheduler_gamma', type=float, default=.6)
args.add_argument('--train', type=bool, default=True)

device = torch.device('cpu' if not cuda.is_available() else 'cuda')
dataset_lookUp = {
    "BBBC": {"path": 'Synthetic_Data_1', "meta": 'MetaData1_GT_link_CP.csv', 'feat': ['GT_y', 'GT_colorR', 'GT_colorG', 'GT_colorB', 'GT_Shape', 'GT_dist_toInit_state']},
    "Hovarth": {"path": 'Peter_Horvath_Subsample', "meta": 'MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv', 'feat': None},
    "Chaffer": {"path": 'Chaffer_Data', "meta": 'MetaData3_Chaffer_GT_link_CP.csv', 'feat': None},
    "Felix_FC": {"path": 'Felix_Full_Complete', "meta": 'MetaData_FC_Felix_GT_link_CP.csv', 'feat': None},
    "Felix_C4": {"path": 'Felix_Full_C4', "meta": 'MetaData_FC_Felix_GT_link_CP.csv', 'feat': None},
    "Felix_Channelwise": {"path": 'Felix_channelwise', "meta": 'MetaData_Channelwise_Felix_GT_link_CP.csv', 'feat': None},
}