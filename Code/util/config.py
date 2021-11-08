'''
    Configuration of meta variables
'''
import argparse
import torch
from torch import cuda

dataset_lookUp = {
    "BBBC": {"path": 'Synthetic_Data_1', "meta": 'MetaData1_GT_link_CP.csv',
             'feat': ['GT_y', 'GT_colorR', 'GT_colorG', 'GT_colorB', 'GT_Shape', 'GT_dist_toInit_state'],
             'GT': "GT_label"},
    "Hovarth": {"path": 'Peter_Horvath_Subsample', "meta": 'MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv',
                'feat': None,
                'GT': "GT_label"},
    "Chaffer": {"path": 'Chaffer_Data', "meta": 'MetaData3_Chaffer_GT_link_CP.csv', 'feat': None,
             'GT': "GT_class"},
    "Felix_FC": {"path": 'Felix_Full_Complete', "meta": 'MetaData_FC_Felix_GT_link_CP.csv', 'feat': None,
             'GT': "GT_class"},
    "Felix_Full_64": {"path": 'Felix_Full_64', "meta": 'MetaData_FC_Felix_GT_link_CP.csv', 'feat': None,
             'GT': "GT_class"},
    "Felix_Full_128": {"path": 'Felix_Full_128', "meta": 'MetaData_FC_Felix_GT_link_CP.csv', 'feat': None,
             'GT': "GT_class"},
    "Felix_C4": {"path": 'Felix_Full_C4', "meta": 'MetaData_FC_Felix_GT_link_CP.csv', 'feat': None,
             'GT': "GT_class"},
    "Felix_Channelwise": {"path": 'Felix_channelwise', "meta": 'MetaData_Channelwise_Felix_GT_link_CP.csv',
                          'feat': None,
             'GT': "GT_class"},
    "Dsprite": {"path": '', "meta": 'MetaData_Dsprite.csv',
                'feat': ['GT_label', 'GT_scale', 'GT_orientation', 'GT_posX', 'GT_posY']},
    "new": {"path": '4Channel_TIFs', "meta": 'MetaData_NEW_GT_link_CP.csv',
            'feat:': None}
}
bool_arg = lambda x: (str(x).lower() == 'true')

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='Felix_Full_64', choices=dataset_lookUp.keys())
args.add_argument('--in', dest="data_path", default='/home/sachahai/Documents/VAE_manifold/DataSets/')
args.add_argument('--out', dest="output_path", default='/mnt/Linux_Storage/outputs/1_experiment')
args.add_argument('--input_size', type=int, help="The size of input images. e.g. 256", default=64)
args.add_argument('--input_channel', type=int, help="The size of input image channel. e.g. 3 (RGB)", default=4)
args.add_argument('-z', dest='hidden_dim', type=int, default=3)
args.add_argument('-l', dest='learning_rate', type=float, default=1e-4)
args.add_argument('--batch', dest='batch_size', type=int, default=64)
args.add_argument('--epochs', type=int, default=2)
args.add_argument('--hidden_size', type=int, default=100)
args.add_argument('--weight_decay', type=float, default=5e-3)
args.add_argument('--early_stopping', type=int, default=30)
args.add_argument('--scheduler_gamma', type=float, default=.6)
args.add_argument('--train', help="Run training steps", type=bool_arg, default=True)
args.add_argument('--eval', type=bool_arg, help="Skip training steps, and run evaluation", default=True)
args.add_argument('--benchmark', type=bool_arg, default=False)
args.add_argument("--saved_model_path", type=str, default="")

device = torch.device('cpu' if not cuda.is_available() else 'cuda')
