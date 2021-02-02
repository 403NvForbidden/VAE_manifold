'''
    Configuration of meta variables
    TODO: complte the config file to aovid
'''

import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='BBBC', choices=['BBBC, MNIST, 2dSprite'])
args.add_argument('--in', dest="input_path", default='../DataSets/')
args.add_argument('--out', dest="output_path", default='../outputs/')
args.add_argument('--input_size', type=int, default=64)
args.add_argument('--input_channel', type=int, default=3)
args.add_argument('-z', dest='hidden_dim', type=int, default=3)
args.add_argument('-l', dest='learning_rate', type=float, default=1e-4)
args.add_argument('--double_embed', type=bool, default=False)
args.add_argument('--batch', dest='batch_size', type=int, default=128)
args.add_argument('--epochs', type=int, default=2)
args.add_argument('--hidden_size', type=int, default=100)
args.add_argument('--save', type=bool, default=True)
args.add_argument('--dropout', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-3)
args.add_argument('--early_stopping', type=int, default=30)
args.add_argument('--scheduler_gamma', type=float, default=.6)
# args = args.parse_args()

dataset_lookUp = {
    "BBBC": {"path": 'Synthetic_Data_1', "meta": 'MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'},
    "Hovarth": {"path": 'Peter_Horvath_Subsample', "meta": 'MetaData1_GT_link_CP.csv'},
    "Chaffer": {"path": 'Chaffer_Data', "meta": 'MetaData3_Chaffer_GT_link_CP.csv'},
}

# print(args)
