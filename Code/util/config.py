'''
    Configuration of meta variables
    TODO: complte the config file to aovid
'''

import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora')
args.add_argument('--model', default='2_stage_infoMaxVAE')
args.add_argument('--input_size', type=int, default=64)
args.add_argument('--input_channel', type=int, default=3)
args.add_argument('--learning_rate', type=float, default=1e-4)
args.add_argument('--double_embed', type=bool, default=False)
args.add_argument('--batch', type=int, default=32)
args.add_argument('--epochs', type=int, default=5)
args.add_argument('--hidden_size', type=int, default=100)
args.add_argument('--dropout', type=float, default=0.5)
args.add_argument('--weight_decay', type=float, default=5e-3)
args.add_argument('--early_stopping', type=int, default=30)

args = args.parse_args()
print(args)
