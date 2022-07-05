import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process logs.')
parser.add_argument('--forward_log', type=str, default='./forward.log',
                    help='Log file for forward')
parser.add_argument('--backward_log', type=str,  default='./backward.log',
                    help='Log file for backward')

args = parser.parse_args()

fw_df = pd.read_csv(args.forward_log, sep='\t')
bw_df = pd.read_csv(args.backward_log, sep='\t')
print(fw_df.describe())
print(bw_df.describe())