import argparse
import yaml
import os
import shutil

from utils.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='config.yaml',
                        type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    shutil.copy(args.cfg, "{}/{}/".format(cfg['output']['output_folder'], cfg['output']['description']))

    trainer = Trainer(cfg)
    trainer.train()