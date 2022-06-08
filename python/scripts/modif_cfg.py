#!/usr/bin/python

import yaml
import os
import argparse

def init_parser():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
    description="Evaluating performance of Single person Tracker given ground truth and detection"
    )
    parser.add_argument('-id', default='0',
                    help='id for the video')
    parser.add_argument('-v', '--verbose', action='store_true',
                    help=('Silent Mode if Verbose is False'))
    args = parser.parse_args()
    return args


os.chdir("../src/perceptionloomo/configs")
print(f"Current WD: {os.getcwd()}")
args=init_parser()
with open("cfg_perception.yaml") as f:
    y = yaml.safe_load(f)
    path_benchmark=y['PERCEPTION']['BENCHMARK_FILE'] 
    path, old_id=path_benchmark.rsplit('/', 1)
    print(f"path is {path} and old_id: {old_id}")
    new_id="/Loomo"+args.id+".avi"
    new_path=path+new_id
    y['PERCEPTION']['BENCHMARK_FILE'] = new_path
    y['RECORDING']['EXPID'] = args.id
    print(yaml.dump(y, default_flow_style=False, sort_keys=False))
