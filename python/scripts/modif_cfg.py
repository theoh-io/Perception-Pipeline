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


os.chdir("../src/perceptionloomo/configs/")
print(f"Current WD: {os.getcwd()}")
args=init_parser()
dict={}
with open("cfg_perception.yaml", "r") as f:
    y = yaml.safe_load(f)
    print(y)
    path_benchmark=y['PERCEPTION']['BENCHMARK_FILE'] 
    path, old_id =path_benchmark.rsplit('/', 1)
    new_id="/person-"+args.id
    #path, old_id=path_benchmark.rsplit('/', 1)
    print(f"path is {path} and old_id: {old_id}")
    #new_id="/Loomo"+args.id+".avi"
    new_path=path+new_id
    y['PERCEPTION']['BENCHMARK_FILE'] = new_path
    y['RECORDING']['EXPID'] = int(args.id)
    dict=y
    print(" dict:", dict)
with open("cfg_perception.yaml", "w") as f2:
    yaml.dump(dict, f2)
    print(yaml.dump(dict))

