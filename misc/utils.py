import os.path as osp
import torch
import os
import json

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val                           = config[key]
        keystr                        = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    with open(osp.join(base_dir, filename), 'w+') as outfile:
        json.dump(data, outfile)
