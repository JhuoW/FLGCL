import os.path as osp
import torch
import os
import json
from collections import defaultdict, OrderedDict
import numpy as np

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

def convert_tensor_to_np(state_dict):
    return OrderedDict([(k,v.clone().detach().cpu().numpy()) for k,v in state_dict.items()])

def get_state_dict(model):
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict


def set_state_dict(model, state_dict, gpu_id, skip_stat=False, skip_mask=False): # 默认
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, skip_mask=skip_mask, model=model.state_dict())
    model.load_state_dict(state_dict)

def convert_np_to_tensor(state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue
        if skip_mask:
            if 'mask' in k or 'pre' in k or 'pos' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id)
        else:
            _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
    return _state_dict