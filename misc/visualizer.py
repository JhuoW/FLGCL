import os
import os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob




def plot_acc_over_communication(models, upto, opt):
    '''
    models: {'name': name, 
             'viz': viz,
             'logs': [os.path.join(current_dir, path) for path in logs]}
             
    其中logs 是每个logs/Cora/disjoint/10/20250405_175613_FedAvg 文件夹
    '''
    return



def get_acc_over_communication(models, upto = -1, eval_type = 'test'):
    data = {}
    for model in models:
        y_trials = []
        for log in range(model['logs']): # log表示一组log文件夹 包括了10个client的训练数据
            min_n_comms = 9999
            # 用来存放每个client的所有communication的测试精度 client[i]表示client i在所有communication中的测试精度
            clients = {} 
            for i, client in enumerate(glob.glob(os.path.join(log, 'client*.txt'))): # 所有client的log文件
                with open(client) as f:
                    client =json.loads(f.read())
                    # rnd_local_test_acc 表示所有communication的测试精度
                    n_comms = len(client['log'][f'rnd_local_{eval_type}_acc'][:upto]) # communication的次数 
                    if n_comms < min_n_comms:
                        min_n_comms = n_comms
                    # client i在所有communication中的测试精度集合
                    clients[i] = client['log'][f'rnd_local_{eval_type}_acc'] 
            # client[:min_n_comms]表示某个client在所有communication中的测试精度集合
            # 下式表示所有client在的在每个communication后的平均准确率
            y_trial = np.round(np.mean([client[:min_n_comms] for i, client in clients.items()], 0)*100, 2)
            y_trials.append(y_trial)  # 保存每个communication上的平均准确率 （10个client上的平均准确率）所以一共200个值
        
        min_n_comms = np.min([len(y) for y in y_trials]) # 所有client中 需要最少communication的次数
        y = [y[:min_n_comms] for y in y_trials]   # 表示前min_n_comms个communication上 10个client的平均准确率
        y_avg = np.round(np.mean(y, 0), 2) # 10个client在


