import os
import os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import matplotlib



def plot_acc_over_communication(models, upto, opt):
    '''
    models: {'name': name, 
             'viz': viz,
             'logs': [os.path.join(current_dir, path) for path in logs]}
             
    其中logs 是每个logs/Cora/disjoint/10/20250405_175613_FedAvg 文件夹
    '''
    processed = get_acc_over_communication(models, upto, eval_type = 'test')

    matplotlib.rcParams.update({'font.size': opt['plt_font_size']})
    matplotlib.rcParams['axes.linewidth']=opt['plt_line_width']
    plt.figure(figsize=opt['plt_fig_size'])
    plt.title(opt['plt_title'], fontsize=opt['plt_font_size'])
    plt.ylabel(opt['plt_y_label'], fontsize=opt['plt_font_size'])
    plt.xlabel(opt['plt_x_label'], fontsize=opt['plt_font_size'])
    if opt['plt_background_grid']:
        plt.grid(linestyle='-.', linewidth=0.5)
    for name, proc in processed.items():
        if 'plt_y_interval' in opt.keys():
            proc['x'] = proc['x'][proc['y'] > opt['plt_y_interval'][0]]
            proc['y'] = proc['y'][proc['y'] > opt['plt_y_interval'][0]]
        plt.plot(proc['x'], proc['y'], 
            label=name, color=proc['viz']['color'], 
            linewidth=proc['viz']['linewidth'], linestyle=proc['viz']['linestyle'],
            marker=proc['viz']['marker'] if 'marker' in proc['viz'] else None, 
            markevery=proc['viz']['markevery']  if 'markevery' in proc['viz'] else None,
            markersize=proc['viz']['markersize']  if 'markersize' in proc['viz'] else None,
            clip_on=False)
    plt.xlim([1,upto+1])
    plt.xticks(np.concatenate([[1],np.arange(0,upto+1,opt['plt_x_interval'])[1:]], 0))
    if 'plt_y_interval' in opt.keys():
        plt.yticks(opt['plt_y_interval'])
    plt.tight_layout()    
    plt.legend(**opt['plt_legend_opt'])
    plt.savefig(opt['plt_save'], dpi=300) 
    plt.show()



def get_acc_over_communication(models, upto = -1, eval_type = 'test'):
    data = {}
    for model in models:
        y_trials = []
        for log in model['logs']: # log表示一组log文件夹 包括了10个client的训练数据 表示同一个模型的运行次数
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
        y = [y[:min_n_comms] for y in y_trials]   # 表示前min_n_comms个communication上 10个client的平均准确率,一共min_n_comms个值
        y_avg = np.round(np.mean(y, 0), 2) # 表示多次运行同一个模型得到的communication level 准确率的平均
        y_std = np.round(np.std(y, 0), 2) # 表示多次运行同一个模型得到的communication level 准确率的标准差

        data[model['name']] = {
            'x': np.arange(len(y_avg)) + 1,
            'y': y_avg,
            'std': y_std,
            'y_all':y, 
            'process_comm': min_n_comms,
            'viz': model['viz']
        }
    return data

def summary(models, upto, target_acc, target_comm):
    COMM_AT_LOC_PREF_ON_PART = f'Comm @ Acc {target_acc}'
    LOC_PERF_ON_PART_AT_COMM = f'ACC @ Comm {target_comm}'
    LOC_PERF_ON_PART_AT_BEST_VAL = f'Acc @ Best Val'
    LOC_PERF_ON_PART_AT_BEST_VAL_ALL = f'Acc @ Best Val All'
    LOC_PERF_ON_PART_AT_BEST_VAL_STD = f'Std @ Best Val'
    LOC_PERF_ON_PART_AT_BEST_VAL_ALL_STD = f'Std @ Best Val All'

    processed = {
        'model': [],
        COMM_AT_LOC_PREF_ON_PART: [],   # 测试准确率>target acc的communication id
        LOC_PERF_ON_PART_AT_COMM: [],    # target comm下模型的测试准确率
        LOC_PERF_ON_PART_AT_BEST_VAL: [],  # 最大验证准确率时的测试acc
        LOC_PERF_ON_PART_AT_BEST_VAL_STD: [],  # 测试准确率的标准差
        LOC_PERF_ON_PART_AT_BEST_VAL_ALL: [],  # 表示运行模型N次后，所有running达到最大验证准确率时的测试准确率 取平均，即保存运行N次程序后的平均测试准确率
        LOC_PERF_ON_PART_AT_BEST_VAL_ALL_STD: [],
        'Processed Comms': []        
    }
    ltest = get_acc_over_communication(models, upto, eval_type = 'test') 
    lval = get_acc_over_communication(models, upto, eval_type = 'val')

    for model in models:
        local_test_acc = ltest[model['name']]['y']  # 当前模型在所有communication上所有client的平均测试精度
        local_val_acc = lval[model['name']]['y']
        local_val_acc_all = lval[model['name']]['y_all']  # 每个communication上，所有client的平均精度
        local_test_acc_all = ltest[model['name']]['y_all']

        # comm, acc表示当前模型每个communication上的所有client的平均精度，若有200个communication，那么就有200个平均精度
        # 每个精度是10个client在该communication结束后的平均精度
        for comm, acc in enumerate(local_test_acc): 
            if acc >= target_acc:
                processed[COMM_AT_LOC_PREF_ON_PART].append(comm + 1) # 保存准确率大于target acc的communication
                break
            if comm + 1 == len(local_test_acc):
                processed[COMM_AT_LOC_PREF_ON_PART].append('N/A')  # 

        # 对于一个target comm, 保存这个communication上所有client的平均精度
        processed[LOC_PERF_ON_PART_AT_COMM].append(local_test_acc[target_comm - 1]) 

        idx = np.argmax(local_val_acc) # 表示得到最大验证准确率的communication索引
        # 保存 在获得最大验证准确率的communication上的test准确率
        processed[LOC_PERF_ON_PART_AT_BEST_VAL].append(local_test_acc[idx])

        processed[LOC_PERF_ON_PART_AT_BEST_VAL_STD].append(np.round(np.std(np.array(local_test_acc_all)[:, idx]), 2))
        

        idx_all = np.argmax(local_val_acc_all, 1) # 若2次运行模型，表示每次运行得到最大验证准确率的communication id
        _local_test_acc_all = []  # 保存每次运行模型得到最大验证准确率

        for _i, _idx in enumerate(idx_all):
            # 第i次运行在第_idx个communication上获得最大验证准确率, 将此时的测试准确率放入_local_test_acc_all 中
            _local_test_acc_all.append(local_test_acc_all[_i][_idx])  # 每次communication，获得最大平均准确率的运行上的测试准确率
        
        processed[LOC_PERF_ON_PART_AT_BEST_VAL_ALL].append(np.round(np.mean(_local_test_acc_all), 2)) # 所有communication上，获得最大平均准确率的运行上的测试准确率
        processed[LOC_PERF_ON_PART_AT_BEST_VAL_ALL_STD].append(np.round(np.std(_local_test_acc_all), 2))
        processed['model'].append(model['name'])
        processed['Processed Comms'].append(len(ltest[model['name']]['y'])) 
    pd.options.display.max_columns = None
    df = pd.DataFrame(data = processed)

    return df