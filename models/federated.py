import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from datahelper.loader import DataLoader
import os.path as osp
from misc.utils import *
import numpy as np
import functools
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from misc.logger import Logger


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

# @repeat(3)
def classification_LR(embeddings, batch, mode):
    X = embeddings.detach().cpu().numpy()
    Y = batch.y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)    
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
    X = normalize(X, norm='l2')
    eval_mask = batch.val_mask if mode == 'val' else  batch.test_mask
    X_train = X[batch.train_mask]
    X_test  = X[eval_mask]
    y_train = Y[batch.train_mask]
    y_test  = Y[eval_mask]    
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)    
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)    
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    return {
        'Acc': micro,
        'F1Macro': macro
    }


class ClientModule:
    def __init__(self, args, config, work_id, gpu_id, sd):
        self.sd = sd
        self.gpu_id = gpu_id
        self.worker_id = work_id
        self.args = args 
        self._args = vars(self.args)
        self.loader = DataLoader(self.args)
        self.config = config
        self.logger = Logger(args=self.args, gpu_id = self.gpu_id, is_server = False)
       
    def switch_state(self, client_id):
        self.client_id = client_id
        self.loader.switch(client_id)  # 切换到这个client，从文件中读取这个client的子图
        self.logger.switch(client_id)  # 切换到当前这个client
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return osp.exists(osp.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()

    @torch.no_grad()
    def validate(self, mode='test'):
        loader = self.loader.patition_loader  # client loader， 每次加载一个client

        with torch.no_grad():
            target, pred, loss = [], [], []
            for _, batch in enumerate(loader):  # 加载一个client的subgraph
                batch = batch.cuda(self.gpu_id)  
                mask = batch.test_mask if mode == 'test' else batch.val_mask
                y_hat, lss = self.validation_step(batch, mask)  
                pred.append(y_hat[mask])  # 预测logits
                target.append(batch.y[mask])  # 真实label
                loss.append(lss)
            # 预测logits, 真实stacked labels
            acc = self.accuracy(torch.stack(pred).view(-1, self.config['num_cls']), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @torch.no_grad()
    def eval_LR(self, mode='test'):
        loader = self.loader.patition_loader  # client loader， 每次加载一个client
        with torch.no_grad():
            for _, batch in enumerate(loader):
                batch = batch.cuda()
                if self.config['BaseGCL'] == 'grace':
                    emb, _ = self.local_model(batch.x, batch.edge_index)
                    def repeat(n_times = 3):
                        stats = []
                        for _ in range(n_times):
                            results = classification_LR(emb, batch, mode)
                            stats.append(results)
                        acc_avg = np.mean(np.array([d['Acc'] for d in stats]))
                        acc_std = np.std(np.array([d['Acc'] for d in stats]))
                        ma_avg = np.mean(np.array([d['F1Macro'] for d in stats]))
                        ma_std = np.std(np.array([d['F1Macro'] for d in stats]))
                        stats_results = {'acc_avg': acc_avg, 'acc_std': acc_std, 'f1ma_avg':ma_avg, 'f1ma_std':ma_std}
                        return stats_results
                    stats_results = repeat()
        return stats_results



    @torch.no_grad()
    def validation_step(self, batch, mask=None):
        self.local_model.eval()
        y_hat = self.local_model(batch)
        if torch.sum(mask).item() == 0: return y_hat, 0.0
        lss = self.loss(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    @torch.no_grad()
    def accuracy(self, preds, targets):
        if targets.size(0) == 0: return 1.0
        with torch.no_grad():
            preds = preds.max(1)[1] # 每行的最大值id
            acc = preds.eq(targets).sum().item() / targets.size(0)
        return acc

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log,
            'config': self.config
        })

    def get_optimizer_state(self, optimizer):
        state = {}
        for param_key, param_values in optimizer.state_dict()['state'].items():
            state[param_key] = {}
            for name, value in param_values.items():
                if torch.is_tensor(value) == False: continue
                state[param_key][name] = value.clone().detach().cpu().numpy()
        return state


class ServerModule:
    def __init__(self, args, config, sd, gpu_id):
        self.args = args
        self._args = vars(self.args)
        self.config = config
        self.sd = sd
        self.gpu_id = gpu_id
        self.logger = Logger(args=self.args, gpu_id = self.gpu_id, is_server = True)
    

    def aggregate(self, local_weights, ratio=None):
        raise NotImplementedError()