import os
import torch.multiprocessing as mp
from models.client import Client
import time
import sys

class ParentProcecss:
    def __init__(self, args, config, Server, Client):
        self.args = args
        self.gpus = [int(g) for g in args.gpu_id.split(',')] # multi-process gpus 
        self.gpu_server = self.gpus[0]
        self.proc_id = os.getppid() # 主进程id
        print(f'Main Process ID: {self.proc_id}')

        self.sd = mp.Manager().dict() # 多进程共享的数据 shared data (sd)
        self.sd['is_done'] = False
        
    
    def create_workers(self, Client):
        self.processes = []
        self.q = {}  # 用来存放每个worker需要处理的client_id和communication id
        worker_gpu_dict = {}  # {0:1, 1: 2, 2: 0, 3: 1, 4: 2, 5: 0}  一共0-5共6个线程，每个线程逐一分配gpu 
        for worker_id in range(self.args.n_workers): # 创建多个子线程，并且为每个子线程分配一个gpu
            # 若n_workers = 6 说明有6个子线程
            # 为每个子线程逐一分配GPU
            # 当前子线程的gpu id
            gpu_id = self.gpus[worker_id+1] if worker_id < len(self.gpus)-1 else self.gpus[(worker_id-(len(self.gpus)-1))%len(self.gpus)]
            worker_gpu_dict[worker_id] = gpu_id
            print(f'Worker {worker_id} is assigned to GPU {gpu_id}')
            self.q[worker_id] = mp.Queue()  # 为每个worker创建一个queue
            # 创建一个子进程，每个子进程都是一个worker，
            p = mp.Process(target = WorkerProcess, args = (self.args, self.config, worker_id, gpu_id, self.q[worker_id], self.sd, Client))
            p.start()
            self.processes.append(p)
        

class WorkerProcess:
    def __init__(self, args, config, worker_id, gpu_id, q, sd, Client: Client): 
        self.args = args
        self.config = config
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.q = q
        self.sd = sd
        self.client = Client(self.args, self.config, self.worker_id, self.gpu_id, self.sd) # 创建一个client对象
        self.is_done = False
        self.listen()

    def listen(self):
        while not self.sd['is_done']:
            msg = self.q.get()
            if not msg == None: # 监听，一旦mesg有值（client_id， communication_id）立刻开始训练该client
                client_id, curr_comm = msg

                self.client.switch_state(client_id) # 设置loader和logger 使他们加载client id的子图数据，将他们设置成当前Client对象的数据来源
                self.client.on_receive_message(curr_comm)  # 从server端拷贝数据到当前client
                self.client.on_communication_begin()  # 完成当前client的训练，并将模型参数存入sd中
                self.client.save_state()  # 保存当前communication后的client模型
            time.sleep(1.0)
        print('[main] Terminating worker processes ... ')
        sys.exit()