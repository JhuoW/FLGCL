import os
import torch.multiprocessing as mp
import time
import sys
import os.path as osp
import numpy as np
import atexit

class ParentProcess:
    def __init__(self, args, config, Server, Client):
        self.args = args
        self.gpus = [int(g) for g in args.gpu_ids.split(',')] # multi-process gpus  [0,1]
        self.gpu_server = self.gpus[0] # gpu of server
        self.proc_id = os.getppid() # 主进程id
        self.config = config
        print(f'Main Process ID: {self.proc_id}')
        mp.set_start_method('spawn')
        self.sd = mp.Manager().dict() # 多进程共享的数据 shared data (sd)
        self.sd['is_done'] = False  # 用于存放是否所有communication都执行完
        self.create_workers(Client) # 创建多个子线程
        self.server = Server(args, config, self.sd, self.gpu_server)
        atexit.register(self.done)
    
    def create_workers(self, Client):
        self.processes = []
        self.q = {}  # 用来存放每个worker需要处理的client_id和communication id
        # 一个GPU用于处理多个线程
        worker_gpu_dict = {}  # {0:1, 1: 2, 2: 0, 3: 1, 4: 2, 5: 0}  一共0-5共6个线程，每个线程逐一分配gpu 
        # 6个线程并行，平均分配给3个GPU
        for worker_id in range(self.args.n_workers): # 创建多个子线程，并且为每个子线程分配一个gpu
            # 若n_workers = 6 说明有6个子线程
            # 为每个子线程逐一分配GPU
            # 当前子线程的gpu id
            gpu_id = self.gpus[worker_id+1] if worker_id < len(self.gpus)-1 else self.gpus[(worker_id-(len(self.gpus)-1))%len(self.gpus)]
            # gpu_id = self.gpus[worker_id+1] if worker_id < len(self.gpus)-1 else self.gpus[(worker_id-(len(self.gpus)-1))%len(self.gpus)]
            # 设置用于处理当前线程的GPU
            worker_gpu_dict[worker_id] = gpu_id
            print(f'Worker {worker_id} is assigned to GPU {gpu_id}')
            # 为当前线程创建一个队列， 该队列中存放(client_id, communication_id)
            # 表示该线程中要处理的 client在某次communication中的训练过程
            self.q[worker_id] = mp.Queue()
            # 生成一个独立于主进程的新的线程
            # 开启线程后，会执行WorkerProcess中的代码
            # 该线程会创建一个client，并将该client的worker id和gpu id设置为该线程
            p = mp.Process(target = WorkerProcess, args = (self.args, self.config, worker_id, gpu_id, self.q[worker_id], self.sd, Client))
            p.start()
            self.processes.append(p)
        
            
    def start(self):
        '''
        开启主进程: 主进程中开始communication的训练
        '''
        self.sd['is_done'] = False
        if not osp.isdir(self.args.checkpt_path):
            os.makedirs(self.args.checkpt_path)
        if not osp.isdir(self.args.log_path):
            os.makedirs(self.args.log_path)
        
        self.num_clients = round(self.args.n_clients * self.args.frac)  # 10
        # 每次communication 训练所有client
        for curr_comm in range(self.config['n_comm']):  
            
            self.curr_comm = curr_comm # 当前communication
            self.updated = set()
            np.random.seed(self.args.seed + curr_comm)  #当前client的seed
            # 随机排列所有client id
            self.selected = np.random.choice(self.args.n_clients, self.num_clients, replace=False).tolist()  # [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]
            # 将当前server中的global model参数设置到 sd['global']['model']中 用于在server-client间共享
            self.server.on_communication_begin(curr_comm)   
            while len(self.selected)>0: # 逐一取出client
                _selected = [] 
                
                for worker, q in self.q.items():
                    c_id = self.selected.pop(0) # 取出一个client
                    _selected.append(c_id)  # 存放已处理过的client
                    # q[worker] = (c_id, curr_comm) 表示为线程worker分配它要处理的client的当前的communication id
                    q.put((c_id, curr_comm))  # c_id = 8, curr_comm = 0; c_id =5 curr_comm = 0 ... 
                    if len(self.selected) == 0: # 如果所有client都被分配完
                        break
                self.wait(_selected) # 将c_id加入self.update中，用于保存在当前communication中待训练的client
            self.server.on_communication_end(self.updated) # 一次communication结束后，对参与的client参数做聚合后赋值个global model
            print(f'[Global] Communication {curr_comm} DONE')
        
        self.sd['is_done'] = True

        for worker_id, q in self.q.items():
            q.put(None)
        print(f'Server DONE')
        sys.exit()

                

    def wait(self, _selected):
        cont = True
        while cont:
            cont = False
            for c_id in _selected:
                if not c_id in self.sd:
                    cont = True
                else:
                    self.updated.add(c_id)
            time.sleep(0.1)
        
            
    def done(self):
        for p in self.processes:
            p.join()
        print('[main] All children have joined. Destroying main process ...')       

class WorkerProcess:
    def __init__(self, args, config, worker_id, gpu_id, q, sd, Client): 
        '''
        q: 表示
        '''
        self.args = args 
        self.config = config
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.q = q
        self.sd = sd
        self.is_done = False
        # 创建一个由线程gpu_id上的线程work_id处理的client对象
        self.client = Client(self.args, self.config, self.worker_id, self.gpu_id, self.sd) # 创建一个client对象
        self.listen() # 训练q中的所有client

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
