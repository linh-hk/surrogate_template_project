#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:35:14 2023

@author: h_k_linh
"""
import os
import numpy as np
from multiprocessing import Process, Queue

"""
                Start by initiating a Multiprocessor object
mp = Multiprocessor()
                Then add all processes to object
for smt in smt:
    ARGs=(X, each_PS[1], each_PS[0], KX, RX, n, embed_dim, tau, pred_lag, weights, score) # ARGs is tuple of all arguments
    mp.add(ccm_iterate_predict_surr, ARGs) # add a process
                After having all processes saved in self.processes, 
                execute them by 
mp.run(5) # 5 is the number of processes that can be run at once
                Then pull all results back
result = mp.results()
"""
def info(title):
    # inf = {'fxn name': title, 'from module': __name__, 'parent process': os.getppid(), 'process id': os.getpid()}
    print(f"multiprocessing {title}")

class Multiprocessor:

    def __init__(self):
        self.processes = []
        self.queue = Queue()
        self.result = []

    @staticmethod
    def _wrapper(func, queue, args):
        if type(args) == type({}):
            ret = func(**args)
        else:
            ret = func(*args)
        queue.put(ret)
        info(func.__name__)
        
    # def done(self):
    #     self.queue.put(["DONE"])
        

    def add(self, func, args):
        args2 = [func, self.queue, args]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        # print(func.__name__)
        
    def run(self, num_proc):
        tot_proc = len(self.processes)
        print(tot_proc)
        for proc in np.arange(0,tot_proc, num_proc):
            for i in np.arange(num_proc):
                if proc + i < tot_proc:
                    self.processes[proc+i].start()
                    print(f"Starting {proc+i}")
            for i in np.arange(num_proc):
                if proc + i < tot_proc:
                    self.processes[proc+i].join()
                    print(f"Waiting for {proc+i} res")
            q_size = self.queue.qsize()
            print(q_size)
            while q_size:
                ret = self.queue.get()
                self.result.append(ret)
                q_size -= 1
                print(q_size)
        # if tot_proc < num_proc:
        #     for proc in np.arange(tot_proc):
        #         self.processes[proc].start()
        #     for proc in np.arange(tot_proc):
        #         self.processes[proc].join()
        #     q_size = self.queue.qsize()
        #     while q_size:
        #         ret = self.queue.get()
        #         print(self.queue.qsize())
        #         self.result.append(ret)
        #         q_size -= 1
                
        # else:
        #     for proc in np.arange(tot_proc):
        #         self.processes[proc].start()
        #         print(f'Proc {proc} running')
        #         if proc % num_proc == num_proc-1 or proc == tot_proc - 1:
        #             for i in np.arange(num_proc):
        #                 if num_proc-i <tot_proc:
        #                     print(proc, i)
        #                     self.processes[proc-i].join()
        #             q_size = self.queue.qsize()
        #             print(f"before get {q_size}")
        #             while q_size:
        #                 ret = self.queue.get()
        #                 print(self.queue.qsize())
        #                 self.result.append(ret)
        #                 q_size -= 1
        #             print(f"after get {q_size}")
        #     # self.processes.append(Process(target = self.done))
        #     # self.processes[tot_proc].start()
        #     # self.processes[tot_proc].join()
        #                     # print(f'Waiting for proc {proc+i}')

    def results(self):
        q_size = self.queue.qsize()
        print(f"{q_size} left")
        while q_size:
                    ret = self.queue.get()
                    print(self.queue.qsize())
                    self.result.append(ret)
                    q_size -= 1
        if self.queue.empty():
            print('Queue empty')
        
        return self.result