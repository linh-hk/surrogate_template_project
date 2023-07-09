#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 15:23:07 2023

@author: h_k_linh
"""
# https://stackoverflow.com/questions/10415028/how-to-get-the-return-value-of-a-function-passed-to-multiprocessing-process
from multiprocessing import Process, Queue
class Multiprocessor():

    def __init__(self):
        self.processes = []
        self.queue = Queue(2)

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets

# tester
if __name__ == "__main__":
    mp = Multiprocessor()
    num_proc = 64
    for _ in range(num_proc): # queue up multiple tasks running `sum`
        mp.run(sum, [1, 2, 3, 4, 5])
    ret = mp.wait() # get all results
    print(ret)
    assert len(ret) == num_proc and all(r == 15 for r in ret)
