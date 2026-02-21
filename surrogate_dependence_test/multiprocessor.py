#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:35:14 2023

@author: h_k_linh
"""
import logging
log = logging.getLogger(__name__)
import os
# import numpy as np
# import time
import sys
import traceback
import pickle
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

class Multiprocessor:
    def __init__(self, output_file=None):
        self.processes = []
        self.queue = Queue()
        self.result = []
        self.output_file = output_file
        if self.output_file is not None:
            outdir = os.path.dirname(self.output_file)
            if outdir:
                os.makedirs(outdir, exist_ok= True)

    @staticmethod
    def _wrapper(func, queue, args):
        try:
            if isinstance(args, dict):
                ret = func(**args)
            else:
                ret = func(*args)
            queue.put(("OK", ret))
        except Exception:
            tb = traceback.format_exc()

            # Print from worker so it appears in the job log even if parent gets stuck
            log.error("=== WORKER EXCEPTION TRACEBACK ===")
            log.error(tb)

            # Send full traceback back to parent
            queue.put(("ERR", tb))

    def add(self, func, args):
        p = Process(target=self._wrapper, args=(func, self.queue, args))
        self.processes.append(p)

    def run(self, num_proc, per_result_timeout_s=None):
        tot_proc = len(self.processes)
        for start in range(0, tot_proc, num_proc):
            batch = self.processes[start:start+num_proc]

            for p in batch:
                p.start()
                # print("Started", p.pid, flush=True)

            # pull results for this batch as they finish (prevents queue filling)
            for _ in batch:
                try:
                    if per_result_timeout_s is None:
                        status, payload = self.queue.get()  # blocks until one result
                    else:
                        status, payload = self.queue.get(timeout=per_result_timeout_s)
                except Exception:
                    for p in batch:
                        if p.is_alive():
                            p.terminate()
                    for p in batch:
                        p.join()
                    raise TimeoutError(f"Multiprocessor timeout: no worker result within "
                                       f"{per_result_timeout_s} seconds (batch starting index {start})."
                                       )
                if status == "ERR":
                    # worker raised exception (payload is full traceback)
                    log.error("=== PARENT RECEIVED WORKER TRACEBACK ===")
                    log.error(payload)
                    for p in batch:
                        if p.is_alive():
                            p.terminate()
                    for p in batch:
                        p.join()
                    raise RuntimeError(f"Worker error: {payload}")
                
                if self.output_file is not None:
                    with open(self.output_file, "ab") as f:
                        pickle.dump(payload, f)
                else:
                    self.result.append(payload)

            for p in batch:
                p.join()
                # print("Joined", p.pid, flush=True)

    def results(self):
        return self.result