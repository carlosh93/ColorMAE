from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import os
import time
import datetime
import matplotlib.pyplot as plt
import subprocess

@HOOKS.register_module()
class RunIterations(Hook):
    """
    Run custom command at the end of a specified number of iterations and submit the next job
    """
    def __init__(self, command, max_iters):
        self.command = command
        self.max_iters = max_iters
        self.counter = 0
        self.start_time = None

    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        if self.counter+1 > self.max_iters:
            runner.logger.info("Exit and submit the next job!")
            if runner.rank == 0:
                assert self.command != ""
                os.system(self.command.replace("+", " "))
            exit(0)
        else:
            # self.start_time = time.time()
            # runner.logger.info(f'SLURM training iteration {self.counter + 1}/{self.max_iters}')
            self.counter += 1

    # def after_train_iter(self, runner, batch_idx, data_batch=None):
    #     total_time = time.time() - self.start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     runner.logger.info('One epoch training time {}'.format(total_time_str))