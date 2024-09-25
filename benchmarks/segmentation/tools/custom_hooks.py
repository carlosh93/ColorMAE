from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import os
import time
import datetime
import matplotlib.pyplot as plt
import subprocess

@HOOKS.register_module()
class RunOneEpoch(Hook):
    """
    Run custom command at the end of each epoch and submit the next job
    """
    def __init__(self, command, max_epochs):
        self.command = command
        self.max_epochs = max_epochs
        self.counter = 0
        self.start_time = None
        self.backups = 0
    def before_train_epoch(self, runner):
        if self.counter+1 > self.max_epochs:
            runner.logger.info("Exit and submit the next job!")
            if runner.rank == 0:
                assert self.command != ""
                os.system(self.command.replace("+", " "))
            exit(0)
        else:
            if 'mae' in runner.experiment_name:
                # self.backup_specific_ckpt(runner, [100, 200])
                self.check_loaded_color(runner)
                self.print_example_codes(runner, [0, 100, 200])
            self.start_time = time.time()
            runner.logger.info(f'SLURM training epoch {self.counter + 1}/{self.max_epochs}')
            self.counter += 1

    def backup_specific_ckpt(self, runner, bk_epochs):
        if runner.epoch-1 not in bk_epochs:
            return
        if runner.rank == 0:
            for ep in bk_epochs:
                ckpt_path = os.path.join(runner.work_dir, f'epoch_{ep}.pth')
                backup_path = os.path.join(runner.work_dir, f'bk_epoch_{ep}.pth')

                if os.path.exists(ckpt_path) and not os.path.exists(backup_path):
                    try:
                        process = subprocess.Popen(["cp", ckpt_path, backup_path])
                        runner.logger.info(f'Backed up {ckpt_path} to {backup_path}')
                        self.backups += 1
                    except Exception as e:
                        runner.logger.error(f"Error backing up {ckpt_path} to {backup_path}: {e}")

    def after_train_epoch(self, runner):
        total_time = time.time() - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        runner.logger.info('One epoch training time {}'.format(total_time_str))

    def check_loaded_color(self, runner_obj):
        if "MMDistributedDataParallel" in str(runner_obj.model):
            run_model = runner_obj.model.module
        else:
            run_model = runner_obj.model
        schedule = run_model.backbone.mask_type
        if isinstance(schedule, str):
            return
        current_epoch = runner_obj.epoch
        # Create a list to hold the epoch numbers and their corresponding data paths
        epoch_data_paths = []

        # Iterate through the keys in epoch_dict
        for key, value in schedule.items():
            if key == 'begin':
                epoch_number = 0
            else:
                # Extract the epoch number from the key
                epoch_number = int(key.split('_')[1])

            # Append the epoch number and data path to epoch_data_paths
            epoch_data_paths.append((epoch_number, value['data_path'], value['name']))

        # Sort epoch_data_paths by epoch number in ascending order
        epoch_data_paths.sort(key=lambda x: x[0])
        # Find the correct data path based on the current epoch
        current_data_path = None
        for i in range(len(epoch_data_paths)):
            if epoch_data_paths[i][0] <= current_epoch < (
            epoch_data_paths[i + 1][0] if i + 1 < len(epoch_data_paths) else float('inf')):
                current_data_path=epoch_data_paths[i]
                break

        if run_model.backbone.masking_generator.loaded_color != current_data_path[2]:
            runner_obj.logger.info(f"====> Change Color Patterns. Loading {current_data_path[2]} patterns from {current_data_path[1]} <====")
            run_model.backbone.masking_generator.change_color_pattern(current_data_path[1])
            self.print_example_codes(runner_obj, [current_epoch])


    def print_example_codes(self, runner_obj, epochs=None):
        if runner_obj.rank == 0:
            if "MMDistributedDataParallel" in str(runner_obj.model):
                run_model = runner_obj.model.module
            else:
                run_model = runner_obj.model
            if isinstance(run_model.backbone.mask_type, str):
                # if the mask_type is a string (random) then we are not using the masking generator
                return
            if epochs is None:
                epochs = [100, 200]
            if runner_obj.epoch in epochs:
                runner_obj.logger.info(f"Printing some example {run_model.backbone.masking_generator.loaded_color} pattern codes")
                res = run_model.backbone.masking_generator.blue_noise_mask(run_model.backbone.masking_generator.extract_windows(5))
                # plot the first 4 examples with matplotlib in a 2x2 grid
                plt.figure(figsize=(10,10))
                for i in range(4):
                    plt.subplot(2,2,i+1)
                    plt.imshow(res[0][i].reshape([14,14]).cpu().numpy())

                plt.savefig(f"{runner_obj.work_dir}/example_codes_{runner_obj.epoch}.png")
                plt.close()
