import torch
from torchvision import io
from torchvision import transforms
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt


class NormalizeBySliceMax:
    """Normalize each slice of the image tensor by its maximum."""

    def __init__(self):
        pass

    def __call__(self, img):
        # Assuming img is a PyTorch tensor with shape [L, W, W]
        max_values = img.max(dim=-1).values.max(dim=-1).values
        max_values = max_values.unsqueeze(1).unsqueeze(2)
        return img / max_values

    def __repr__(self):
        return self.__class__.__name__


class ColorMasking:
    def __init__(self, W, mask_ratio=0.75, data_path="noise_colors/green/green_noise_data_3072.npz"):

        self.change_color_pattern(data_path)

        self.W = W
        self.mask_ratio = mask_ratio
        self.trans_sequence = transforms.Compose([
            transforms.RandomCrop(W),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            NormalizeBySliceMax()
        ])

    def change_color_pattern(self, data_path):
        try:
            image_tensor = np.load(data_path)
            image_tensor = torch.from_numpy(image_tensor[image_tensor.files[0]])
            if "green" in data_path:
                print(f"=========> Loading Green Noise Patterns: {data_path} <=========")
                self.loaded_color = "green"
            elif "blue" in data_path:
                print(f"=========> Loading Blue Noise Patterns: {data_path} <=========")
                self.loaded_color = "blue"
            elif "purple" in data_path:
                print(f"=========> Loading Purple Noise Patterns: {data_path} <=========")
                self.loaded_color = "purple"
            elif "red" in data_path:
                print(f"=========> Loading Red Noise Patterns: {data_path} <=========")
                self.loaded_color = "red"
            elif "white" in data_path:
                print(f"=========> Loading White Noise Patterns: {data_path} <=========")
                self.loaded_color = "white"

        except:
            raise Exception("Color Noise patterns not found. Download the patterns and place the npz file in the corresponding folder inside noise_colors.")
        self.image_tensor = image_tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_tensor = self.image_tensor.to(self.device)

    def extract_windows(self, B):
        # Define the random crop transform
        L, M, N = self.image_tensor.shape
        windows = []

        # Number of full iterations
        full_iterations = B // L
        # Residual elements
        residual = B % L

        # Extract full L-sized windows
        for _ in range(full_iterations):
            # Apply random crop to get a WxW window
            w_tensor = self.trans_sequence(self.image_tensor)
            windows.append(w_tensor)

        # Extract residual elements if necessary
        if residual > 0:
            w_tensor = self.trans_sequence(self.image_tensor)[:residual]
            windows.append(w_tensor)


        # Stack the windows to form a batch tensor
        return torch.concatenate(windows, dim=0)

    def color_noise_mask(self, window):
        N, W, W = window.shape
        L = W * W
        len_keep = int((L) * (1 - self.mask_ratio))
        window = window.view(N, -1)
        ids_shuffle = torch.argsort(window, dim=1, descending=True)  # keep stronger values from color noise pattern

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Reshape the mask back to the window shape
        return mask, ids_restore, ids_keep
    
    def plotFFT(self, pattern, plt_name="fig.png", show=False):
        fig, axs = plt.subplots(1, 3, figsize=(15, 10))
        axs[0].imshow(pattern)
        axs[1].set_title('Periodogram')
        im = axs[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(pattern - np.mean(pattern)) / pattern.shape[0])),
                           vmin=0.0, vmax=0.85)
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        axs[2].set_title('Log periodogram')
        eps = 1e-12
        im = axs[2].imshow(
            np.log10(abs(np.fft.fftshift(np.fft.fft2(pattern - np.mean(pattern)) / pattern.shape[0])) + eps), vmin=-4,
            vmax=1)
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        if show:
            plt.show()
        else:
            plt.savefig(plt_name)