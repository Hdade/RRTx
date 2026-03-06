import multiprocessing
import numpy as np
import cv2
import traceback
import sys
import os
import torch

sys.path.append(os.getcwd())
from Hybrid_ViT_GAN.inferenceHelper import GANInference

INFO = '\033[94m'
SUCCESS = '\033[92m'
ERROR = '\033[91m'
WARNING = '\033[93m'
RESET = '\033[0m'

def gan_worker_loop(input_queue, output_queue, checkpoint_path, config_screen_dims):
    #Tự động chọn thiết bị: CUDA (Nvidia) > MPS (Mac M1/M2) > CPU
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        
    print(f"{INFO}[GAN Worker] Process started. Initializing model on {device}...{RESET}")
    
    try:
        gan_model = GANInference(checkpoint_path, device=device) 
        print(f"{SUCCESS}[GAN Worker] Model loaded successfully from {checkpoint_path}!{RESET}")

        w, h = config_screen_dims
        while True:
            data = input_queue.get()
            if data == 'STOP':
                print(f"{SUCCESS}[GAN Worker] Received STOP signal. Shutting down...{RESET}")
                break
            
            map_arr, points_arr = data
            heatmap_224 = gan_model.predict(map_arr, points_arr)
            heatmap_full = cv2.resize(heatmap_224, (w, h))
            flat_map = heatmap_full.flatten().astype(np.float64)
            output_queue.put(flat_map)
            print(f"{SUCCESS}[GAN Worker] Job done. Result sent back.{RESET}")

    except Exception as e:
        print(f"{ERROR}[GAN Worker] CRITICAL ERROR: {e}{RESET}")
        traceback.print_exc()