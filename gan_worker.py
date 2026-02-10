import multiprocessing
import numpy as np
import cv2
import traceback
import sys
import os
import torch

sys.path.append(os.getcwd())
from ViT_GAN.inferenceHelper import GANInference

def gan_worker_loop(input_queue, output_queue, checkpoint_path, config_screen_dims):
    #Tự động chọn thiết bị: CUDA (Nvidia) > MPS (Mac M1/M2) > CPU
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        
    print(f"[GAN Worker] Process started. Initializing model on {device}...")
    
    try:
        gan_model = GANInference(checkpoint_path, device=device) 
        print(f"[GAN Worker] Model loaded successfully from {checkpoint_path}!")

        w, h = config_screen_dims
        while True:
            data = input_queue.get()
            if data == 'STOP':
                print("[GAN Worker] Received STOP signal. Shutting down...")
                break
            
            map_arr, points_arr = data
            heatmap_224 = gan_model.predict(map_arr, points_arr)
            heatmap_full = cv2.resize(heatmap_224, (w, h))
            flat_map = heatmap_full.flatten().astype(np.float64)
            output_queue.put(flat_map)
            print("[GAN Worker] Job done. Result sent back.")

    except Exception as e:
        print(f"[GAN Worker] CRITICAL ERROR: {e}")
        traceback.print_exc()