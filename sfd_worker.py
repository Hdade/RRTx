import numpy as np
import torch, deepinv
import multiprocessing
import cv2, traceback, sys, os

INFO = '\033[94m'
SUCCESS = '\033[92m'
ERROR = '\033[91m'
WARNING = '\033[93m'
RESET = '\033[0m'

def get_seq_schedule(num_steps, total_timesteps=1000):
    if num_steps == 2:
        return torch.tensor([999, 100], dtype=torch.long)
    if num_steps == 4:
        return torch.tensor([999, 600, 200, 0], dtype=torch.long)
    return torch.linspace(total_timesteps - 1, 0, num_steps).long()

def compute_alpha_bars(timesteps, device):
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)

def ddim_step(model, x, t, t_next, alphas_cumprod, condition_img):
    model_input = torch.cat([x, condition_img], dim=1)
    t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    
    output = model(model_input, t_tensor, type_t="timestep")
    pred_noise = output[:, :3, :, :]
    
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_t_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=x.device)
    
    pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    dir_xt = torch.sqrt(1 - alpha_bar_t_next) * pred_noise
    x_next = torch.sqrt(alpha_bar_t_next) * pred_x0 + dir_xt
    return x_next

def compute_offline_ema(checkpoint_dir, start_ep, end_ep, sfd_steps, beta=0.9):
    print(f"{INFO}[SFD Worker] Calculating Offline EMA from epoch {start_ep} to {end_ep} with beta={beta}...{RESET}")
    ema_state_dict = None
    
    for ep in range(start_ep, end_ep + 1):
        ckpt_path = os.path.join(checkpoint_dir, f"tpu_sfd_{sfd_steps}steps_ep{ep}.pth")
        if not os.path.exists(ckpt_path):
            print(f"{INFO}[SFD Worker] WARNING: Cannot find {ckpt_path}, skip.{RESET}")
            continue
            
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        
        if ema_state_dict is None:
            ema_state_dict = state_dict
        else:
            for key in ema_state_dict.keys():
                if ema_state_dict[key].dtype.is_floating_point:
                    ema_state_dict[key] = beta * ema_state_dict[key] + (1 - beta) * state_dict[key]
                    
    print(f"{SUCCESS}[SFD Worker] successfully caculated EMA!{RESET}")
    return ema_state_dict

def sfd_worker_loop(input_queue, output_queue, checkpoint_dir, config_screen_dims, sfd_steps=4, ema_beta=0.85, img_size=128):
    torch.set_num_threads(multiprocessing.cpu_count())
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        
    print(f"{INFO}[SFD Worker] Process started. Initializing DiffUNet on {device}...{RESET}")
    if device == 'cpu':
        print(f"{WARNING}[SFD Worker] WARNING: Not found GPU! Speed will be slow!{RESET}")
    
    try:
        model = deepinv.models.DiffUNet(in_channels=7, out_channels=3, pretrained=None)
        ema_weights = compute_offline_ema(checkpoint_dir, start_ep=1, end_ep=20, sfd_steps=sfd_steps, beta=ema_beta)
        if ema_weights is not None:
            model.load_state_dict(ema_weights)
        else:
            print(f"{ERROR}[SFD Worker] ERROR: Cannot read any checkpoint!. Stop worker.{RESET}")
            return

        model.to(device)
        model.eval()
        alphas_cumprod = compute_alpha_bars(1000, device)
        student_schedule = get_seq_schedule(sfd_steps, 1000).to(device)

        print(f"{INFO}[SFD Worker] Warming-up {device}...{RESET}")
        with torch.no_grad():
            dummy_cond = torch.randn(1, 4, img_size, img_size, device=device)
            dummy_x = torch.randn(1, 3, img_size, img_size, device=device)
            for i in range(len(student_schedule)):
                t_student = student_schedule[i]
                t_next = student_schedule[i+1] if i < len(student_schedule) - 1 else -1
                dummy_x = ddim_step(model, dummy_x, t_student, t_next, alphas_cumprod, dummy_cond)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        print(f"{INFO}[SFD Worker] Model ready for receive data!{RESET}")

        w, h = config_screen_dims
        while True:
            data = input_queue.get()
            if data == 'STOP':
                print(f"{INFO}[SFD Worker] Received STOP signal. Shutting down...{RESET}")
                break
            
            # --- PRE-PROCESSING ---
            map_arr, points_arr = data
            map_gray = cv2.cvtColor(map_arr, cv2.COLOR_RGB2GRAY)
            map_resized = cv2.resize(map_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
            map_tensor = torch.from_numpy(map_resized).float() / 127.5 - 1.0 
            map_tensor = map_tensor.unsqueeze(0).unsqueeze(0)
            points_resized = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255 
            points_arr = np.array(points_arr)
            
            start_mask = (points_arr[:, :, 0] > 150) & (points_arr[:, :, 1] < 100) & (points_arr[:, :, 2] < 100)
            if np.any(start_mask):
                coords = np.argwhere(start_mask)
                y, x = coords.mean(axis=0)
                y_128, x_128 = int(y * img_size / h), int(x * img_size / w)
                cv2.circle(points_resized, (x_128, y_128), 4, (255, 0, 0), -1) 
                
            goal_mask = (points_arr[:, :, 2] > 150) & (points_arr[:, :, 0] < 100) & (points_arr[:, :, 1] < 100)
            if np.any(goal_mask):
                coords = np.argwhere(goal_mask)
                y, x = coords.mean(axis=0)
                y_128, x_128 = int(y * img_size / h), int(x * img_size / w)
                cv2.circle(points_resized, (x_128, y_128), 4, (0, 0, 255), -1) 

            points_tensor = torch.from_numpy(points_resized).float() / 127.5 - 1.0 
            points_tensor = points_tensor.permute(2, 0, 1).unsqueeze(0)
            condition_img = torch.cat([map_tensor, points_tensor], dim=1).to(device)

            # --- INFERENCE ---
            current_x = torch.randn(1, 3, img_size, img_size, device=device)
            with torch.no_grad():
                for i in range(len(student_schedule)):
                    t_student = student_schedule[i]
                    t_next_student = student_schedule[i+1] if i < len(student_schedule) - 1 else -1
                    current_x = ddim_step(model, current_x, t_student, t_next_student, alphas_cumprod, condition_img)

            # --- POST-PROCESSING ---
            pred_img = (current_x.squeeze(0) + 1.0) / 2.0 
            pred_img = torch.clamp(pred_img, 0, 1)
            pred_np = pred_img.cpu().numpy()
            
            r_channel = pred_np[0, :, :]
            g_channel = pred_np[1, :, :]
            b_channel = pred_np[2, :, :]
            heuristic_128 = np.clip(g_channel - np.maximum(r_channel, b_channel), 0, 1)
            heatmap_128 = (heuristic_128 * 255).astype(np.uint8)
            heatmap_full = cv2.resize(heatmap_128, (w, h), interpolation=cv2.INTER_LINEAR)
            
            flat_map = (heatmap_full.astype(np.float64) / 255.0).flatten()
            output_queue.put(flat_map)
            print(f"{SUCCESS}[SFD Worker] Job done! Clean heuristic heatmap sent back.{RESET}")

    except Exception as e:
        print(f"{ERROR}[SFD Worker] CRITICAL ERROR: {e}{RESET}")
        traceback.print_exc()