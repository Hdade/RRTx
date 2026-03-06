import torch
import deepinv
import os, time, glob
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

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
    print(f"--> Đang tính toán Offline EMA từ epoch {start_ep} đến {end_ep} với beta={beta}...")
    ema_state_dict = None
    
    for ep in range(start_ep, end_ep + 1):
        ckpt_path = os.path.join(checkpoint_dir, f"tpu_sfd_{sfd_steps}steps_ep{ep}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Cảnh báo: Không tìm thấy {ckpt_path}, bỏ qua.")
            continue
            
        print(f"  + Đang merge checkpoint epoch {ep}...")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        if ema_state_dict is None:
            ema_state_dict = state_dict
        else:
            for key in ema_state_dict.keys():
                if ema_state_dict[key].dtype.is_floating_point:
                    ema_state_dict[key] = beta * ema_state_dict[key] + (1 - beta) * state_dict[key]
                    
    print("--> Hoàn tất tính EMA!")
    return ema_state_dict

def load_test_condition(map_path, points_path, image_size=128, device='cuda'):
    transform_gray = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_rgb = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    map_img = Image.open(map_path).convert("L")
    points_img = Image.open(points_path).convert("RGB")
    
    map_tensor = transform_gray(map_img)
    points_tensor = transform_rgb(points_img)
    
    condition_tensor = torch.cat([map_tensor, points_tensor], dim=0).unsqueeze(0)
    return condition_tensor.to(device)

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 128
    SFD_STEPS = 4
    CHECKPOINT_DIR = "../checkpoints/SFD_checkpoints"
    BASE_TEST_DIR = "./"
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = deepinv.models.DiffUNet(in_channels=7, out_channels=3, pretrained=None)
    
    ema_weights = compute_offline_ema(CHECKPOINT_DIR, start_ep=1, end_ep=20, sfd_steps=SFD_STEPS, beta=0.85)
    model.load_state_dict(ema_weights)
    model.to(DEVICE)
    model.eval()
    # model = torch.compile(model, mode="reduce-overhead")

    alphas_cumprod = compute_alpha_bars(1000, DEVICE)
    student_schedule = get_seq_schedule(SFD_STEPS, 1000).to(DEVICE)
    
    print("--> Đang khởi động (Warm-up) GPU...")
    with torch.no_grad():
        dummy_cond = torch.randn(1, 4, IMG_SIZE, IMG_SIZE, device=DEVICE)
        dummy_x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        for i in range(len(student_schedule)):
            t_student = student_schedule[i]
            t_next = student_schedule[i+1] if i < len(student_schedule) - 1 else -1
            dummy_x = ddim_step(model, dummy_x, t_student, t_next, alphas_cumprod, dummy_cond)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    map_path = os.path.join(BASE_TEST_DIR, "map.png")
    
    total_inference_time = 0
    with torch.no_grad():
        points_path = os.path.join(BASE_TEST_DIR, BASE_TEST_DIR, "points.png")
        condition_img = load_test_condition(map_path, points_path, IMG_SIZE, DEVICE)
        
        current_x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        start_time = time.perf_counter()
        for i in range(len(student_schedule)):
            t_student = student_schedule[i]
            t_next_student = student_schedule[i+1] if i < len(student_schedule) - 1 else -1
            current_x = ddim_step(model, current_x, t_student, t_next_student, alphas_cumprod, condition_img)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        total_inference_time += (end_time - start_time)
        
        viz_map = condition_img[:, 0:1, :, :].repeat(1, 3, 1, 1)
        viz_points = condition_img[:, 1:4, :, :]
        viz_pred = current_x
        
        combined = torch.cat([viz_map, viz_points, viz_pred], dim=3)
        combined = (combined + 1) / 2
        save_image(combined.cpu(), f"ema_result_test_s4.png", nrow=1, padding=2)
        print(f"  + Đã xong", end='\r')

    avg_time_ms = (total_inference_time) * 1000
    fps = 1000 / avg_time_ms
    print("\n--> Đã lưu toàn bộ ảnh tại thư mục ")
    print(f"========== THỐNG KÊ ==========")
    print(f"Tổng số mẫu đã test: {1} samples")
    print(f"Thời gian Inference TRUNG BÌNH (1 sample): {avg_time_ms:.2f} ms")
    print(f"FPS tương đương: {fps:.2f} frames/second")
    print(f"==============================")