import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .generator import ViTGenerator

class GANInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.netG = ViTGenerator().to(self.device)
        
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
        self.netG.load_state_dict(checkpoint)
        self.netG.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, map_arr, points_arr):
        map_pil = Image.fromarray(map_arr.astype('uint8')).convert("L")
        points_pil = Image.fromarray(points_arr.astype('uint8')).convert("RGB")
        
        map_tensor = self.transform(map_pil).to(self.device)
        points_tensor = self.transform(points_pil).to(self.device)
        
        input_tensor = points_tensor * map_tensor 
        input_tensor = input_tensor.unsqueeze(0) 

        with torch.no_grad():
            output = self.netG(input_tensor)
            
        heatmap = output.squeeze().cpu().numpy()
        return heatmap