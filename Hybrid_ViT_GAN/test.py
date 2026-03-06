import argparse, torch, os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from generator import ViTGenerator

def get_args():
    parser = argparse.ArgumentParser(description="Testing model ViT GAN after training")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path of checkpoint")
    parser.add_argument('--map_path', type=str, required=True, help="Path of map.png")
    parser.add_argument('--points_path', type=str, required=True, help="Path of points.png")
    parser.add_argument('--output_path', type=str, required=True, help="Path of result.png")
    return parser.parse_args()

def load_model(checkpoint_path, device):
    print(f"Using device: {device}")

    netG = ViTGenerator().to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Not found checkpoint at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG

def preprocess_input(map_path, points_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    convert2PILImage = transforms.ToPILImage()

    map_img = Image.open(map_path).convert("L")
    points_img = Image.open(points_path).convert("RGB")

    map_tensor = transform(map_img).to(device)
    points_tensor = transform(points_img).to(device)

    input_tensor = points_tensor * map_tensor
    input_img = convert2PILImage(input_tensor)

    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor, input_img

def test(args):
    convert2PILImage = transforms.ToPILImage()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        netG = load_model(args.checkpoint_path, DEVICE)
        input_tensor, input_img = preprocess_input(args.map_path, args.points_path, DEVICE)

        print(f"Processing....")

        with torch.no_grad():
            result_tensor = netG(input_tensor).squeeze().cpu()
            print(f"Min value: {result_tensor.min():.4f}, Max value: {result_tensor.max():.4f}, Mean: {result_tensor.mean():.4f}")
            threshold_output = (result_tensor > 0.95).float()

        result_img = convert2PILImage(result_tensor)
        result_img.save(args.output_path)
        print(f"Successfully saved result at {args.output_path}!")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Input map")
        plt.imshow(input_img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Promising Region")
        plt.imshow(result_img, cmap='gray')
        plt.axis('off')

        plt.show()

    except Exception as e:
        print(f"An error has occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    args = get_args()
    test(args)