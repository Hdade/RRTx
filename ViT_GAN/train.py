import os, torch, argparse
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import PathPlanningDataset
from generator import ViTGenerator
from discriminator import ViTDiscriminator
from criterion import GANLoss, WeightedL1Loss, get_weight_map

# -------Hyperparameters-------

# BATCH_SIZE = 8
# LEARNING_RATE = 2e-4
# BETA1, BETA2 = 0.5, 0.999
# EPOCHS = 100
# LAMBDA_L1 = 100
# LAMDBA_ADV = 1
# SAVE_DIR = "checkpoints"
# DATA_DIR = "datasets"

def get_args():
    parser = argparse.ArgumentParser(description="Train ViT-GAN for Path Planning")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--lr_G', type=float, default=2e-4, help="Learning Rate of Generator")
    parser.add_argument('--lr_D', type=float, default=1e-5, help="Learning Rate of Discriminator")
    parser.add_argument('--beta1', type=float, default=0.5, help="beta1 param in Adam Optimizer")
    parser.add_argument('--beta2', type=float, default=0.999, help="beta2 param in Adam Optimizer")
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='Weight of L1 Loss')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Weight of Adversarial Loss')
    parser.add_argument('--save_dir', type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument('--data_dir', type=str, default="datasets", help="Directory of datasets")
    parser.add_argument('--start_epoch', type=int, default=1, help="Start training from epoch")
    parser.add_argument('--end_epoch', type=int, default=100, help="End training at epoch")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory of previous checkpoint")
    return parser.parse_args()

def get_noisy_input(input_tensor, std=0.1):
    noise = torch.randn_like(input_tensor) * std
    return input_tensor + noise

def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = PathPlanningDataset(root_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    netG = ViTGenerator().to(DEVICE)
    netD = ViTDiscriminator().to(DEVICE)
    
    criterion_GAN = GANLoss(real_target=0.9, fake_target=0.1).to(DEVICE)
    criterion_L1 = WeightedL1Loss().to(DEVICE)

    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))

    start_epoch = args.start_epoch

    if start_epoch > 1:
        checkpoint_path = args.checkpoint_dir
        if os.path.exists(checkpoint_path):
            netG.load_state_dict(torch.load(checkpoint_path))
            netD.load_state_dict(torch.load(checkpoint_path.replace("netG", "netD")))
            print(f"Resuming training from epoch {start_epoch}...")
        else:
            print(f"Cannot find the checkpoint path!")

    for epoch in range(start_epoch, args.end_epoch + 1):
        netG.train()
        netD.train()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.end_epoch}")
        for i, (condition, target) in pbar:
            condition = condition.to(DEVICE)
            target = target.to(DEVICE)

            # -------Train Discriminator-------
            optimizer_D.zero_grad()

            noisy_condition = get_noisy_input(condition, 0.05)
            noisy_target = get_noisy_input(target, 0.05)

            pred_real = netD(noisy_condition, noisy_target)
            loss_D_real = criterion_GAN(pred_real, True)

            with torch.no_grad():
                fake_image = netG(condition)

            noisy_fake = get_noisy_input(fake_image.detach(), 0.05)

            pred_fake = netD(noisy_condition, noisy_fake)
            loss_D_fake = criterion_GAN(pred_fake, False)

            loss_D = 0.5 * (loss_D_fake + loss_D_real)
            loss_D.backward()
            optimizer_D.step()

            # -------Train Generator-------
            GENERATOR_STEPS = 1
            for __ in range(GENERATOR_STEPS):
                optimizer_G.zero_grad()

                fake_image_new = netG(condition)
                noisy_condition_for_G = get_noisy_input(condition, 0.05) 
                
                pred_fake_new = netD(noisy_condition_for_G, fake_image_new)
                loss_G_Adv = criterion_GAN(pred_fake_new, True)

                weights = get_weight_map(condition, target, DEVICE, penalty_weight=50.0) 
                loss_G_L1 = criterion_L1(fake_image_new, target, weights)

                loss_G = args.lambda_l1 * loss_G_L1 + args.lambda_adv * loss_G_Adv
                loss_G.backward()
                optimizer_G.step()

            # -------LOGGING-------
            pbar.set_postfix({
                'Loss_D': f"{loss_D.item():.4f}",
                'Loss_G': f"{loss_G.item():.4f}",
                'L1': f"{loss_G_L1.item():.4f}",
            })

        if epoch % 1 == 0:
            torch.save(netG.state_dict(), f"{args.save_dir}/netG_epoch_{epoch}.pth")
            torch.save(netD.state_dict(), f"{args.save_dir}/netD_epoch_{epoch}.pth")
            print(f"Saved model at epoch {epoch}!")

            netG.eval()
            with torch.no_grad():
                sample_input = condition[0].cpu() 
                sample_target = target[0].cpu()

                sample_output = netG(condition[0].unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

                sample_target_3c = sample_target.repeat(3, 1, 1)
                sample_output_3c = sample_output.repeat(3, 1, 1)

                combined_img = torch.cat((sample_input, sample_target_3c, sample_output_3c), dim=2)
                save_image(combined_img, f"{args.save_dir}/vis_epoch_{epoch}.png")
            
            netG.train()

if __name__ == "__main__":
    args = get_args()
    train(args)