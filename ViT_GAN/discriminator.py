import torch
import torch.nn as nn
import torch.nn.utils as utils

class ViTDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()

        self.model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
        )

        self.apply_custom_init()

    def apply_custom_init(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, condition_tensor, target_tensor):
        x = torch.cat([condition_tensor, target_tensor], dim=1)
        return self.model(x)
    
if __name__ == "__main__":
    D = ViTDiscriminator()

    dummy_condition = torch.randn(1, 3, 224, 224)
    dummy_target = torch.randn(1, 1, 224, 224)

    output = D(dummy_condition, dummy_target)
    print(f"Input shape: {(torch.cat([dummy_condition, dummy_target], dim=1)).shape}")
    print(f"Output shape: {output.shape}")