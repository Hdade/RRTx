import torch, timm
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels * 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.PixelShuffle(2)
        )
    
    def forward(self, x):
        return self.block(x)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class ViTGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.encoder.head = nn.Identity()

        embed_dim = 192 

        self.decoderBlock1 = DecoderBlock(embed_dim, 256)
        self.decoderBlock2 = DecoderBlock(256, 128)
        self.decoderBlock3 = DecoderBlock(128, 64)
        self.decoderBlock4 = DecoderBlock(64, 32)

        self.bottleneck1 = BottleneckBlock(256 + embed_dim, 256) 
        self.bottleneck2 = BottleneckBlock(128 + embed_dim, 128)
        self.bottleneck3 = BottleneckBlock(64 + embed_dim, 64)

        self.refining_block = ResidualBlock(32)

        self.final_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.apply_custom_init()

    def apply_custom_init(self):
        new_modules = [
            self.decoderBlock1,
            self.decoderBlock2,
            self.decoderBlock3,
            self.decoderBlock4,
            self.bottleneck1,
            self.bottleneck2,
            self.bottleneck3,
            self.final_head,
            self.refining_block,
        ]

        for module in new_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def skip_connection(self, skip_tensor, target_size):
        x = F.interpolate(skip_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
        return x
    
    def forward(self, x):
        skips = self.encoder.get_intermediate_layers(x, n=(3, 6, 9, 11), reshape=True)
        x = self.skip_connection(skips[3], 14)
        
        x = self.decoderBlock1(x)
        res3 = self.skip_connection(skips[2], 28)
        x = torch.cat([x, res3], dim=1)
        x = self.bottleneck1(x)

        x = self.decoderBlock2(x)
        res2 = self.skip_connection(skips[1], 56)
        x = torch.cat([x, res2], dim=1)
        x = self.bottleneck2(x)

        x = self.decoderBlock3(x)
        res1 = self.skip_connection(skips[0], 112)
        x = torch.cat([x, res1], dim=1)
        x = self.bottleneck3(x)

        x = self.decoderBlock4(x)
        x = self.refining_block(x)
        x = self.final_head(x)
        return x