import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, real_target=1.0, fake_target=0.0):
        super().__init__()
        
        self.register_buffer('real_label', torch.tensor(real_target))
        self.register_buffer('fake_label', torch.tensor(fake_target))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction, target_is_real):
        """
        prediction: output from discriminator
        target_is_real: True if we need to compare with real label otherwise False
        """
        target_tensor = self.get_target(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
    
class WeightedL1Loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.criterion = nn.L1Loss(reduction='none')

    def forward(self, prediction, target, weights=None):
        """
        prediction: image generated from Generator
        target: Ground truth
        weights: upscale weight for red/blue pixels in order 
                 to penalize loss not it ignore those small pixel
        """
        loss = self.criterion(prediction, target)

        if weights is not None:
            loss = loss * weights

        return loss.mean()
    
def get_weight_map(condition, target, device, penalty_weight=50.0):
    weights = torch.ones((condition.shape[0], 1, condition.shape[2], condition.shape[3]), device=device)
    path_mask = target > 0.5
    weights[path_mask] = penalty_weight
    start_goal_mask = (condition[:, 0:1, :, :] + condition[:, 2:3, :, :]) > 0.1
    weights[start_goal_mask] = penalty_weight
    return weights