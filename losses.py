import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.vgg16().features
        self.slices = [3, 8, 15, 22]
        self.criteria = nn.MSELoss()
        
    def get_features(self, img):
        ret = []
        y = img
        for begin, end in zip([0]+self.slices, self.slices):
            y = self.features[begin:end](y)
            ret.append(y)
        return ret

    def forward(self, output, target):
        output = F.interpolate(output, size=224)
        target = F.interpolate(target, size=224)
        
        out_features = self.get_features(output)
        target_features = self.get_features(target)

        loss = 0

        for x_f, y_f in zip(out_features, target_features):
            loss += self.criteria(x_f, y_f)

        return loss


