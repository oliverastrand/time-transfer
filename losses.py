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


class PerceptualLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.vgg16().features
        self.content_slices = [None]
        self.style_slices = []
        self.criteria = nn.MSELoss()
        
    def get_features(self, img):
        ret = []

        if self.content_slices[0] is None:
            ret.append(img)
            slices = self.content_slices[1:]
        else:
            slices = self.content_slices

        y = img
        for begin, end in zip([0]+slices, slices):
            y = self.features[begin:end](y)
            ret.append(y)
        
        y = img
        for begin, end in zip([0]+self.style_slices, self.style_slices):
            y = self.features[begin:end](y)
            gram = self.gram_matrix(y)
            ret.append(gram)
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

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

   
