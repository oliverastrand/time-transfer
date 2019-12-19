import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

class SpatialLoss(nn.Module):

    def __init__(self, features, slices):
        super().__init__()
        self.features = features
        self.slices = slices
        self.criteria = nn.MSELoss()

    def get_features(self, img):
        ret = []

        # Check if we need loss before first layer
        if self.slices[0] is None:
            ret.append(img)
            slices = self.slices[1:]
        else:
            slices = self.slices

        y = img
        for begin, end in zip([0]+slices, slices):
            y = self.features[begin:end](y)
            ret.append(y)

        return ret
    
    def forward(self, output, target):

        out_features = self.get_features(output)
        target_features = self.get_features(target)

        loss = 0

        for x_f, y_f in zip(out_features, target_features):
            loss += self.criteria(x_f, y_f)

        return loss

class StyleLoss(SpatialLoss):

    def get_features(self, img):
        ret = []

        y = img
        for begin, end in zip([0]+self.slices,self.slices):
            y = self.features[begin:end](y)
            gram = self.gram_matrix(y)
            ret.append(y)

        return ret

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.vgg16(pretrained=True).features
        self.content_slices = [3, 8]
        self.style_slices = [8, 15, 22]

        #self.content_loss = SpatialLoss(self.features, self.content_slices)
        self.style_loss = StyleLoss(self.features, self.style_slices)
        self.criteria = nn.MSELoss()
        

    def forward(self, output, target):
        output = F.interpolate(output, size=224, mode='bilinear')
        target = F.interpolate(target, size=224, mode='bilinear')

        loss = 0
        #loss += self.content_loss(output, target)
        loss += self.style_loss(output, target)

        return loss


   
