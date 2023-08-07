from torch import nn
import torch.nn.functional as F
import timm


class Timm_ViT_Small(nn.Module):

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()

        filename = '/hy-tmp/continual_learning_with_vit/src/networks/pretrained_weights/augreg_Ti_16-i1k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        if pretrained:
            self.vit = timm.create_model('vit_small_patch16_224', pretrained=False)
            self.vit.load_state_dict(torch.load(filename))
        else:
            self.vit = timm.create_model('vit_small_patch16_224', pretrained=False)

        self.fc = nn.Linear(in_features=384, out_features=num_classes, bias=True)
        self.head_var = 'fc'

    def forward(self, x):
        h = self.fc(self.vit(x))
        return h


def Timm_ViT_Small(num_out=100, pretrained=False):
    if pretrained:
        return Timm_ViT_Small(num_out, pretrained)
    else:
        raise NotImplementedError
