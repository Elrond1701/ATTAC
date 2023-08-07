from torch import nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import vit_small_patch16_224


class ViT_small_32(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        
        self.vit = timm.models.vision_transformer.vit_small_patch16_224(pretrained=pretrained)
        self.patch_embed = nn.Conv2d(3, self.vit.embed_dim, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))  # 修改patch_embed层
        self.fc = nn.Linear(self.vit.embed_dim, num_classes)
        self.head_var = 'fc'

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.vit(x)
        h = self.fc(x[:, 0, :])  # 取CLS标记位置对应的特征向量作为输出
        return h

def vit_small_32(num_out=100, pretrained=False):
    if pretrained:
        return ViT_small_32(num_out, pretrained)
    else:
        raise NotImplementedError


