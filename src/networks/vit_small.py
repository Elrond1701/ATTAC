from torch import nn
import torch.utils.model_zoo as model_zoo
import timm

class ViT_small(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
#vit_tiny_patch16_224
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        # if pretrained:
        #     state_dict = model_zoo.load_url("https://github.com/google-research/vision_transformer/releases/download/v1_0/vit_small_patch16_224-c9c6f14d.pth")
        #     self.vit.load_state_dict(state_dict)

        self.fc = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        self.head_var = 'fc'

    def forward(self, x):
        h = self.fc(self.vit(x))
        return h

def vit_small(num_out=100, pretrained=False):
    if pretrained:
        return ViT_small(num_out, pretrained)
    else:
        raise NotImplementedError







# from torch import nn
# import torch.nn.functional as F
# from .vit_original import VisionTransformer, _load_weights
# import timm

# class ViT_small(nn.Module):

#     # def __init__(self, num_classes=100, pretrained=False):
#     #     super().__init__()

#     #     filename = '/hy-tmp/continual_learning_with_vit/src/networks/pretrained_weights/small-224.npz'

#     #     self.vit = VisionTransformer(embed_dim=384, num_heads=6, num_classes=0)
#     #     if pretrained:
#     #         _load_weights(model=self.vit, checkpoint_path=filename)

#     #     self.fc = nn.Linear(in_features=384, out_features=num_classes, bias=True)
#     #     self.head_var = 'fc'

#     # def forward(self, x):
#     #     h = self.fc(self.vit(x))
#     #     return h
#     def __init__(self, num_classes=100, pretrained=True):
#         super().__init__()

#         filename = '/hy-tmp/continual_learning_with_vit/src/networks/pretrained_weights/augreg_Ti_16-i1k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

#         if pretrained:
#             self.vit = timm.create_model('vit_small_patch16_224', pretrained=False)
#             self.vit.default_cfg = None  # Set default_cfg attribute to None
#             self.vit.load_state_dict(torch.load(filename))
#         else:
#             self.vit = timm.create_model('vit_small_patch16_224', pretrained=False)

#         self.fc = nn.Linear(in_features=384, out_features=num_classes, bias=True)
#         self.head_var = 'fc'

#     def forward(self, x):
#         h = self.fc(self.vit(x))
#         return h

# def vit_small(num_out=100, pretrained=False):
#     # if pretrained:
#     #     return ViT_small(num_out, pretrained)
#     # else:
#     #     raise NotImplementedError
#     # assert 1==0, "you should not be here :/"
#     if pretrained:
#         return ViT_small(num_out, pretrained)
#     else:
#         raise NotImplementedError