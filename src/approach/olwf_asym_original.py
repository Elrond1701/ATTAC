import torch
import math
from copy import deepcopy
from argparse import ArgumentParser
import torch.nn.functional as F

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from networks.ovit import get_attention_list, start_rec, stop_rec
from einops import rearrange#, reduce, repeat


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, sparsefact=100.,  plast_mu=1, lamb=1, T=2, sym=False, distance_metric="JS", alpha=0.5, scale_factor=1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.sparsefact = sparsefact
        self.lamb = lamb
        self.T = T
        self.plast_mu = plast_mu
        self._task_size = 0
        self._n_classes = 0
        self._pod_spatial_factor = 3.
        self.sym = sym
        self.distance_metric=distance_metric
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.pool_along = "spatial"

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--sparsefact', default=100., type=float, required=False, help='add sparse attention regularization for asym loss')
        parser.add_argument('--sym', action='store_true', default=False, required=False,
                            help='Use symmetric version of the loss if given (default=%(default)s)')
        parser.add_argument('--plast_mu', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--distance_metric', default='JS', type=str, required=False,
                    help='The distance metric to use for the plasticity loss (default=%(default)s)')
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                        help='Alpha parameter for Renyi divergence (default=%(default)s)')
        parser.add_argument('--scale_factor', default=1.0, type=float, required=False,
                        help='Scale factor for Renyi divergence (default=%(default)s)')

        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
       
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def plasticity_loss(self, old_attention_list, attention_list):
        """ jensen shannon (JS) plasticity loss between the attention maps
            of the old model and the new model, we sum the mean JS for each layer.
            Tiny ViTs models have 12 layers: each layer has 3 heads, the attention map size is (197,197).
            you will have a len(attention_list) = 12
            and each element of the list is (batch_size,3,197,197)
            we compute the JS on the columns, after normalizing (transforming the columns in probabilities).
        """

        totloss = 0.
        for i in range(len(attention_list)):

            # reshape
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')

            # get rid of negative values
            p = torch.abs(p)
            q = torch.abs(q)

            # transform them in probabilities
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)

            # JS
            m = (1./2.) * (p + q)
            t1 = (1./2.) * (p * ((p / m)+1e-05).log()).sum(dim=1)
            t2 = (1./2.) * (q * ((q / m)+1e-05).log()).sum(dim=1)
            loss = t1 + t2

            # we sum the mean for each layer
            totloss += loss.mean()

            # print("已计算JS",loss)

        return totloss

    from sklearn.metrics.pairwise import cosine_similarity
    
    def plasticity_loss_cosine(self, old_attention_list, attention_list):
        totloss = 0.
        for i in range(len(attention_list)):
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            p = torch.abs(p)
            q = torch.abs(q)
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)
            loss = 1 - cosine_similarity(p, q)
            totloss += loss.mean()
        return totloss

    def plasticity_loss_L1norm(self, old_attention_list, attention_list):
        totloss = 0.
        for i in range(len(attention_list)):
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            p = torch.abs(p)
            q = torch.abs(q)
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)
            loss = torch.norm(p - q, 1)
            totloss += loss.mean()
        return totloss

    def plasticity_loss_mmi(self, old_attention_list, attention_list):
        totloss = 0.
        for i in range(len(attention_list)):
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            p = torch.abs(p)
            q = torch.abs(q)
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)
            mutual_info = torch.sum(p * torch.log(p / q))
            loss = mutual_info
            totloss += loss
            # print("mmi")
        return totloss

    def plasticity_loss_bhattacharyya(self, old_attention_list, attention_list):
        totloss = 0.
        for i in range(len(attention_list)):
            # 将旧的和新的注意力列表重塑为合适的形状
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            
            # 取绝对值并归一化，使得p和q都成为概率分布
            p = torch.abs(p)
            q = torch.abs(q)
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)
            
            # 计算Bhattacharyya距离
            loss = -torch.log(torch.sum(torch.sqrt(p * q)))
            
            # 累加所有的loss
            totloss += loss
        return totloss

    import torch
    from torch.nn.functional import kl_div, softmax
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import cosine
    from sklearn.metrics import mutual_info_score
#     Total Variation (TV) 距离：TV距离是一种度量两个概率分布之间的差异的方法，它的定义是两个概率分布的绝对差的一半的最大值。TV距离的计算复杂度也是线性的。

# Hellinger距离：Hellinger距离是一种衡量两个概率分布之间的差异的方法，它的定义是两个概率分布的平方根之差的平方的一半。Hellinger距离的计算复杂度也是线性的。

# Rényi 散度：Rényi散度是一种泛化的KL散度，它包含了一系列的度量，包括KL散度和JS散度。Rényi散度的计算复杂度与JS散度相同，都是线性的。

# Mutual Information (MI)：MI是一种衡量两个随机变量之间的依赖性的度量，它的定义是两个随机变量的联合分布和它们各自的边缘分布的KL散度。MI的计算复杂度与JS散度相同，都是线性的。
    def plasticity_loss_total_variation(self, old_attention_list, attention_list):
        totloss = 0.
        for i in range(len(attention_list)):
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            p = torch.abs(p)
            q = torch.abs(q)
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)
            loss = torch.abs(p - q).sum() / 2
            totloss += loss.mean()
        return totloss
    
    def plasticity_loss_hellinger(self, old_attention_list, attention_list):
        totloss = 0.
        for i in range(len(attention_list)):
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            p = torch.abs(p)
            q = torch.abs(q)
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)
            loss = torch.sqrt(1 - torch.sqrt(p * q).sum())
            totloss += loss.mean()
        return totloss

    def plasticity_loss_renyi(self, old_attention_list, attention_list):
        totloss = 0.
        for i in range(len(attention_list)):
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            p = torch.abs(p) * self.scale_factor
            q = torch.abs(q) * self.scale_factor
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)
            loss = 1 / (self.alpha - 1) * torch.log((p ** self.alpha * q ** (1 - self.alpha)).sum())
            totloss += loss.mean()
        return totloss

    # def plasticity_loss_renyi(self, old_attention_list, attention_list, alpha=0.5):
    #     totloss = 0.
    #     for i in range(len(attention_list)):
    #         p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
    #         q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
    #         p = torch.abs(p)
    #         q = torch.abs(q)
    #         p /=  p.sum(dim=1).unsqueeze(1)
    #         q /= q.sum(dim=1).unsqueeze(1)
    #         loss = 1 / (alpha - 1) * torch.log((p ** alpha * q ** (1 - alpha)).sum())
    #         totloss += loss.mean()
    #     return totloss
    
    def plasticity_loss_renyi_ATTALL(self, old_attention_list, attention_list):
        # 将所有层的注意力图合并到一起
        old_attention_all = torch.cat(old_attention_list, dim=1)
        attention_all = torch.cat(attention_list, dim=1)

        # 计算Rényi散度
        p = rearrange(old_attention_all.view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
        q = rearrange(attention_all.view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
        p = torch.abs(p)
        q = torch.abs(q)
        p /=  p.sum(dim=1).unsqueeze(1)
        q /= q.sum(dim=1).unsqueeze(1)
        loss = 1 / (self.alpha - 1) * torch.log((p ** self.alpha * q ** (1 - self.alpha)).sum())

        return loss.mean()


    
    def permissive_relu(self, att_diff, asym_choice):
        relu_out_ = asym_choice(att_diff)
        penalty_factor = math.log(math.sqrt(
                    self._n_classes / self._task_size
                ))
        scaled_att_diff = torch.abs(att_diff) * penalty_factor
        # scaled_att_diff = torch.abs(att_diff) / 2.0 # make the negative values go smaller after abs() so that they are penalized less
        zero_relu_indices = relu_out_ == 0
        relu_out = relu_out_.clone()
        relu_out[zero_relu_indices] = scaled_att_diff[zero_relu_indices]
        return relu_out

 
    # def pod(self,
    #     list_attentions_a,
    #     list_attentions_b,
    #     collapse_channels="spatial",
    #     normalize=True
    # ):
    #     """Pooled Output Distillation.
    #     Reference:
    #         * Douillard et al.
    #         Small Task Incremental Learning.
    #         arXiv 2020.
    #     :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    #     :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    #     :param collapse_channels: How to pool the channels.
    #     :param memory_flags: Integer flags denoting exemplars.
    #     :param only_old: Only apply loss to exemplars.
    #     :return: A float scalar loss.
    #     """

    #     loss = torch.tensor(0.).to(self.device)
    #     for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
    #         # shape of (b, n, w, h)
    #         assert a.shape == b.shape, (a.shape, b.shape)

    #         a = torch.pow(a, 2)
    #         b = torch.pow(b, 2)

    #         if collapse_channels == "channels":
    #             a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
    #             b = b.sum(dim=1).view(b.shape[0], -1)
    #         elif collapse_channels == "width":
    #             a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
    #             b = b.sum(dim=2).view(b.shape[0], -1)
    #         elif collapse_channels == "height":
    #             a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
    #             b = b.sum(dim=3).view(b.shape[0], -1)
    #         elif collapse_channels == "gap":
    #             a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
    #             b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
    #         elif collapse_channels == "spatial":
    #             a_h = a.sum(dim=3).view(a.shape[0], -1)
    #             b_h = b.sum(dim=3).view(b.shape[0], -1)
    #             a_w = a.sum(dim=2).view(a.shape[0], -1)
    #             b_w = b.sum(dim=2).view(b.shape[0], -1)
    #             a = torch.cat([a_h, a_w], dim=-1)
    #             b = torch.cat([b_h, b_w], dim=-1)
    #         elif collapse_channels == 'pixel':
    #             pass
    #         else:
    #             raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

    #         if not self.sym:
    #             asym_choice = torch.nn.ReLU(inplace=True)
    #             if normalize:
    #                 a = F.normalize(a, dim=1, p=2)
    #                 b = F.normalize(b, dim=1, p=2)
    #             diff = a-b
    #             relu_out = asym_choice(diff)  
    #             layer_loss = torch.mean(torch.frobenius_norm(relu_out, dim=-1))
    #         else:
    #             if normalize:
    #                 a = F.normalize(a, dim=1, p=2)
    #                 b = F.normalize(b, dim=1, p=2)
    #             layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
    #         loss += layer_loss 

    #     return loss / len(list_attentions_a)


    def asymmetric_headwise_loss(self, old_attention_list, attention_list, collapse_channels = "spatial"):
        totloss = torch.tensor(0.).to(self.device)
        
        asym_loss = "relu"
        layers_to_pool = range(len(old_attention_list)) if not self.int_layer else [each-1 for each in self.pool_layers]

        # for i in layers_to_pool:
            # p = rearrange(old_attention_list[i].to(self.device), 's h b w -> h s b w') # rearrange to make head as the first dimension
            # q = rearrange(attention_list[i].to(self.device), 's h b w -> h s b w')

        for idx, (a, b) in enumerate(zip(old_attention_list, attention_list)):
            # each element is now of shape (96, 197, 197)
            assert a.shape == b.shape, 'Shape error'
            if self.sym:
                a = torch.pow(a, 2)
                b = torch.pow(b, 2)
            if collapse_channels == "spatial":
                a_h = a.sum(dim=2).view(a.shape[0], -1)  # [bs, w]
                b_h = b.sum(dim=2).view(b.shape[0], -1)  # [bs, w]
                a_w = a.sum(dim=3).view(a.shape[0], -1)  # [bs, h]
                b_w = b.sum(dim=3).view(b.shape[0], -1)  # [bs, h]
                a = torch.cat([a_h, a_w], dim=-1) # concatenates two [96, 197] to give [96, 394], dim = -1 does concatenation along the last axis
                b = torch.cat([b_h, b_w], dim=-1)
            elif collapse_channels == "gap":
                # compute avg pool2d over each 32x32 image to reduce the dimension to 1x1
                a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0] # [..., 0, 0] preserves only the [0][0]th element of last two dimensions, i.e., [96, 197, 197] into [96], since 197x197 reduced to 1x1 and pooled together
                b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
            elif collapse_channels == "width":
                a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * h)
                b = b.sum(dim=3).view(b.shape[0], -1)
            elif collapse_channels == "height":
                a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * w)
                b = b.sum(dim=2).view(b.shape[0], -1)
            elif collapse_channels == 'pixel':
                pass

            distance_loss_weight = self.pod_spatial_factor if self.use_pod_factor else self.plast_mu
            if not self.sym:
                asym_choice = torch.nn.LeakyReLU(inplace=True) if asym_loss == "leaky_relu" else \
                                    torch.nn.ELU(inplace=True) if asym_loss == "elu" else torch.nn.ReLU(inplace=True)
                if self.after_norm:
                    a = F.normalize(a, dim=1, p=2)
                    b = F.normalize(b, dim=1, p=2)

                    if self.reverse_relu:
                        diff = b-a 
                    else:
                        diff = a-b
                    if self.perm_relu:
                        relu_out = self.permissive_relu(diff, asym_choice)
                    else:
                        relu_out = asym_choice(diff)                        
                    layer_loss = torch.mean(torch.frobenius_norm(relu_out, dim=-1)) * distance_loss_weight
                else:
                    
                    if self.reverse_relu:
                        diff = b-a 
                    else:
                        diff = a-b
                    if self.perm_relu:
                        relu_out = self.permissive_relu(diff, asym_choice)
                    else:
                        relu_out = asym_choice(diff)
                    # layer_loss = torch.mean(F.normalize(relu_out, dim=1, p=2)) # (a) good mu for this = 10
                    layer_loss = torch.mean(torch.frobenius_norm(F.normalize(relu_out, dim=1, p=2))) / 100.0 # (d) works but only after /100
                    layer_loss = layer_loss * distance_loss_weight
                    
            else:
                a = F.normalize(a, dim=1, p=2)
                b = F.normalize(b, dim=1, p=2)
                layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1)) * distance_loss_weight # right now the loss is symmetric, i.e., the new model is told to attend to the same region as the old model
            totloss += layer_loss

            if self.sparse:
                attention_sparsity_term = torch.norm(torch.abs(b))/ 10. #self.sparse_factor
                attention_sparsity_term = attention_sparsity_term * self.sparse_factor
                # if attention_sparsity_term > 0.0:
                #     sparse_loss_magnitude = math.floor(math.log10(attention_sparsity_term))
                #     attention_sparsity_term = attention_sparsity_term / math.pow(10, sparse_loss_magnitude+1)
                totloss += attention_sparsity_term

            # totloss = totloss / len(p)
        if self.sparse:
            totloss = totloss / (2 * len(layers_to_pool))
        else:
            totloss = totloss / len(layers_to_pool)
        return totloss

    
    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
            
        is_printed = 0  # 设置初始值为0
        for images, targets in trn_loader:
            loss = 0.
            plastic_loss = 0.

            # Forward old model
            targets_old = None
            if t > 0:
                self.model_old.to(self.device)
                start_rec()
                targets_old = self.model_old(images.to(self.device))
                stop_rec()

                old_attention_list = get_attention_list()
                # print(old_attention_list)

            # Forward current model
            start_rec()
            outputs = self.model(images.to(self.device))
            stop_rec()

            attention_list = get_attention_list()
            


            #     """ Headwise asymmetric loss (4 possible settings):
            #     for symmetric version of the loss, set asymmetric_loss = False
            #     for simple asym version, set asymmetric_loss = True , i.e., sparse_reg = None by default)
            #     for asym version with sparse attention with mean of |b|, set sparse_reg = 'mean'
            #     for asym version with sparse attention with norm of |b|, set sparse_reg = 'norm'
            #     """
            #     # pod_spatial_factor = self._pod_spatial_factor * math.sqrt(
            #     #     self._n_classes / self._task_size
            #     # )
            #     # plastic_loss += self.pod(old_attention_list, attention_list) * self.plast_mu * pod_spatial_factor
            if t > 0:
                plastic_loss += self.asymmetric_headwise_loss(old_attention_list, attention_list, collapse_channels=self.pool_along) 
                if self.distance_metric == 'JS':
                    plastic_loss += self.plasticity_loss(old_attention_list, attention_list)*self.plast_mu
                elif self.distance_metric == 'cosine':
                    plastic_loss += self.plasticity_loss_cosine(old_attention_list, attention_list)*self.plast_mu
                elif self.distance_metric == 'L1norm':
                    plastic_loss += self.plasticity_loss_L1norm(old_attention_list, attention_list)*self.plast_mu
                elif self.distance_metric == 'mmi':
                    plastic_loss += self.plasticity_loss_mmi(old_attention_list, attention_list)*self.plast_mu
                elif self.distance_metric == 'renyi':
                    plastic_loss += self.plasticity_loss_renyi(old_attention_list, attention_list)*self.plast_mu
                elif self.distance_metric == 'renyi_ALL':
                    plastic_loss += self.plasticity_loss_renyi_ATTALL(old_attention_list, attention_list)*self.plast_mu                    
                elif self.distance_metric == 'hellinger':
                    plastic_loss += self.plasticity_loss_hellinger(old_attention_list, attention_list)*self.plast_mu
                elif self.distance_metric == 'variation':
                    plastic_loss += self.plasticity_loss_total_variation(old_attention_list, attention_list)*self.plast_mu
                elif self.distance_metric == 'bhattacharyya':
                    plastic_loss += self.plasticity_loss_bhattacharyya(old_attention_list, attention_list)*self.plast_mu
                else:
                    raise ValueError(f'Invalid distance metric: {self.distance_metric}')
            loss += self.criterion(t, outputs, targets.to(self.device), targets_old)
            if is_printed <3:  # 如果还未打印过
                print(f"这是我编写的程序[Task {t}] l:{loss:.3f} p:{plastic_loss:.3f}")
                print(plastic_loss)
                is_printed = is_printed + 1  # 标记已打印过
            loss += plastic_loss


            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()



    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce
   
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        # if t > 0:
        #     # Knowledge distillation loss for all previous tasks
        #     loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
        #                                            torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])



