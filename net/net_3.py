import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from timm.layers import DropPath,trunc_normal_
import torch.fft

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch

class LocalChannelAttention3D(nn.Module):
    def __init__(self, in_channels, feature_map_size, kernel_size=3):
        super().__init__()


        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool3d(feature_map_size)

    def forward(self, x):
        N, C, D, H, W = x.shape

        # 先进行全局池化，得到每个通道的平均值
        att = self.GAP(x).reshape(N, C)  # 输出形状 (N, C, 1, 1, 1)
        # 对池化结果进行卷积，获得注意力权重
        att = att.unsqueeze(2)
        att = self.conv(att).sigmoid()
        att = att.reshape(N, C, 1, 1, 1)
        return (x * att)+x


class LocalSpatialAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # self.conv1x1_1 = nn.Conv3d(in_channels, num_reduced_channels, kernel_size=1, stride=1)
        self.conv1x1_1 = nn.Conv3d(int(in_channels * 3), int(in_channels * 3)//2, kernel_size=1, stride=1)
        self.conv1x1_2 = nn.Conv3d(int(in_channels * 3)//2, 1, kernel_size=1, stride=1)
        self.dilated_conv3x3 = nn.Conv3d(in_channels, in_channels, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv3d(in_channels, in_channels, 3, 1, padding=2, dilation=2)
        self.dilated_conv7x7 = nn.Conv3d(in_channels, in_channels, 3, 1, padding=3, dilation=3)
        self.bn3x3 = nn.BatchNorm3d(in_channels)
        self.bn5x5 = nn.BatchNorm3d(in_channels)
        self.bn7x7 = nn.BatchNorm3d(in_channels)

        self.relu = nn.ReLU()

    def forward(self, feature_maps):
        # att = self.conv1x1_1(feature_maps)
        d1 = self.relu(self.bn3x3(self.dilated_conv3x3(feature_maps)))  # 3x3卷积后加BN和ReLU
        d2 = self.relu(self.bn5x5(self.dilated_conv5x5(feature_maps)))  # 5x5卷积后加BN和ReLU
        d3 = self.relu(self.bn7x7(self.dilated_conv7x7(feature_maps)))  # 7x7卷积后加BN和ReLU
        att = torch.cat((d1, d2, d3), dim=1)
        att = self.conv1x1_1(att)
        att = self.conv1x1_2(att)
        return (feature_maps*att)+feature_maps


class GLAM3D(nn.Module):
    def __init__(self, in_channels, feature_map_size, kernel_size=3):
        super().__init__()

        self.local_channel_att = LocalChannelAttention3D(in_channels, feature_map_size, kernel_size)
        self.local_spatial_att = LocalSpatialAttention3D(in_channels)

        # 可训练的融合权重
        self.fusion_weights = nn.Parameter(torch.Tensor([0.333,0.333,0.333]))

    def forward(self, x):
        local_channel_att = self.local_channel_att(x)
        local_spatial_att = self.local_spatial_att(x)

        # (N, 1, C, D, H, W)
        x = x.unsqueeze(1)
        local_spatial_att = local_spatial_att.unsqueeze(1)
        local_channel_att = local_channel_att.unsqueeze(1)

        #(N, 3, C, D, H, W)
        all_feature_maps = torch.cat((x,  local_spatial_att,local_channel_att), dim=1)
        weights = self.fusion_weights.softmax(dim=-1).reshape(1, 3, 1, 1, 1, 1)
        fused_feature_maps = (all_feature_maps * weights).sum(dim=1)

        return fused_feature_maps


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()

        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))

        self.add_module('Attention',GLAM3D(in_channels=16,feature_map_size=(46,55,46),kernel_size=3))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(self.drop_rate)

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class DenseNet(nn.Module):

    def __init__(self,
                 n_input_channels=1,
                 no_max_pool=True,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 ):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(7, 7, 7),
                                    stride=(2, 2, 2),
                                    padding=(3, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))

        self.features = nn.Sequential(OrderedDict(self.features))


        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        return features

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, rank=None, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        rank = rank or in_features // 2


        self.u1 = nn.Linear(in_features, rank, bias=False)
        self.v1 = nn.Linear(rank, hidden_features, bias=True)

        # 低秩分解 W2: hidden_features -> out_features
        self.u2 = nn.Linear(hidden_features, rank, bias=False)
        self.v2 = nn.Linear(rank, out_features, bias=True)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.u1(x)
        x = self.v1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.u2(x)
        x = self.v2(x)
        x = self.drop(x)
        return x


# Global_Filter:[batch_size x patch_size6804 ×dim1000] ----> [batch_size x patch_size ×dim]
class Global_Filter(nn.Module):
    def __init__(self, h=9, w=11, d=5, dim=900):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(
            h, w, d, dim, 2, dtype=torch.float32) * 0.02)
        self.h = h
        self.w = w
        self.d = d
    def forward(self, x):
        B, N, C = x.shape
        # print("N",N)
        # print("C",C)
        # print("B",B)
        x = x.to(torch.float32)
        x = x.view(B, 9,11, 9, 900)
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfftn(x, s=(9,11,9), dim=(1, 2, 3), norm='ortho')
        x = x.reshape(B, N, C)
        return x




class Block(nn.Module):
    def __init__(self, dim=3750, mlp_ratio=2., drop=0.5, drop_path=0.6, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=9, w=11, d=5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = Global_Filter(dim=dim, h=h, w=w, d=d)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + \
            self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


# x:(batch size,1,182,218,182)-》(batch size,patch_dim=1000,num_patches=18×21×18)-》(batch size,6804,1000)
class PatchEmbed(nn.Module):
    #image to patch embedding
    def __init__(self, img_size=(182, 218, 182), patch_size=(10, 10, 10), num_classes=2, in_channels=1):
        super().__init__()
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] //
                                                        patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_dim = in_channels * patch_size[0]*patch_size[1]*patch_size[2]
        self.proj = nn.Conv3d(in_channels, self.patch_dim//5,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        N, C, H, W, D = x.size()
        # print(N)
        # print(C)
        # print(H,W,D)
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2],\
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class GFNet(nn.Module):
    def __init__(self, img_size=(182, 218, 182), patch_size=(10, 10, 10),
                 embed_dim=1000, num_classes=2, in_channels=1, drop_rate=0.5, depth=8, mlp_ratio=2.,
                 representation_size=None, uniform_drop=False, drop_path_rate=0.6, norm_layer=False, dropcls=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        h = 9
        w = 11
        d = 5
        if uniform_drop:
            print('using uniform droppath with expected rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            print('using linear droppath with expected rate', drop_path_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w, d=d)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(
            self.num_features, self.num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.Sequential(
        #     nn.Linear(self.embed_dim, 64),
        #     nn.Linear(64, num_classes)
        # )

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x

class DenseNetWithGFNet(nn.Module):
    def __init__(self,
                 n_input_channels=1,
                 no_max_pool=True,
                 growth_rate=8,
                 block_config=(3,),
                 num_init_features=12,
                 bn_size=2,
                 drop_rate=0.3,
                 patch_size=(5, 5, 5),
                 embed_dim=900,
                 num_classes=2,
                 depth=8,
                 mlp_ratio=2,
                 drop_path_rate=0.5,
                 dropcls=0.4):
        super().__init__()

        # DenseNet部分
        self.densenet = DenseNet(
            n_input_channels=n_input_channels,
            no_max_pool=no_max_pool,
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,

        )
        densenet_output_channels = 12 + sum(block_config[i] * growth_rate for i in range(len(block_config)))

        sample_input = torch.randn(1, n_input_channels, 182, 218, 182)
        with torch.no_grad():
            sample_output = self.densenet(sample_input)
        densenet_output_size = sample_output.shape[2:]
        # print("densenet_out:",densenet_output_size)
        self.gfnet = GFNet(
            img_size=densenet_output_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            in_channels=densenet_output_channels,
            drop_rate=dropcls,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x):
        x = self.densenet(x)
        x = self.gfnet(x)

        return x




