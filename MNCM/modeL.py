from typing import Callable, List, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
import math


def _make_divisible(ch, divisor=8, min_ch=None):
    """确保通道数可被8整除"""
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    """卷积+BN+激活函数模块"""

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    """原始 SE 模块"""

    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class CPCA(nn.Module):
    """通道与像素上下文聚合模块"""

    def __init__(self, in_channels: int, reduction: int = 4, dilation: int = 1):
        super().__init__()
        self.channel_fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.channel_fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.pixel_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # 通道注意力
        channel_attn = F.adaptive_avg_pool2d(x, (1, 1))
        channel_attn = self.channel_fc1(channel_attn)
        channel_attn = self.relu(channel_attn)
        channel_attn = self.channel_fc2(channel_attn)
        channel_attn = self.sigmoid(channel_attn)  # [B, C, 1, 1]

        # 像素注意力
        pixel_attn = self.pixel_conv(x)
        pixel_attn = self.sigmoid(pixel_attn)  # [B, 1, H, W]

        return x * (channel_attn * pixel_attn)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块（适用于空间维度）"""

    def __init__(self, in_channels, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子

        # 生成Q/K/V的线性层（通道维度不变，保持空间维度处理）
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 形状：[B, num_heads, N, C_head]，N=H*W

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力分数
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)  # 恢复空间维度
        x = self.proj(x)
        x = self.proj_drop(x)
        return x + x  # 残差连接（与输入相加）


class InvertedResidualConfig:
    """倒置残差块配置类（支持自注意力和CPCA）"""

    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 use_cpca: bool = False,
                 use_self_attn: bool = False,  # 新增自注意力标志位
                 width_multi: float = 1.0):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.use_cpca = use_cpca  # 启用CPCA
        self.use_self_attn = use_self_attn  # 启用自注意力

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    """带自注意力和CPCA的倒置残差块"""

    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super().__init__()
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        layers = []

        # 扩展层
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c, cnf.expanded_c, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # 深度卷积层
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel,
                                       stride=cnf.stride, groups=cnf.expanded_c,
                                       norm_layer=norm_layer, activation_layer=activation_layer))

        # 添加注意力模块（CPCA和自注意力可共存）
        if cnf.use_cpca:
            layers.append(CPCA(cnf.expanded_c))
        if cnf.use_self_attn:
            layers.append(MultiHeadSelfAttention(cnf.expanded_c))  # 添加自注意力模块

        # 投影层
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x) if self.use_res_connect else self.block(x)


class MobileNetV3(nn.Module):
    """支持CPCA和自注意力的MobileNetV3"""

    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # 构建特征提取层
        layers = [
            ConvBNActivation(3, inverted_residual_setting[0].input_c, kernel_size=3, stride=2,
                             norm_layer=norm_layer, activation_layer=nn.Hardswish)
        ]
        layers.extend([block(cnf, norm_layer) for cnf in inverted_residual_setting])
        layers.append(ConvBNActivation(
            inverted_residual_setting[-1].out_c, 6 * inverted_residual_setting[-1].out_c,
            kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish
        ))
        self.features = nn.Sequential(*layers)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(6 * inverted_residual_setting[-1].out_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


# ---------------------- 配置函数（以Large版本为例） ---------------------- #
def mobilenet_v3_large_with_attentions(num_classes: int = 1000,
                                       cpca_layers: List[int] = None,
                                       self_attn_layers: List[int] = None) -> MobileNetV3:
    """同时启用CPCA和自注意力的Large版本配置"""
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    cpca_layers = cpca_layers or [3, 4, 5]  # CPCA作用层
    self_attn_layers = self_attn_layers or [10, 11, 12]  # 自注意力作用层

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride, use_cpca, use_self_attn
        bneck_conf(16, 3, 16, 16, False, "RE", 1, False, False),  # 0
        bneck_conf(16, 3, 64, 24, False, "RE", 2, False, False),  # 1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, False, False),  # 2
        bneck_conf(24, 5, 72, 40, True, "RE", 2, 3 in cpca_layers, False),  # 3（CPCA）
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 4 in cpca_layers, False),  # 4（CPCA）
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 5 in cpca_layers, False),  # 5（CPCA）
        bneck_conf(40, 3, 240, 80, False, "HS", 2, False, False),  # 6
        bneck_conf(80, 3, 200, 80, False, "HS", 1, False, False),  # 7
        bneck_conf(80, 3, 184, 80, False, "HS", 1, False, False),  # 8
        bneck_conf(80, 3, 184, 80, False, "HS", 1, False, False),  # 9
        bneck_conf(80, 3, 480, 112, True, "HS", 1, False, 10 in self_attn_layers),  # 10（自注意力）
        bneck_conf(112, 3, 672, 112, True, "HS", 1, False, 11 in self_attn_layers),  # 11（自注意力）
        bneck_conf(112, 5, 672, 160, True, "HS", 2, False, 12 in self_attn_layers),  # 12（自注意力）
        bneck_conf(160, 5, 960, 160, True, "HS", 1, False, False),  # 13
        bneck_conf(160, 5, 960, 160, True, "HS", 1, False, False),  # 14
    ]
    last_channel = InvertedResidualConfig.adjust_channels(1280, width_multi)
    return MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        num_classes=num_classes
    )


# ---------------------- 示例用法 ---------------------- #
if __name__ == "__main__":
    # 创建同时启用CPCA和自注意力的模型（CPCA在层3-5，自注意力在层10-12）
    model = mobilenet_v3_large_with_attentions(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")  # 应输出 torch.Size([2, 1000])
    print("模型中已成功加入CPCA和自注意力机制并在指定层启用")