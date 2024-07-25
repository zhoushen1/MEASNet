import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class FD(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(FD, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,self.kernel_size ** 2, h * w)
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        high = identity_input - low

        return low,high

def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x

class MESE(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 use_shuffle: bool = False,
                 lr_space: str = "linear",
                 recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0)
        )
        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )
        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True),
            nn.GELU())
        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")
        self.moe_layer = Layer(
            experts=[EL(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)],
            wet=WET(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x
        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)
        x = self.conv_2(x)
        k = self.calibrate(k)
        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x

class Layer(nn.Module):
    def __init__(self, experts: List[nn.Module], wet: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.wet = wet
        self.num_expert = num_expert
    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.wet(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        out = inputs.clone()
        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i:i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i:i + 1, None, None]
        return out

class EL(nn.Module):
    def __init__(self,
                 in_ch: int,
                 low_dim: int, ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x
        x = self.conv_3(x)
        return x

class WET(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

class StripedConv2d(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, self.kernel_size), padding=(0, self.padding),
                      groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(self.kernel_size, 1), padding=(self.padding, 0),
                      groups=in_ch if depthwise else 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MEE(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 lr_space: int = 1,
                 recursive: int = 2,
                 use_shuffle: bool = False, ):
        super().__init__()
        lr_space_mapping = {1: "linear", 2: "exp", 3: "double"}
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = MESE(in_ch=in_ch, num_experts=num_experts, topk=topk, use_shuffle=use_shuffle,
                              recursive=recursive, lr_space=lr_space_mapping.get(lr_space, "linear"))
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        return x

class FD_MEE(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 lr_space: int = 1,
                 topk: int = 2,
                 recursive: int = 2,
                 use_shuffle: bool = False):
        super().__init__()
        self.spilit = FD(inchannels=in_ch)
        self.high_block = MEE(in_ch=in_ch,
                               num_experts=num_experts,
                               use_shuffle=use_shuffle,
                               lr_space=lr_space,
                               topk=topk,
                               recursive=recursive)
        self.low_block = MEE(in_ch=in_ch,
                               num_experts=num_experts,
                               use_shuffle=use_shuffle,
                               lr_space=lr_space,
                               topk=topk,
                               recursive=recursive)
        self.proj = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high =  self.spilit(x)
        high = self.high_block(high)
        low = self.low_block(low)
        out = torch.cat([high, low], dim=1)
        out = self.proj(out)
        return out


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == "__main__":
    # from thop import profile
    model = ContentMoE(in_ch=48, num_experts=5)
    model.eval()
    print("params", count_param(model))
    inputs1 = torch.randn(1, 48, 128, 128)
    output=model(inputs1)