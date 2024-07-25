import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from net.STPG_G_MESE import STPG_G_MESE
from net.FD_MEE import FD_MEE

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Taskprompt(nn.Module):
    def __init__(self, in_dim, atom_num=32, atom_dim=256):
        super(Taskprompt, self).__init__()
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1))
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        logits = F.softmax(out, -1)
        out = logits @ self.dictionary
        out = self.act(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class mm(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(mm, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        out = self.project_out(out)
        return out

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class modulate1(nn.Module):
    def __init__(self):
        super(modulate1, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

class modulate2(nn.Module):
    def __init__(self, dim):
        super(modulate2, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))
        self.squential = nn.Sequential(nn.Conv2d(dim, dim//16, 1, bias=False),nn.ReLU(),nn.Conv2d(dim//16, dim, 1, bias=False))
    def forward(self, x):
        x1 = self.squential(self.avg(x))
        x2 = self.squential(self.max(x))
        x = x1 + x2
        x = F.sigmoid(x)
        return x

class modulate(nn.Module):
    def __init__(self, dim):
        super(modulate, self).__init__()
        self.modulate1 = modulate1()
        self.modulate2 = modulate2(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self, low, high):
        weight1 = self.modulate1(high)
        weight2 = self.modulate2(low)
        high2 = high * weight2
        low1 = low * weight1
        out = low1 + high2
        out = self.conv(out)
        return out

class Block(nn.Module):
    def __init__(self, dimensions, head_count, use_bias, input_dim=3):
        super(Block, self).__init__()
        self.initial_conv = nn.Conv2d(input_dim, dimensions, kernel_size=3, stride=1, padding=1, bias=False)
        self.aux_conv = nn.Conv2d(input_dim, dimensions, kernel_size=3, stride=1, padding=1, bias=False)
        self.score_generator = nn.Conv2d(2, 2, 7, padding=3)
        self.param_alpha = nn.Parameter(torch.zeros(dimensions, 1, 1))
        self.param_beta = nn.Parameter(torch.ones(dimensions, 1, 1))
        self.cross_layer_low = mm(dimensions, num_head=head_count, bias=use_bias)
        self.cross_layer_high = mm(dimensions, num_head=head_count, bias=use_bias)
        self.cross_layer_agg = mm(dimensions, num_head=head_count, bias=use_bias)
        self.freq_refinement = modulate(dimensions)
        self.rate_adaptive_conv = nn.Sequential(nn.Conv2d(dimensions, dimensions//8, 1, bias=False),nn.GELU(),nn.Conv2d(dimensions//8, 2, 1, bias=False),)

    def forward(self, x, y):
        _, _, height, width = y.size()
        x = F.interpolate(x, (height, width), mode='bilinear')
        feature_high, feature_low = self.fft(x)
        feature_high = self.cross_layer_low(feature_high, y)
        feature_low = self.cross_layer_high(feature_low, y)
        aggregate = self.freq_refinement(feature_low, feature_high)
        output = self.cross_layer_agg(y, aggregate)
        return output * self.param_alpha + y * self.param_beta

    def shift(self, x):
        batch, channels, height, width = x.shape
        return torch.roll(x, shifts=(int(height/2), int(width/2)), dims=(2,3))

    def unshift(self, x):
        batch, channels, height, width = x.shape
        return torch.roll(x, shifts=(-int(height/2), -int(width/2)), dims=(2,3))

    def fft(self, x, segments=128):
        x = self.aux_conv(x)
        masking = torch.zeros(x.shape).to(x.device)
        height, width = x.shape[-2:]
        avg_threshold = F.adaptive_avg_pool2d(x, 1)
        avg_threshold = self.rate_adaptive_conv(avg_threshold).sigmoid()

        for i in range(masking.shape[0]):
            h_seg = (height//segments * avg_threshold[i,0,:,:]).int()
            w_seg = (width//segments * avg_threshold[i,1,:,:]).int()
            masking[i, :, height//2-h_seg:height//2+h_seg, width//2-w_seg:width//2+w_seg] = 1
        transformed = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        transformed = self.shift(transformed)
        high_freq = transformed * (1 - masking)
        high_detail = self.unshift(high_freq)
        high_detail = torch.fft.ifft2(high_detail, norm='forward', dim=(-2,-1))
        high_detail = torch.abs(high_detail)
        low_freq = transformed * masking
        low_detail = self.unshift(low_freq)
        low_detail = torch.fft.ifft2(low_detail, norm='forward', dim=(-2,-1))
        low_detail = torch.abs(low_detail)
        return high_detail, low_detail


class IRmodel(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim = 48,num_blocks = [4,4,6,8],num_refinement_blocks = 4,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias'):
        super(IRmodel, self).__init__()
        atom_dim = 256
        atom_num = 32
        ffn_expansion_factor = 2.66
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.block1 = Block(dim*2**3, head_count=heads[2], use_bias=bias)
        self.block2 = Block(dim*2**2, head_count=heads[2], use_bias=bias)
        self.block3 = Block(dim*2**1, head_count=heads[2], use_bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.task_prompt = Taskprompt(in_dim=3, atom_num=atom_num, atom_dim=atom_dim)
        self.stpg_g_mese1 = STPG_G_MESE(atom_dim=atom_dim, dim=dim,ffn_expansion_factor=ffn_expansion_factor)
        self.stpg_g_mese2 = STPG_G_MESE(atom_dim=atom_dim, dim=int(dim * 2 ** 1),ffn_expansion_factor=ffn_expansion_factor)
        self.stpg_g_mese3 = STPG_G_MESE(atom_dim=atom_dim, dim=int(dim * 2 ** 2),ffn_expansion_factor=ffn_expansion_factor)
        self.fe_mee1 = FD_MEE(in_ch=dim * 2 ** 2, num_experts=5)
        self.fe_mee2 = FD_MEE(in_ch=dim * 2 ** 1, num_experts=5)
        self.fe_mee3 = FD_MEE(in_ch=dim * 2 ** 1 , num_experts=5)

    def forward(self, inp_img):
        task_prompt = self.task_prompt(inp_img)
        inp_enc_level1 = self.patch_embed(inp_img)
        task_harmonization_output1, loss_tmp = self.stpg_g_mese1(inp_enc_level1 , task_prompt)
        loss_importance = loss_tmp
        out_enc_level1 = self.encoder_level1(task_harmonization_output1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        task_harmonization_output2, loss_tmp = self.stpg_g_mese2(inp_enc_level2 , task_prompt)
        loss_importance = loss_importance + loss_tmp
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        task_harmonization_output3, loss_tmp = self.stpg_g_mese3(inp_enc_level3 , task_prompt)
        loss_importance = loss_importance + loss_tmp
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        latent = self.block1(inp_img, latent)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = self.fe_mee1(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec_level3 = self.block2(inp_img, out_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = self.fe_mee2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.block3(inp_img, out_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.fe_mee3(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        if self.training:
            return out_dec_level1, loss_importance
        else:
            return out_dec_level1