import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MonaOp(nn.Module):
    """
    双输入时序融合版 MonaOp，用于捕获变化差异与结构共性
    """
    def __init__(self, in_features):
        super().__init__()

        # 多尺度深度卷积：各自输入先感知自身结构
        self.conv1_x = nn.Conv2d(in_features, in_features,
                                 kernel_size=3, padding=1, groups=in_features)
        self.conv2_x = nn.Conv2d(in_features, in_features,
                                 kernel_size=5, padding=2, groups=in_features)
        self.conv3_x = nn.Conv2d(in_features, in_features,
                                 kernel_size=7, padding=3, groups=in_features)

        self.conv1_y = nn.Conv2d(in_features, in_features,
                                 kernel_size=3, padding=1, groups=in_features)
        self.conv2_y = nn.Conv2d(in_features, in_features,
                                 kernel_size=5, padding=2, groups=in_features)
        self.conv3_y = nn.Conv2d(in_features, in_features,
                                 kernel_size=7, padding=3, groups=in_features)

        # 差异信息与共同信息融合
        self.fuse = nn.Conv2d(in_features * 2, in_features,
                              kernel_size=1, bias=False)

        # 融合后恢复结构表达能力
        self.projector = nn.Conv2d(in_features, in_features,
                                   kernel_size=1, bias=False)

    def forward(self, x, y):
        # --- 多尺度结构编码 ---
        x_ms = (self.conv1_x(x) + self.conv2_x(x) + self.conv3_x(x)) / 3.0 + x
        y_ms = (self.conv1_y(y) + self.conv2_y(y) + self.conv3_y(y)) / 3.0 + y

 
        # --- 时序融合 ---
        fused = torch.cat([x_ms, y_ms], dim=1)  # 维度 = 2*C
        fused = self.fuse(fused)

        # --- 输出两路时序语义增强特征 ---
        out_x = self.projector(fused + x) + x_ms  # 加回自身结构
        out_y = self.projector(fused + y) + y_ms

        return out_x, out_y


class CrossDiffFusion(nn.Module):
    def __init__(self, dim_in, dim_mid, act_layer=nn.GELU):
        super().__init__()
        self.downx = nn.Linear(dim_in, dim_mid)
        self.actx = nn.GELU()
        self.downy =  nn.Linear(dim_in, dim_mid)
        self.acty = nn.GELU()

        # 差分卷积：计算互补信息
        self.diff_conv = nn.Conv2d(dim_mid, dim_mid, kernel_size=3, padding=1, bias=False)

        # 融合后的通道还原
        self.fuse_conv = nn.Conv2d(dim_mid * 2, dim_mid, kernel_size=1, bias=False)
        self.adapter_conv1 = MonaOp(dim_mid)
        self.adapter_conv2 = MonaOp(dim_mid)
        # 最终上采样（或保留原上采样模块）
        self.actallx = nn.GELU()
        self.actally = nn.GELU()
        self.final_act = act_layer()
        self.upx = nn.Linear(dim_mid, dim_in) 
        self.upy = nn.Linear(dim_mid, dim_in) 
    def forward(self, x_rgbb, x_tt):
        """
        x_rgbb, x_tt: [B, C, H, W]
        """
        # Step 1: 通道降维
        rgb_low = self.downx(x_rgbb)  # [B, mid, H, W]
        t_low = self.downy(x_tt)      # [B, mid, H, W]
        rgb_low = self.actx(rgb_low)
        t_low = self.acty(t_low)

 

        # Step 3: 融合（cat + 1x1 conv 还原）
        rgb_fuse = rgb_low.permute(0, 3, 1, 2).contiguous()
        t_fuse = t_low.permute(0, 3, 1, 2).contiguous()

        rgb_out,t_out = self.adapter_conv1(rgb_fuse,t_fuse)  # [B, C, H, W]
 
        # Step 4: 上采样 + 激活（或直接送入上层解码）
        rgb_up = self.actallx(self.upx(rgb_out.permute(0, 2, 3, 1).contiguous()))
        t_up = self.actally(self.upy(t_out.permute(0, 2, 3, 1).contiguous()))

        return rgb_up, t_up

 
class Adapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
         
        self.blockx = blk
        self.blocky = blk

        dim = blk.attn.qkv.in_features
        self.prompt_learnx = CrossDiffFusion(dim, dim_mid=32)
 

    def forward(self, x, y):
        promptx, prompty = self.prompt_learnx(x, y)
        prompedx = x + promptx
        prompedy = y + prompty
        netx = self.blockx(prompedx)
        nety = self.blocky(prompedy)
        return netx, nety
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class LGCA(nn.Module):
    """
    Local-Global Consistency Alignment v2
    - 全局相似度翻转 (global-guided diff)
    - 局部差异能量 (local diff norm)
    - 通道/空间门控 + 跨分支注入
    """
    def __init__(self, in_channels, patch_size=4, reduction=2, smooth_kernel=3, stopgrad_global=True):
        super().__init__()
        assert reduction >= 1
        self.patch_size = patch_size
        c_red = max(4, in_channels // reduction)

        # 维度压缩（防炸显存）
        self.proj = nn.Conv2d(in_channels, c_red, 1, bias=False)
        self.norm = nn.GroupNorm(8, c_red)

        # 全局 token
        self.g_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_red, c_red, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(c_red, c_red, 1, bias=False),
        )
        self.stopgrad_global = stopgrad_global

        # 差异图融合的可学习权重（>0）
        self.w_g = nn.Parameter(torch.tensor(0.5))
        self.w_l = nn.Parameter(torch.tensor(0.5))
        self.softplus = nn.Softplus()

        # 平滑
        pad = smooth_kernel // 2
        self.smooth = nn.Conv2d(1, 1, smooth_kernel, padding=pad, groups=1, bias=False)
        nn.init.constant_(self.smooth.weight, 1.0 / (smooth_kernel * smooth_kernel))

        # 通道门控（SE） + 空间门控
        self.se_x = SE(c_red)
        self.se_y = SE(c_red)
        self.spa_gate = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 注入并还原通道数
        self.inject_x = nn.Conv2d(c_red, in_channels, 1, bias=False)
        self.inject_y = nn.Conv2d(c_red, in_channels, 1, bias=False)
        # 总体残差系数（可学习小步注入，稳定训练）
        self.gamma = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _patch_tokens(x, p):
        # x: [B,C,H,W] -> [B, Ph*Pw, C]
        B, C, H, W = x.shape
        assert H % p == 0 and W % p == 0
        Ph, Pw = H // p, W // p
        t = x.unfold(2, p, p).unfold(3, p, p)                  # [B,C,Ph,Pw,p,p]
        t = t.contiguous().view(B, C, Ph, Pw, p * p).mean(-1)  # patch mean
        t = t.permute(0, 2, 3, 1).contiguous().view(B, Ph * Pw, C)
        return t, Ph, Pw

    def forward(self, x, y):
        """
        x,y: [B,C,H,W]
        return: enhanced_x, enhanced_y, sim_maps_x, sim_maps_y
        """
        # 压缩 + 归一化
        x_low = self.norm(self.proj(x))    # [B,C',H,W]
        y_low = self.norm(self.proj(y))

        # 全局 token（可选择 stop-grad 保守引导）
        gx = self.g_mlp(x_low)  # [B,C',1,1]
        gy = self.g_mlp(y_low)
        if self.stopgrad_global:
            gx = gx.detach()
            gy = gy.detach()

        # Global-guided similarity (patch vs global)
        px, Ph, Pw = self._patch_tokens(x_low, self.patch_size)   # [B,N,C']
        py, _,  _  = self._patch_tokens(y_low, self.patch_size)
        gyv = gy.flatten(2).transpose(1, 2)                       # [B,1,C']
        gxv = gx.flatten(2).transpose(1, 2)

        # cos(patch, global)
        sim_x = F.cosine_similarity(px, gyv.expand(-1, px.size(1), -1), dim=-1)  # [B,N]
        sim_y = F.cosine_similarity(py, gxv.expand(-1, py.size(1), -1), dim=-1)

        # 翻转：差异更亮
        sim_x = 1.0 - sim_x
        sim_y = 1.0 - sim_y

        # reshape为特征图尺度
        sim_x = sim_x.view(-1, 1, Ph, Pw)
        sim_y = sim_y.view(-1, 1, Ph, Pw)
        sim_x = F.interpolate(sim_x, size=x_low.shape[-2:], mode='nearest')
        sim_y = F.interpolate(sim_y, size=y_low.shape[-2:], mode='nearest')

        # Local diff energy（通道L2范数）
        diff = (y_low - x_low)
        lmap = torch.sqrt((diff ** 2).sum(1, keepdim=True) + 1e-6)   # [B,1,H,W]
        lmap = lmap / (lmap.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        # 融合差异图 + 平滑
        w_g = self.softplus(self.w_g)
        w_l = self.softplus(self.w_l)
        mx = self.smooth(torch.clamp(w_g * sim_x + w_l * lmap, 0, 1))
        my = self.smooth(torch.clamp(w_g * sim_y + w_l * lmap, 0, 1))

        # 通道 SE 门控 + 空间门控
        xg = self.se_x(x_low)
        yg = self.se_y(y_low)
        spa = self.spa_gate(torch.cat([mx, my], dim=1))               # [B,1,H,W]

        # 跨分支注入（x 注入来自 y 的差异成分，反之亦然）
        inj_x = self.inject_x(yg * mx * spa)                          # to C
        inj_y = self.inject_y(xg * my * spa)

        enhanced_x = x + self.gamma * inj_x
        enhanced_y = y + self.gamma * inj_y

        return enhanced_x, enhanced_y, mx, my
class SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.lgca1 = LGCA(in_channels=144, patch_size=8, reduction=2)
        self.lgca2 = LGCA(in_channels=288, patch_size=4, reduction=2)
        self.lgca3 = LGCA(in_channels=576, patch_size=2, reduction=2)
        self.lgca4 = LGCA(in_channels=1152, patch_size=1, reduction=2)

        self.rfb1 = RFB_modified(288  , 64)
        self.rfb2 = RFB_modified(576 , 64)
        self.rfb3 = RFB_modified(1152, 64)
        self.rfb4 = RFB_modified(2304 , 64)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x,y):
        # 两个编码器分别处理x和y，共享解码器
        x, y = self.encoder(x,y)  # 第一个编码器处理输入x
        x1, x2, x3, x4 = x[0],x[1],x[2],x[3]
        y1, y2, y3, y4 = y[0],y[1],y[2],y[3]
        x1_enh, y1_enh, _, _ = self.lgca1(x1, y1)  # [1,144,112,112]

        # Stage 2
        x2_enh, y2_enh, _, _ = self.lgca2(x2, y2)  # [1,288,56,56]

        # Stage 3
        x3_enh, y3_enh, _, _ = self.lgca3(x3, y3)  # [1,576,28,28]

        # Stage 4
        x4_enh, y4_enh, _, _ = self.lgca4(x4, y4)  # [1,1152,14,14]

        f1 =  torch.cat([x1_enh,y1_enh],dim=1) # [B, 144*2, 112, 112]
        f2 =  torch.cat([x2_enh,y2_enh],dim=1) # [B, 288*2, 56, 56]
        f3 =  torch.cat([x3_enh,y3_enh],dim=1) # [B, 576*2, 28, 28]
        f4 =  torch.cat([x4_enh,y4_enh],dim=1) # [B,1152*2, 14, 14]

        f1, f2, f3, f4 = self.rfb1(f1), self.rfb2(f2), self.rfb3(f3), self.rfb4(f4)

        # 共享解码器解码融合后的特征
        x = self.up1(f4, f3)
        # out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, f2)
        # out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, f1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')

        return out 

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)