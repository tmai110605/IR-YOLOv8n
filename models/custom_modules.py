import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import Conv, C2f, Bottleneck

_MODULES_REGISTERED = False


class IlluminationGatedAttention(nn.Module):
    """
    Illumination-Gated Attention (IGA) with Structural Gradient Energy:
    Lightweight dual-path attention mechanism specifically designed to mitigate
    specular reflection (glare), harsh shadows, and backlighting in defect detection.

    Physical & Architectural Principle:
    Specular glare on polymer banknotes forms smooth, saturated highlights (low gradient energy),
    whereas true tear defects form sharp mechanical fractures (high gradient energy).
    By coupling luminance statistics (Mean, Max) with local structural gradient energy:
        D(x) = |x - AvgPool_3x3(x)|
    the spatial gate provides signed bipolar modulation:
        w_s = Tanh(BN(Conv_7x7([Mean(x), Max(x), D(x)]))) in (-1, 1)
        - In specular glare hotspots: w_s < 0  ==> actively attenuates glare blooming.
        - Provides signed feature modulation across diverse structural regions.
    """
    def __init__(self, c, r=4):
        super().__init__()
        mid = max(8, c // r)
        # Spectral channel gate (GAP + GMP descriptor)
        self.fc1 = nn.Conv2d(c, mid, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(mid, c, kernel_size=1, bias=False)

        # Structural & Glare Spatial Gate (3 input channels: Mean, Max, Local Gradient Energy)
        self.spatial = nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 1. Channel descriptor (GAP + GMP)
        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)
        ch_attn = self.sigmoid(self.fc2(self.act(self.fc1(gap + gmp))))

        # 2. Structural gradient descriptor (separates smooth specular glare from sharp fracture edges)
        avg_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        grad_energy = torch.mean(torch.abs(x - avg_pool), dim=1, keepdim=True)

        # 3. Spatial luminance statistics
        mean_stat = torch.mean(x, dim=1, keepdim=True)
        max_stat, _ = torch.max(x, dim=1, keepdim=True)

        sp_stat = torch.cat([mean_stat, max_stat, grad_energy], dim=1)
        sp_attn = self.tanh(self.bn(self.spatial(sp_stat)))

        # Bipolar feature modulation
        return x * (ch_attn * sp_attn)


class C2f_IGA(C2f):
    """
    Illumination-Gated C2f Block (C2f-IGA):
    Subclasses standard YOLOv8 C2f block to maintain 100% weight-tensor compatibility
    with pretrained checkpoints (cv1, cv2, Bottlenecks) while adding Illumination-Gated
    Attention to calibrate feature representations under extreme lighting conditions.

    Zero-Disruption Identity Residual:
        y = out + gamma * IGA(out)
    With gamma initialized to 0.05, the module starts with 95% baseline identity mapping
    at epoch 0, eliminating feature explosion while enabling immediate gradient flow
    into the IGA kernels right from step 0.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.iga = IlluminationGatedAttention(c2)
        # Small positive initialization for immediate active gradient propagation
        self.gamma = nn.Parameter(torch.full((1,), 0.05))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out + self.gamma * self.iga(out)


# =============================================================================
# ABLATION VARIANTS: CBAM AND IGA WITHOUT D(x)
# =============================================================================

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., ECCV 2018).
    Standard spatial and channel attention using max/average pooling across raw pixel intensities.
    Under specular glare, max/avg pooling latches onto the saturated reflection peak.
    """
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(8, c // r)
        # Channel attention: shared MLP on GAP and GMP
        self.fc1 = nn.Conv2d(c, mid, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, c, kernel_size=1, bias=False)
        self.sigmoid_ch = nn.Sigmoid()

        # Spatial attention: Conv 7x7 on [AvgPool, MaxPool]
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid_sp = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel gate
        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)
        ch_attn = self.sigmoid_ch(self.fc2(self.act(self.fc1(gap))) + self.fc2(self.act(self.fc1(gmp))))
        x_ch = x * ch_attn

        # 2. Spatial gate (raw intensity statistics)
        avg_stat = torch.mean(x_ch, dim=1, keepdim=True)
        max_stat, _ = torch.max(x_ch, dim=1, keepdim=True)
        sp_stat = torch.cat([avg_stat, max_stat], dim=1)
        sp_attn = self.sigmoid_sp(self.spatial(sp_stat))
        return x_ch * sp_attn


class C2f_CBAM(C2f):
    """
    Ablation variant: YOLOv8 C2f Block augmented with conventional CBAM attention.
    Placed at defect heads P3/8 and P4/16 to benchmark standard spatial attention against IGA.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.cbam(out)


class IlluminationGatedAttention_NoDx(nn.Module):
    """
    Ablation variant: Illumination-Gated Attention WITHOUT Local Structural Gradient Energy D(x).
    Uses only Mean and Max in the spatial gate with Tanh bipolar modulation.
    Lacks the physical gradient operator D(x) = |x - AvgPool_3x3(x)|, testing whether
    intensity statistics alone are sufficient to suppress specular glare hotspots.
    """
    def __init__(self, c, r=4):
        super().__init__()
        mid = max(8, c // r)
        self.fc1 = nn.Conv2d(c, mid, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(mid, c, kernel_size=1, bias=False)

        # 2 spatial channels: Mean and Max (omits D(x))
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)
        ch_attn = self.sigmoid(self.fc2(self.act(self.fc1(gap + gmp))))

        mean_stat = torch.mean(x, dim=1, keepdim=True)
        max_stat, _ = torch.max(x, dim=1, keepdim=True)
        sp_stat = torch.cat([mean_stat, max_stat], dim=1)
        sp_attn = self.tanh(self.bn(self.spatial(sp_stat)))

        return x * (ch_attn * sp_attn)


class C2f_IGA_NoDx(C2f):
    """
    Ablation variant: C2f-IGA WITHOUT Local Structural Gradient Energy D(x).
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.iga = IlluminationGatedAttention_NoDx(c2)
        self.gamma = nn.Parameter(torch.full((1,), 0.05))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out + self.gamma * self.iga(out)


# Backward compatibility classes for older checkpoints/configs
class LightResilientStem(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)

    def forward(self, x):
        return self.conv(x)


class IlluminationRobustC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.c2f = C2f(c1, c2, n=n, shortcut=shortcut, g=g, e=e)

    def forward(self, x):
        return self.c2f(x)


def register_custom_modules():
    """
    Dynamically registers custom modules into Ultralytics registry and monkey-patches
    tasks.parse_model so custom blocks can be loaded directly from YAML configuration files.
    Idempotent: will only execute once.
    """
    global _MODULES_REGISTERED
    if _MODULES_REGISTERED:
        return

    custom_classes = [
        C2f_IGA, IlluminationGatedAttention,
        C2f_CBAM, CBAM,
        C2f_IGA_NoDx, IlluminationGatedAttention_NoDx,
        LightResilientStem, IlluminationRobustC2f
    ]
    for cls in custom_classes:
        setattr(modules, cls.__name__, cls)
        setattr(tasks, cls.__name__, cls)

    try:
        src = inspect.getsource(tasks.parse_model)
        modified = False
        for c2f_variant in ["C2f_IGA", "C2f_CBAM", "C2f_IGA_NoDx"]:
            if f"{c2f_variant}," not in src:
                src = src.replace("            C2f,\n", f"            C2f,\n            {c2f_variant},\n")
                modified = True
                
        if "IlluminationGatedAttention" not in src:
            target = "elif m is AIFI:"
            patch_code = (
                "elif m in (IlluminationGatedAttention, IlluminationGatedAttention_NoDx, CBAM):\n"
                "            c2 = ch[f]\n"
                "            args = [c2, *args]\n"
                "        elif m is AIFI:"
            )
            src = src.replace(target, patch_code)
            modified = True

        if modified:
            local_vars = tasks.__dict__.copy()
            for cls in custom_classes:
                local_vars[cls.__name__] = cls
            exec(src, local_vars)
            tasks.parse_model = local_vars['parse_model']
            print("[INFO] Successfully registered custom modules [C2f_IGA, C2f_CBAM, C2f_IGA_NoDx] into Ultralytics parse_model.")
    except Exception as e:
        print(f"[WARN] Failed to patch parse_model: {e}")

    _MODULES_REGISTERED = True
