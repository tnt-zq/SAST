import argparse, os
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
from skimage import color,exposure

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle
import pywt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.color")

from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# === CLIPSeg 掩码生成 ===
from semantic_mask_clipseg import get_preference_mask as get_clipseg_mask

# === 新的人类偏好驱动语义掩码 ===
from semantic_preference_mask_sam import get_semantic_preference_mask_sam as get_semantic_preference_mask
# from semantic_preference_mask import get_semantic_preference_mask
# from semantic_preference_mask_segformer import get_semantic_preference_mask
from text_protect_mask import get_text_protect_mask


#全局变量，用于存储特征图
feat_maps = []
user_preference = {
    # 人物与动物
    "person": 0.5, "face": 0.5,"text": 0.1, "dog": 0.6, "cat": 0.6, "horse": 0.6, "bird": 0.6, "sheep": 0.6, "cow": 0.6,
    # 建筑与人工设施
    "building": 0.8, "bridge": 0.5, "tower": 0.8, "house": 0.8, "road": 0.8, "sidewalk": 0.7,
    "bus": 0.8, "car": 0.7, "truck": 0.7, "train": 0.8, "boat": 0.7,
    # 自然场景
    "sky": 0.9, "mountain": 1.0, "tree": 0.9, "grass": 0.8, "river": 0.9, "sea": 0.9,
    # 室内家具
    "chair": 0.6, "couch": 0.6, "sofa": 0.6, "table": 0.6, "bed": 0.6, "tv": 0.7,
    # 低优先级背景
    "background": 0.9, "wall": 0.5, "floor": 0.5, "window": 0.6
}

#从样本中解码图像并保存为文件
def save_img_from_sample(model, samples_ddim, fname):
    #对样本进行第一阶段解码
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    #将解码后的样本值范围从[-1,1]调整到[0,1]
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    #将样本从GPU移到CPU并调整维度顺序
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    #将样本转换为PyTorch张量并调整维度顺序
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    #将第一个样本转换为0-255范围的图像数据
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    #将图像数据转换为PIL图像
    img = Image.fromarray(x_sample.astype(np.uint8))
    #保存图像
    img.save(fname)

#根据步数合并内容和风格特征
# def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
#     #初始化特征图列表，每个元素包含配置信息
#     feat_maps = [{'config': {
#                 'gamma':opt.gamma,
#                 'T':opt.T,
#                 'timestep':_,
#                 }} for _ in range(50)]
#
#     for i in range(len(feat_maps)):
#         if i < (50 - start_step):
#             continue
#         #获取当前步数的内容特征和风格特征
#         cnt_feat = cnt_feats[i]
#         sty_feat = sty_feats[i]
#         #获取风格特征的键
#         ori_keys = sty_feat.keys()
#
#         for ori_key in ori_keys:
#
#             if ori_key[-1] == 'q':
#                 feat_maps[i][ori_key] = cnt_feat[ori_key]
#             if ori_key[-1] == 'k' or ori_key[-1] == 'v':
#                 feat_maps[i][ori_key] = sty_feat[ori_key]
#     return feat_maps


def feat_merge(opt, cnt_feats, sty_feats, start_step=0, mask=None,
               mask_gamma=0.7, alpha=1.5, depth_boost=True):
    """
    人类偏好驱动的语义掩码融合 + 风格增强。
    参数:
        mask_gamma: 掩码非线性压缩系数 (<1 提高风格化比例)
        alpha: 风格分量放大系数 (>1 增强整体风格信号)
        depth_boost: 是否对深层特征使用更强风格融合
    """
    feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T, 'timestep': _}}
                 for _ in range(50)]

    if mask is not None:
        # 非线性压缩，提升风格区域占比
        mask = mask ** mask_gamma
        mask = torch.clamp(mask, 0, 1)
        print(f"[feat_merge] Mask non-linear enhanced (gamma={mask_gamma}), mean={mask.mean():.3f}")

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue

        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key.endswith("q"):
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            elif ori_key.endswith(("k", "v")):
                if mask is not None:
                    feat = sty_feat[ori_key]
                    layer_strength = alpha

                    # 对 deeper 层增强风格权重
                    if depth_boost:
                        # 分层风格增强：重点强化浅层纹理
                        if 4 <= i <= 6:
                            layer_strength *= 1.8  # 强笔触
                        elif 6 < i <= 8:
                            layer_strength *= 1.3  # 中等纹理
                        else:
                            layer_strength *= 1.0  # 其他层维持原样

                    if feat.ndim == 4:
                        _, _, H, W = feat.shape
                        mask_resized = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
                        feat_maps[i][ori_key] = (
                            cnt_feat[ori_key] * (1 - mask_resized) +
                            sty_feat[ori_key] * mask_resized * layer_strength
                        )
                    elif feat.ndim == 3:
                        B, N, C = feat.shape
                        H = W = int(N ** 0.5)
                        if H * W == N:
                            mask_resized = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
                            mask_flat = mask_resized.flatten(2).permute(0, 2, 1)
                            feat_maps[i][ori_key] = (
                                cnt_feat[ori_key] * (1 - mask_flat) +
                                sty_feat[ori_key] * mask_flat * layer_strength
                            )
                        else:
                            feat_maps[i][ori_key] = sty_feat[ori_key] * layer_strength
                    else:
                        feat_maps[i][ori_key] = sty_feat[ori_key] * layer_strength
                else:
                    feat_maps[i][ori_key] = sty_feat[ori_key]

        if mask is not None and any(k.endswith('v') for k in ori_keys):
            print(f"[feat_merge] Layer {i} fusion applied, mean(mask)={mask.mean():.3f}")

    return feat_maps
# def feat_merge(
#     opt, cnt_feats, sty_feats,
#     start_step=0, mask=None,
#     mask_gamma=0.9,   # 🔧 提高 gamma，避免掩码整体抬升
#     alpha=1.2,        # 🔧 降低默认 alpha
#     depth_boost=True
# ):
#     """
#     稳定版 CFG-inspired Attention Blending
#     """
#
#     feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T, 'timestep': _}}
#                  for _ in range(50)]
#
#     if mask is not None:
#         mask = torch.clamp(mask ** mask_gamma, 0, 1)
#         print(f"[feat_merge] Mask stabilized (gamma={mask_gamma}), mean={mask.mean():.3f}")
#
#     for i in range(len(feat_maps)):
#         if i < (50 - start_step):
#             continue
#
#         cnt_feat = cnt_feats[i]
#         sty_feat = sty_feats[i]
#
#         for key in cnt_feat.keys():
#
#             # ---- Q always from content ----
#             if key.endswith("q"):
#                 feat_maps[i][key] = cnt_feat[key]
#                 continue
#
#             # ---- CFG-style K/V blending ----
#             if key.endswith(("k", "v")):
#                 Kc = cnt_feat[key]
#                 Ks = sty_feat[key]
#
#                 # -------- layer-wise bias (additive, not multiplicative) --------
#                 layer_alpha = alpha
#                 if depth_boost:
#                     if 4 <= i <= 6:
#                         layer_alpha += 0.4
#                     elif 6 < i <= 8:
#                         layer_alpha += 0.2
#
#                 # -------- semantic-aware alpha --------
#                 if mask is not None and Kc.ndim == 4:
#                     _, _, H, W = Kc.shape
#                     m = F.interpolate(mask, size=(H, W),
#                                       mode="bilinear", align_corners=False)
#                     alpha_r = layer_alpha * m
#                 else:
#                     alpha_r = layer_alpha
#
#                 # -------- soft saturation (关键稳定器) --------
#                 alpha_r = alpha_r / (1.0 + alpha_r)
#
#                 # -------- CFG blending --------
#                 feat_maps[i][key] = Kc + alpha_r * (Ks - Kc)
#
#     return feat_maps






#加载并处理图像，将其转换为张量格式
#图像被裁剪成正方形，调整为512*512像素大小，并标准化到【-1，1】范围
def load_img(path, style_path=None,save_lab_preview=False, save_dir="lab_preview"):
    #打开图像并转换为RGB模式
    image = Image.open(path).convert("RGB")
    #获取图像的宽度和高度
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    if style_path is not None:
        # print("[Init] Applying LAB color transfer between content and style ...")
        # style_img = Image.open(style_path).convert("RGB")
        # lab_aligned = color_transfer_lab(image, style_img)
        #
        # # ====== ⭐ 保存 LAB 对齐结果 ⭐ ======
        # if save_lab_preview:
        #     os.makedirs(save_dir, exist_ok=True)
        #     fname = os.path.join(
        #         save_dir,
        #         f"LAB_{os.path.basename(path).split('.')[0]}_aligned.png"
        #     )
        #     Image.fromarray(lab_aligned).save(fname)
        #     print(f"[LAB] Saved LAB-aligned preview → {fname}")
        #
        # image = Image.fromarray(lab_aligned)
        pass
    #设定目标图像的宽度和高度
    h = w = 512
    #对图像进行中心裁剪，使其为正方形
    image = transforms.CenterCrop(min(x,y))(image)
    #调整图像为512*512
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    #将图像转换为numpy数组并归一化到【0，1】
    image = np.array(image).astype(np.float32) / 255.0
    #调整数组维度顺序
    image = image[None].transpose(0, 3, 1, 2)
    #将numpy数组转换为Pytorch张量
    image = torch.from_numpy(image)
    # 将张量的值范围从 [0, 1] 调整到 [-1, 1]
    return 2.*image - 1.
# 1. prepare_mask(mask) —— 适配尺寸 + 平滑 + 扩展通道
# ---------------------------------------------------------
def prepare_mask(mask, target_hw, blur=21):
    """
    mask: [1,1,H,W] or PIL/np
    返回: [1,1,h,w] （与特征图同分辨率）
    """
    if mask is None:
        return None

    if isinstance(mask, torch.Tensor):
        m = mask
    else:
        # 转 tensor + 归一化
        m = torch.from_numpy(mask).float()
        if m.ndim == 3:
            m = m.mean(0, keepdim=True)
        m = m.unsqueeze(0)

    # resize
    m = F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)

    # 高斯模糊（使用卷积实现）
    if blur > 0:
        k = blur
        sigma = blur / 6
        x = torch.arange(k) - k // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel.to(m.device)
        kernel = kernel.expand(1,1,k,k)
        m = F.conv2d(m, kernel, padding=k//2)

    # clamp
    return torch.clamp(m, 0, 1)


#实现了自适应归一化（AdaIN），用于将内容特征调整为风格特征的统计特性
def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output
def adain(cnt_feat, sty_feat, alpha=0.6, phi=1e-5):
    """
    噪声初始化模块
    先用AdaIN将内容特征对齐到风格统计量，再与原始内容特征加权融合

    参数:
        cnt_feat: 内容特征/噪声 [B, C, H, W]
        sty_feat: 风格特征/噪声 [B, C, H, W]
        alpha: 融合权重，控制AdaIN结果与原始内容的比例
        phi: 防止除零的小常数

    返回:
        融合后的特征/噪声
    """
    # 1. 计算内容特征的统计量（均值和标准差）
    cnt_mean = cnt_feat.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
    cnt_std = cnt_feat.std(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

    # 2. 计算风格特征的统计量
    sty_mean = sty_feat.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
    sty_std = sty_feat.std(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

    # 3. 应用AdaIN：将内容特征统计量对齐到风格统计量
    # 公式: Z_CS = √(σ(Z_S) + φ) * (Z_C - μ(Z_C)) / √(σ(Z_C) + φ) + μ(Z_S)
    cnt_feat_normalized = (cnt_feat - cnt_mean) / (cnt_std + phi)
    cnt_feat_aligned = cnt_feat_normalized * (sty_std + phi).sqrt() + sty_mean

    # 4. 加权融合：将AdaIN对齐结果与原始内容特征融合
    # 公式: Z_T = α × Z_CS + (1-α) × Z_C
    fused_feat = alpha * cnt_feat_aligned + (1 - alpha) * cnt_feat

    return fused_feat
# def adain(cnt_feat, sty_feat, alpha=0.6, phi=1e-5):
#     """
#     自适应归一化 (AdaIN) + 可选融合内容特征

#     参数:
#         cnt_feat: 内容特征 [B, C, H, W]
#         sty_feat: 风格特征 [B, C, H, W]
#         alpha: 融合权重，0-1，alpha=1 完全采用风格
#         phi: 防止除零的小常数

#     返回:
#         融合后的特征 [B, C, H, W]
#     """

#     # 1. 内容特征均值和标准差
#     cnt_mean = cnt_feat.mean(dim=[2, 3], keepdim=True)      # [B, C, 1, 1]
#     cnt_std  = cnt_feat.std(dim=[2, 3], keepdim=True) + phi # [B, C, 1, 1]

#     # 2. 风格特征均值和标准差
#     sty_mean = sty_feat.mean(dim=[2, 3], keepdim=True)      # [B, C, 1, 1]
#     sty_std  = sty_feat.std(dim=[2, 3], keepdim=True) + phi # [B, C, 1, 1]

#     # 3. 标准化内容特征
#     cnt_normalized = (cnt_feat - cnt_mean) / cnt_std

#     # 4. 对齐到风格统计量
#     cnt_aligned = cnt_normalized * sty_std + sty_mean

#     # 5. 融合原始内容特征
#     fused_feat = alpha * cnt_aligned + (1 - alpha) * cnt_feat

#     return fused_feat
def get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    net = nn.Conv2d if pool else nn.ConvTranspose2d

    LL = net(in_channels, in_channels, 2, 2, 0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, 2, 2, 0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, 2, 2, 0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, 2, 2, 0, bias=False, groups=in_channels)

    for layer, f in zip([LL, LH, HL, HH], [filter_LL, filter_LH, filter_HL, filter_HH]):
        layer.weight.requires_grad = False
        layer.weight.data = f.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, pool=False)

    def forward(self, LL, LH, HL, HH):
        return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
class AdaINWaveletInit(nn.Module):
    """
    完全复刻 DGPST 的 AdaIN + Wavelet 初始化
    """
    def __init__(self, latent_channels=4):
        super().__init__()
        self.wav = WavePool(latent_channels)
        self.wavunpool = WaveUnpool(latent_channels)

    def forward(self, z_content, z_style):
        """
        z_content: 内容 latent (B,4,H,W)
        z_style:   风格 latent (B,4,H,W)
        """

        #AdaIN
        z_ada = adain(z_content, z_style)

        #  Wavelet 分解
        LL_s, _, _, _ = self.wav(z_style)      # 低频 ← 风格
        _, LH, HL, HH = self.wav(z_ada)        # 高频 ← AdaIN latent

        # Wavelet 重构
        z_init = self.wavunpool(LL_s, LH, HL, HH)

        return z_init


adain_wavelet = AdaINWaveletInit(latent_channels=4).cuda()

def color_transfer_lab(content_img, style_img, lum_clip=(0.05, 0.95), chroma_scale=0.8):
    if isinstance(content_img, Image.Image):
        content = np.asarray(content_img).astype(np.float32) / 255.0
    else:
        content = content_img.astype(np.float32) / 255.0
    if isinstance(style_img, Image.Image):
        style = np.asarray(style_img).astype(np.float32) / 255.0
    else:
        style = style_img.astype(np.float32) / 255.0

    content_lab = color.rgb2lab(content)
    style_lab = color.rgb2lab(style)

    c_mean, c_std = content_lab.mean(axis=(0,1)), content_lab.std(axis=(0,1))
    s_mean, s_std = style_lab.mean(axis=(0,1)), style_lab.std(axis=(0,1))

    # 限制 L 通道亮度范围，避免过亮或过暗
    L = np.clip((content_lab[...,0] - c_mean[0]) / (c_std[0]+1e-8) * s_std[0] + s_mean[0],
                lum_clip[0]*100, lum_clip[1]*100)
    # 缩放色度以防止过饱和
    a = ((content_lab[...,1]-c_mean[1])/c_std[1]*s_std[1] + s_mean[1]) * chroma_scale
    b = ((content_lab[...,2]-c_mean[2])/c_std[2]*s_std[2] + s_mean[2]) * chroma_scale

    lab_matched = np.stack([L,a,b],axis=-1)
    rgb_matched = np.clip(color.lab2rgb(lab_matched),0,1)
    return (rgb_matched*255).astype(np.uint8)

# def color_transfer_lab_hist(gen_img, style_img,
#                             beta=1.15, gamma=2.5, lam=0.5):
#     """
#     论文版：LAB直方图匹配 + 亮度微调 + 伽马校正 + 融合（后处理）
#     """
#
#     if isinstance(gen_img, Image.Image):
#         gen = np.asarray(gen_img).astype(np.float32) / 255.0
#     else:
#         gen = gen_img.astype(np.float32) / 255.0
#
#     if isinstance(style_img, Image.Image):
#         sty = np.asarray(style_img).astype(np.float32) / 255.0
#     else:
#         sty = style_img.astype(np.float32) / 255.0
#
#     # RGB->LAB
#     gen_lab = color.rgb2lab(gen)
#     sty_lab = color.rgb2lab(sty)
#
#     out_lab = np.zeros_like(gen_lab)
#
#     # ---- 1) 三通道直方图匹配 ----
#     for c in range(3):
#         out_lab[..., c] = exposure.match_histograms(
#             gen_lab[..., c], sty_lab[..., c]
#         )
#
#     # ---- 2) 亮度微调 + 伽马校正 ----
#     L = out_lab[..., 0] / 100.0
#     L = np.clip(beta * L, 0, 1)
#     L = np.power(L, 1.0 / gamma)
#     out_lab[..., 0] = L * 100.0
#
#     # ---- 3) LAB->RGB ----
#     I_ac = np.clip(color.lab2rgb(out_lab), 0, 1)
#
#     # ---- 4) 融合 ----
#     I_cs = (1 - lam) * gen + lam * I_ac
#
#     return (np.clip(I_cs, 0, 1) * 255).astype(np.uint8)

#从配置和检查点加载模型
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty', default = './data/sty')
    # 添加DDIM反转步数参数
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    # 添加保存特征部署参数
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    # 添加开始步数参数
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    # 添加DDIM eta参数
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    # 添加下采样因子参数
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    # 添加注意力温度缩放超参数
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    # 添加查询保留超参数
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--mask_gamma", type=float, default=0.7, help="掩码非线性增强系数")
    parser.add_argument("--alpha", type=float, default=1.5, help="注意力融合风格放大系数")
    parser.add_argument("--depth_boost", action="store_true", help="是否增强深层风格融合")
    parser.add_argument("--invert_mask", action="store_true", help="若掩码语义与期望相反则反转掩码")
    # 添加注入注意力特征层参数
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    # 添加模型配置文件路径参数
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    # 添加预计算特征保存路径参数
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='save path for precomputed feature')
    # 添加模型检查点文件路径参数
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt', help='model checkpoint')
    # 添加精度参数
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    # 添加输出路径参数
    parser.add_argument('--output_path', type=str, default='output')
    # 添加是否不使用初始AdaIN的参数
    parser.add_argument("--without_init_adain", action='store_true')
    # 添加是否不使用注意力注入的参数
    parser.add_argument("--without_attn_injection", action='store_true')
    # 解析命令行参数
    opt = parser.parse_args()
    # 获取预计算特征的根路径
    feat_path_root = opt.precomputed
    # 设置随机种子
    seed_everything(22)
    output_path = opt.output_path
    # 创建输出路径目录
    os.makedirs(output_path, exist_ok=True)
    if len(feat_path_root) > 0:
        # 创建预计算特征保存路径目录
        os.makedirs(feat_path_root, exist_ok=True)
    # 加载模型配置文件
    model_config = OmegaConf.load(f"{opt.model_config}")
    # 从配置和检查点加载模型
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    # 解析注意力层参数
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))

    #初始化DDIM采样器
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    # 创建时间步映射字典
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn 保存自注意力层的Q/K/V
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    # 特征保存回调函数
    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map
        # feat_maps[cur_idx][f"{filename}"] = feature_map.detach().cpu().half()

    # ****主处理流程
    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])  #无条件文本嵌入
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f] #潜在空间形状

    # 遍历风格和内容对
    # sty_img_list = sorted(os.listdir(opt.sty))
    # cnt_img_list = sorted(os.listdir(opt.cnt))
    # --- 支持目录 or 单一文件 ---
    def get_image_list(path):
        if os.path.isdir(path):
            return [os.path.join(path, f)
                    for f in sorted(os.listdir(path))
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        elif os.path.isfile(path):
            return [path]
        else:
            raise ValueError(f"Invalid path: {path}")

    sty_img_list = get_image_list(opt.sty)
    cnt_img_list = get_image_list(opt.cnt)

    begin = time.time()
    for sty_name in sty_img_list:
        # 加载风格图像
        sty_name_ = sty_name  # 因为现在 sty_name 已经是完整路径
        style_pil = Image.open(sty_name_).convert("RGB")  # 新增
        init_sty = load_img(sty_name_).to(device)
        seed = -1

        # 尝试加载预计算风格特征
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            # DDIM反转获得风格特征
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                callback_ddim_timesteps=save_feature_timesteps,
                                                img_callback=ddim_sampler_callback)
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']

        # 处理内容图像
        for cnt_name in cnt_img_list:
            cnt_name_ = cnt_name
            # 生成语义偏好掩码
            try:
                content_pil = Image.open(cnt_name_).convert("RGB")
                
                torch.cuda.empty_cache()
                preference_mask = get_semantic_preference_mask(
                    content_pil=content_pil,
                    style_pil=style_pil,  # 核心新增参数
                    device=device,
                )


                # 如果用户发现掩码语义正好相反，可反转
                preference_mask = 1.0 - preference_mask
                print(
                    f"[SemanticPref] Combined preference mask generated for {cnt_name}, mean={preference_mask.mean():.3f}")
            except Exception as e:
                print(f"[SemanticPref] Mask generation failed: {e}")
                preference_mask = None


            init_cnt = load_img(cnt_name_, style_path=sty_name_,save_lab_preview=True).to(device)
            cnt_feat_name = os.path.join(feat_path_root, os.path.basename(cnt_name).split('.')[0] + '_cnt.pkl')
            cnt_feat = None

            # ddim inversion encoding
            if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                print("Precomputed content feature loading: ", cnt_feat_name)
                with open(cnt_feat_name, 'rb') as h:
                    cnt_feat = pickle.load(h)
                    cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
            else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                    end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                    callback_ddim_timesteps=save_feature_timesteps,
                                                    img_callback=ddim_sampler_callback)
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc']
            # 风格迁移生成
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        # inversion
                        output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"

                        print(f"Inversion end: {time.time() - begin}")
                        # 初始化潜在空间处理
                        if opt.without_init_adain:
                            adain_z_enc = cnt_z_enc
                        else:
                             adain_z_enc = adain(cnt_z_enc, sty_z_enc) #AdaIN风格迁移
                             # adain_z_enc = adain_wavelet(cnt_z_enc, sty_z_enc)
                        # adain_z_enc = adain_masked(cnt_z_enc, sty_z_enc, mask=preference_mask)

                        # 注意力特征融合
                        # feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                        print("[DEBUG] Calling feat_merge, mask is None?", preference_mask is None)
                        feat_maps = feat_merge(
                            opt, cnt_feat, sty_feat, start_step=start_step, mask=preference_mask,
                            mask_gamma=opt.mask_gamma, alpha=opt.alpha, depth_boost=opt.depth_boost
                        )
                        if opt.without_attn_injection:
                            feat_maps = None

                        # inference DDIM采样生成图像
                        samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                        batch_size=1,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=adain_z_enc,
                                                        injected_features=feat_maps,
                                                        start_step=start_step,
                                                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        final_img = color_transfer_lab(img, style_pil)
                        Image.fromarray(final_img).save(os.path.join(output_path, output_name))
                        torch.cuda.empty_cache()

                        # img.save(os.path.join(output_path, output_name))
                        # if len(feat_path_root) > 0:
                        #     print("Save features")
                        #     if not os.path.isfile(cnt_feat_name):
                        #         with open(cnt_feat_name, 'wb') as h:
                        #             pickle.dump(cnt_feat, h)
                        #     if not os.path.isfile(sty_feat_name):
                        #         with open(sty_feat_name, 'wb') as h:
                        #             pickle.dump(sty_feat, h)

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    main()
