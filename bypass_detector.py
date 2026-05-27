#!/usr/bin/env python3
"""
简易 AI 图片检测对抗工具
去掉 LPIPS 对抗攻击（太重），保留：FFT频谱匹配 + 相机管线 + 纹理归一化 + 噪声
用法: python bypass_detector.py input.jpg output.jpg
"""

import sys
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.ndimage import gaussian_filter, convolve

# ============================================================
# 参数调节区 —— 觉得效果不好就改这里
# ============================================================
PARAMS = {
    # ---- FFT 频谱匹配（核心）----
    "fft_enabled": True,
    "fft_mode": "model",         # "model"=1/f幂律, "ref"=参考图, "auto"=自动
    "fft_cutoff": 0.15,          # 低频保护半径缩小，让更多频率参与修改
    "fft_strength": 0.65,        # 取 v8(0.85过高) 和 v9(0.55过糊) 中间
    "fft_alpha": 1.0,            # alpha 太高=压高频=模糊，回到1.0
    "fft_randomness": 0.04,
    "fft_radial_smooth": 5,      # 回到5，保持细节

    # ---- DFL 频谱对抗（拉 residual std）----
    "dfl_enabled": True,
    "dfl_iterations": 80,
    "dfl_strength": 0.25,        # 微调

    # ---- 相机管线 ----
    "camera_enabled": True,
    "bayer": True,
    "chroma_strength": 1.2,
    "vignette_strength": 0.22,
    "iso_scale": 1.0,
    "read_noise": 2.5,
    "hot_pixel_prob": 5e-7,
    "banding_strength": 0.01,
    "motion_blur_kernel": 1,
    "jpeg_cycles": 2,
    "jpeg_qmin": 80,
    "jpeg_qmax": 92,

    # ---- 噪声和扰动 ----
    "noise_enabled": True,
    "noise_std_frac": 0.012,     # 减半
    "perturb_enabled": True,
    "perturb_magnitude": 0.008,  # 减半

    # ---- 纹理归一化 (拉 residual std) ----
    "glcm_enabled": True,
    "glcm_strength": 0.45,       # GLCM 纹理对比度增强强度
    "lbp_enabled": True,
    "lbp_strength": 0.35,        # LBP 纹理多样性增强强度

    # ---- 白平衡 ----
    "awb_enabled": False,        # 关掉，灰世界假设会杀死暖色调

    # ---- EXIF ----
    "add_exif": True,
}

# ============================================================
# 工具函数
# ============================================================

def info(msg):
    print(f"  [{msg}]")

def get_rng(seed=None):
    return np.random.default_rng(seed)


# ============================================================
# DFL 频谱对抗攻击（轻量版）
# 用梯度下降最大化 FFT 频谱差异，LPIPS + L2 约束保证视觉不变
# ============================================================

def dfl_attack(img_rgb, iterations=80, lr=1e-3, strength=0.5, seed=None):
    """
    轻量版 DFL（Differential Frequency Loss）攻击。
    用梯度下降找到"频谱变化最大、同时人眼几乎看不出"的像素级扰动。

    iterations: 迭代次数，原项目 500，轻量版默认 80
    lr: 学习率
    strength: 最终混合强度 (0~1)，0=原图，1=完全应用扰动
    """
    import torch
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = get_rng(seed)  # 固定 PyTorch 之外的随机种子

    h, w = img_rgb.shape[:2]

    # numpy → tensor，归一化到 [-1, 1]
    arr = img_rgb.astype(np.float32)
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor / 127.5 - 1.0
    img_tensor = img_tensor.to(device)

    # 可学习的微小扰动
    delta = (torch.randn_like(img_tensor) * 1e-5).requires_grad_(True).to(device)

    optimizer = torch.optim.Adam([delta], lr=lr)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # 预计算原图 FFT
    img_fft = torch.fft.fft2(img_tensor)

    # 约束阈值（沿用原项目参数）
    t_lpips = 0.04      # LPIPS 感知阈值
    t_l2 = 3e-5         # L2 扰动阈值
    c_lpips = 0.01      # LPIPS 惩罚系数
    c_l2 = 0.6          # L2 惩罚系数
    grad_clip = 0.05    # 梯度裁剪

    for i in range(iterations):
        optimizer.zero_grad()

        # 扰动后的图
        x_nw = img_tensor + delta
        x_nw = torch.clamp(x_nw, -1, 1)

        # (1) DFL: 负号 → 最小化 = 最大化频谱差异
        x_nw_fft = torch.fft.fft2(x_nw)
        loss_dfl = -torch.abs(x_nw_fft - img_fft).sum()

        # (2) LPIPS: 感知相似度
        loss_lpips = lpips_model(x_nw, img_tensor).mean()

        # (3) L2: 扰动幅度
        loss_l2 = torch.linalg.norm(delta)

        # relu 软约束：只在超过阈值时才惩罚
        lpips_penalty = c_lpips * torch.relu(loss_lpips - t_lpips)
        l2_penalty = c_l2 * torch.relu(loss_l2 - t_l2)

        total_loss = loss_dfl + lpips_penalty + l2_penalty
        total_loss.backward()

        # 梯度裁剪，防止单步变化过大
        if delta.grad is not None:
            delta.grad.data.clamp_(-grad_clip, grad_clip)

        optimizer.step()

    # 混合还原
    with torch.no_grad():
        final = img_tensor + delta * strength
        final = torch.clamp(final, -1, 1)
        final = (final + 1) / 2            # [-1,1] → [0,1]
        final = final.squeeze(0).permute(1, 2, 0).cpu().numpy()
        final = np.clip(final * 255, 0, 255).astype(np.uint8)

    return final


# ============================================================
# GLCM 纹理归一化 —— 改纹理统计分布，拉高 residual
# ============================================================

def glcm_normalize(img_rgb, strength=0.4, seed=None):
    """
    在 LAB 亮度通道上做 GLCM 纹理对比度增强。
    AI 图纹理往往过于均匀/homogeneous，通过降低 homogeneity 和提升 contrast
    来模拟真实相机传感器的纹理多样性。
    """
    from skimage.feature import graycomatrix, graycoprops
    import cv2
    rng = get_rng(seed)

    h, w = img_rgb.shape[:2]
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B_ch = cv2.split(lab)
    L_f = L.astype(np.float32)

    # 量化到 64 级以加速 GLCM 计算
    levels = 64
    L_q = np.floor(L_f / 255.0 * (levels - 1)).astype(np.uint8)

    # 计算 GLCM
    glcm = graycomatrix(L_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    # AI 图的 homogeneity 偏高，contrast 偏低
    # 目标：降 homogeneity，升 contrast
    target_contrast = contrast * (1.0 + 0.3 * strength)   # 提升对比度
    target_homogeneity = homogeneity * (1.0 - 0.15 * strength)  # 降低均匀度
    eps = 1e-8

    contrast_scale = np.sqrt(target_contrast / (contrast + eps))
    adjusted_L = L_f * contrast_scale
    # 双边滤波保边，sigma 跟 homogeneity 关联
    sigma = float(np.clip(75.0 / (homogeneity / (target_homogeneity + eps) + eps), 25.0, 150.0))
    adjusted_L = cv2.bilateralFilter(adjusted_L.astype(np.float32), d=9,
                                     sigmaColor=sigma, sigmaSpace=sigma)

    # 微噪声模拟真实纹理
    noise = rng.normal(0, 0.02 * strength, (h, w)).astype(np.float32) * 255.0
    noise = cv2.GaussianBlur(noise, (3, 3), sigmaX=0.5)

    blended_L = (1.0 - strength) * L_f + strength * adjusted_L + noise
    out_L = np.clip(blended_L, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((out_L, A, B_ch)), cv2.COLOR_LAB2RGB)


# ============================================================
# LBP 纹理归一化 —— 增加纹理模式多样性
# ============================================================

def lbp_normalize(img_rgb, radius=3, n_points=24, strength=0.4, seed=None):
    """
    LBP 直方图展宽，增加局部纹理模式多样性。
    AI 图 LBP 分布集中在少数模式（纹理过于规整），通过 CDF 拉伸
    增加纹理模式的丰富度。
    """
    from skimage.feature import local_binary_pattern
    import cv2
    rng = get_rng(seed)

    h, w = img_rgb.shape[:2]
    gray = np.mean(img_rgb.astype(np.float32), axis=2).astype(np.uint8)
    eps = 1e-8

    # 计算 LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_int = np.rint(lbp).astype(np.int32)
    n_bins = n_points + 2  # 'uniform' 模式的 bin 数

    # 统计直方图
    counts = np.bincount(lbp_int.ravel(), minlength=n_bins).astype(np.float64)
    hist = counts / (counts.sum() + eps)

    # CDF 拉伸：让分布更均匀（模拟真图的纹理多样性）
    cdf = np.cumsum(hist)
    # 目标 CDF：从当前分布向均匀分布移动 strength
    uniform_cdf = np.linspace(0, 1, n_bins)
    target_cdf = (1.0 - strength) * cdf + strength * uniform_cdf

    # 构建映射表：源 bin → 目标 bin
    mapping = np.searchsorted(target_cdf, cdf).astype(np.float32)
    mapping = np.clip(mapping, 0, n_bins - 1)

    # 逐像素缩放
    scale_map = mapping[lbp_int]
    denom = lbp_int.astype(np.float32) + eps
    scale = scale_map / denom
    scale = np.clip(scale, 0.7, 1.3)  # 限制单像素变化幅度

    # 只对亮度做
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1]
    B_ch = lab[:, :, 2]

    L_adjusted = L * scale
    L_blend = (1.0 - strength) * L + strength * L_adjusted

    # 微噪声
    noise = rng.normal(0, 0.015 * strength, (h, w)).astype(np.float32) * 255.0
    noise = cv2.GaussianBlur(noise, (3, 3), sigmaX=0.5)
    L_blend += noise

    L_out = np.clip(L_blend, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((L_out, A, B_ch)), cv2.COLOR_LAB2RGB)

# ============================================================
# FFT 频谱匹配 (基于原项目 V3 简化)
# ============================================================

def fft_spectrum_match(img_rgb, mode="model", alpha=1.0, cutoff=0.25,
                       strength=0.5, randomness=0.03, radial_smooth=7, seed=None):
    """
    只修改 LAB 的 L 通道幅度谱，保护 A/B 色彩通道。
    mode='model': 将频谱掰向 1/f^α 自然幂律分布
    """
    import cv2
    rng = get_rng(seed)
    h, w = img_rgb.shape[:2]

    # 转 LAB，只改 L
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # 频率坐标网格
    y = np.linspace(-1, 1, h, endpoint=False)[:, None]
    x = np.linspace(-1, 1, w, endpoint=False)[None, :]
    r = np.sqrt(x * x + y * y)
    r = np.clip(r, 0.0, 1.0 - 1e-6)

    # L 通道的 FFT
    FL = np.fft.fftshift(np.fft.fft2(L))
    mag_src = np.abs(FL)
    phase_src = np.angle(FL)

    # 对幅度图做高斯模糊，保留 2D 结构
    blurred_src = gaussian_filter(mag_src, sigma=radial_smooth)
    eps = 1e-8

    # 构建目标幅度图
    if mode == 'model':
        freq_r = r.copy()
        freq_r[freq_r < eps] = eps
        power_law = (1.0 / freq_r) ** (alpha / 2.0)
        blurred_target = gaussian_filter(power_law, sigma=radial_smooth)
        # 低频能量对齐
        lf_mask = r < cutoff
        blurred_target *= (np.mean(blurred_src[lf_mask]) + eps) / (np.mean(blurred_target[lf_mask]) + eps)

    elif mode == 'ref':
        # 需要参考图，这里暂不支持，回退到 model
        info("FFT ref 模式需要参考图，回退到 model")
        return fft_spectrum_match(img_rgb, mode='model', alpha=alpha, cutoff=cutoff,
                                  strength=strength, randomness=randomness,
                                  radial_smooth=radial_smooth, seed=seed)
    else:
        blurred_target = blurred_src  # 不改

    multiplier_2d = blurred_target / (blurred_src + eps)
    multiplier_2d = np.clip(multiplier_2d, 0.2, 5.0)

    # 权重遮罩：低频保护（不动），高频修改
    edge = max(0.05 + 0.02 * (1.0 - cutoff), 1e-6)
    weight = np.where(
        r < cutoff, 0.0,
        np.where(r < cutoff + edge,
                 0.5 * (1.0 - np.cos(np.pi * (r - cutoff) / edge)),
                 1.0)
    )

    final_mult = 1.0 + (multiplier_2d - 1.0) * (weight * strength)

    if randomness > 0:
        noise = rng.normal(loc=1.0, scale=randomness, size=final_mult.shape)
        final_mult *= (1.0 + (noise - 1.0) * weight)

    # 应用
    mag_new = mag_src * final_mult
    F_new = mag_new * np.exp(1j * phase_src)
    L_new = np.real(np.fft.ifft2(np.fft.ifftshift(F_new)))

    # 混合
    L_blend = (1.0 - strength) * L + strength * L_new
    L_out = np.clip(L_blend, 0, 255).astype(np.uint8)

    lab_out = np.stack([L_out, A, B], axis=2)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


# ============================================================
# 相机管线
# ============================================================

def _bayer_mosaic(img):
    """RGB → Bayer RGGB 单通道"""
    h, w = img.shape[:2]
    mosaic = np.zeros((h, w), dtype=np.uint8)
    mosaic[0::2, 0::2] = img[0::2, 0::2, 0]  # R
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]  # G
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]  # G
    mosaic[1::2, 1::2] = img[1::2, 1::2, 2]  # B
    return mosaic


def _demosaic_bilinear(mosaic):
    """双线性去马赛克 → RGB"""
    h, w = mosaic.shape
    m = mosaic.astype(np.float32)
    R, G, B = np.zeros_like(m), np.zeros_like(m), np.zeros_like(m)

    R[0::2, 0::2] = m[0::2, 0::2]
    G[0::2, 1::2] = m[0::2, 1::2]
    G[1::2, 0::2] = m[1::2, 0::2]
    B[1::2, 1::2] = m[1::2, 1::2]

    k_cross = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 8.0
    R = convolve(R, k_cross, mode='mirror')
    G = convolve(G, k_cross, mode='mirror')
    B = convolve(B, k_cross, mode='mirror')

    return np.clip(np.stack([R, G, B], axis=2), 0, 255).astype(np.uint8)


def _chromatic_aberration(img, strength, rng):
    """R/B 通道横向偏移模拟色差"""
    import cv2
    h, w = img.shape[:2]
    shift_r = rng.normal(0, strength * 0.5)
    shift_b = rng.normal(0, strength * 0.5)

    out = img.copy().astype(np.float32)
    M_r = np.array([[1, 0, shift_r], [0, 1, 0]], dtype=np.float32)
    M_b = np.array([[1, 0, -shift_b], [0, 1, 0]], dtype=np.float32)
    out[:, :, 0] = cv2.warpAffine(out[:, :, 0], M_r, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out[:, :, 2] = cv2.warpAffine(out[:, :, 2], M_b, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8)


def _vignette(img, strength):
    """边缘变暗"""
    h, w = img.shape[:2]
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    r = np.sqrt(x * x + y * y)
    mask = np.clip(1.0 - (r ** 2) * strength, 0.0, 1.0)
    return np.clip(img.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)


def _sensor_noise(img, iso_scale, read_noise, rng):
    """Poisson-Gaussian 传感器噪声模型"""
    img_f = img.astype(np.float32)
    scaled = img_f * iso_scale
    photon_scale = 4.0
    lam = np.clip(scaled * photon_scale, 0, 1e6)
    noisy = rng.poisson(lam).astype(np.float32) / photon_scale
    noisy += rng.normal(0, read_noise, size=noisy.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _hot_pixels(img, prob, rng):
    """随机坏点"""
    h, w = img.shape[:2]
    n = int(h * w * prob)
    if n == 0:
        return img
    out = img.copy()
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    vals = rng.integers(200, 256, size=n)
    for y, x, v in zip(ys, xs, vals):
        out[y, x, :] = v
    return out


def _motion_blur(img, kernel_size):
    """水平运动模糊"""
    if kernel_size <= 1:
        return img
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = convolve(img[:, :, c].astype(np.float32), kernel, mode='mirror')
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg_cycle(img, quality):
    """一次 JPEG 压缩-解压循环"""
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format='JPEG', quality=quality, optimize=False)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def simulate_camera(img_arr, bayer=True, chroma_strength=0.8, vignette_strength=0.12,
                    iso_scale=1.0, read_noise=1.5, hot_pixel_prob=1e-7,
                    banding_strength=0.0, motion_blur_kernel=1,
                    jpeg_cycles=1, jpeg_qmin=90, jpeg_qmax=96, seed=None):
    """完整的相机管线模拟"""
    import cv2
    rng = get_rng(seed)
    out = img_arr.copy()

    # 1. Bayer
    if bayer:
        try:
            mosaic = _bayer_mosaic(out)
            if cv2.getVersion():
                dem = cv2.demosaicing(mosaic, cv2.COLOR_BAYER_RG2BGR)
                out = dem[:, :, ::-1]  # BGR → RGB
            else:
                out = _demosaic_bilinear(mosaic)
        except Exception:
            pass

    # 2. 色差
    if chroma_strength > 0:
        out = _chromatic_aberration(out, chroma_strength, rng)

    # 3. 暗角
    if vignette_strength > 0:
        out = _vignette(out, vignette_strength)

    # 4. 传感器噪声
    out = _sensor_noise(out, iso_scale, read_noise, rng)

    # 5. 坏点
    if hot_pixel_prob > 0:
        out = _hot_pixels(out, hot_pixel_prob, rng)

    # 6. 运动模糊
    if motion_blur_kernel > 1:
        out = _motion_blur(out, motion_blur_kernel)

    # 7. JPEG 循环
    for _ in range(max(1, jpeg_cycles)):
        q = int(rng.integers(jpeg_qmin, jpeg_qmax + 1))
        out = _jpeg_cycle(out, q)

    return out


# ============================================================
# 简单操作：噪声、扰动、白平衡、EXIF
# ============================================================

def add_gaussian_noise(img, std_frac=0.01, seed=None):
    rng = get_rng(seed)
    std = std_frac * 255.0
    noise = rng.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_perturbation(img, magnitude=0.005, seed=None):
    rng = get_rng(seed)
    mag = magnitude * 255.0
    perturb = rng.uniform(-mag, mag, img.shape)
    return np.clip(img.astype(np.float32) + perturb, 0, 255).astype(np.uint8)


def auto_white_balance(img):
    """灰世界假设：各通道均值拉向 128"""
    img_f = img.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)
    scale = 128.0 / (means + 1e-6)
    return np.clip(img_f * scale, 0, 255).astype(np.uint8)


def add_fake_exif(img):
    """写入假相机 EXIF"""
    import random
    import io

    try:
        import piexif
    except ImportError:
        return img  # 没装 piexif 就跳过

    # 品牌型号严格配对，避免 Sony + X-T4 这种自相矛盾
    # 品牌、型号、镜头卡口严格配对
    camera_pool = [
        ("Canon", "Canon EOS 5D Mark III", "EF"),
        ("Canon", "Canon EOS R6", "RF"),
        ("Nikon", "Nikon D850", "AF-S"),
        ("Nikon", "Nikon Z6 II", "NIKKOR Z"),
        ("Sony", "Sony Alpha 7R IV", "FE"),
        ("Sony", "Sony A7 III", "FE"),
        ("Fujifilm", "Fujifilm X-T4", "XF"),
        ("Fujifilm", "Fujifilm X-T5", "XF"),
        ("Olympus", "Olympus OM-D E-M1 Mark III", "M.Zuiko"),
        ("Leica", "Leica Q2", "Summilux"),
    ]
    make, model, mount = random.choice(camera_pool)

    # 生成合理的拍摄参数组合
    focal = random.randint(24, 135)
    fnumber = random.choice([1.4, 1.8, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0])
    iso = random.choice([100, 200, 400, 800, 1600])
    shutter_denom = random.choice([60, 125, 250, 500, 1000, 2000])

    # 镜头名匹配卡口
    if mount == "Summilux":
        lens = f"Leica {mount} {focal}mm f/{fnumber} ASPH."
    elif mount == "M.Zuiko":
        lens = f"Olympus {mount} Digital ED {focal}mm f/{fnumber} PRO"
    else:
        lens = f"{mount} {focal}mm f/{fnumber}"

    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: make,
            piexif.ImageIFD.Model: model,
            piexif.ImageIFD.Software: "Adobe Lightroom Classic",
        },
        "Exif": {
            piexif.ExifIFD.FNumber: (int(fnumber * 10), 10),
            piexif.ExifIFD.ExposureTime: (1, shutter_denom),
            piexif.ExifIFD.ISOSpeedRatings: iso,
            piexif.ExifIFD.FocalLength: (focal, 1),
            piexif.ExifIFD.FocalLengthIn35mmFilm: (focal, 1),
            piexif.ExifIFD.LensModel: lens,
        },
    }
    exif_bytes = piexif.dump(exif_dict)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_bytes)
    buf.seek(0)
    return Image.open(buf)


# ============================================================
# 主流程
# ============================================================

def process(input_path, output_path, params=None):
    if params is None:
        params = PARAMS

    print(f"读取: {input_path}")
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    print(f"  尺寸: {arr.shape[1]}x{arr.shape[0]}")

    rng = get_rng(params.get("seed"))

    # 1. DFL 频谱对抗
    if params.get("dfl_enabled", False):
        info("DFL 频谱对抗...")
        arr = dfl_attack(arr, iterations=params.get("dfl_iterations", 80),
                         strength=params.get("dfl_strength", 0.6),
                         seed=rng.integers(0, 2**31))

    # 2. FFT 频谱匹配
    if params.get("fft_enabled", True):
        info("FFT 频谱匹配...")
        arr = fft_spectrum_match(
            arr,
            mode=params.get("fft_mode", "model"),
            alpha=params.get("fft_alpha", 1.0),
            cutoff=params.get("fft_cutoff", 0.25),
            strength=params.get("fft_strength", 0.5),
            randomness=params.get("fft_randomness", 0.03),
            radial_smooth=params.get("fft_radial_smooth", 7),
            seed=rng.integers(0, 2**31),
        )

    # 2. 相机管线模拟
    if params.get("camera_enabled", True):
        info("相机管线模拟...")
        arr = simulate_camera(
            arr,
            bayer=params.get("bayer", True),
            chroma_strength=params.get("chroma_strength", 0.8),
            vignette_strength=params.get("vignette_strength", 0.12),
            iso_scale=params.get("iso_scale", 1.0),
            read_noise=params.get("read_noise", 1.5),
            hot_pixel_prob=params.get("hot_pixel_prob", 1e-7),
            banding_strength=params.get("banding_strength", 0.0),
            motion_blur_kernel=params.get("motion_blur_kernel", 1),
            jpeg_cycles=params.get("jpeg_cycles", 1),
            jpeg_qmin=params.get("jpeg_qmin", 90),
            jpeg_qmax=params.get("jpeg_qmax", 96),
            seed=rng.integers(0, 2**31),
        )

    # 3. GLCM 纹理归一化
    if params.get("glcm_enabled", False):
        info("GLCM 纹理归一化...")
        arr = glcm_normalize(arr, strength=params.get("glcm_strength", 0.45),
                             seed=rng.integers(0, 2**31))

    # 4. LBP 纹理归一化
    if params.get("lbp_enabled", False):
        info("LBP 纹理归一化...")
        arr = lbp_normalize(arr, strength=params.get("lbp_strength", 0.35),
                            seed=rng.integers(0, 2**31))

    # 5. 白平衡
    if params.get("awb_enabled", True):
        info("自动白平衡...")
        arr = auto_white_balance(arr)

    # 6. 高斯噪声
    if params.get("noise_enabled", True):
        info("高斯噪声注入...")
        arr = add_gaussian_noise(arr, std_frac=params.get("noise_std_frac", 0.01),
                                 seed=rng.integers(0, 2**31))

    # 7. 像素扰动
    if params.get("perturb_enabled", True):
        info("像素扰动...")
        arr = random_perturbation(arr, magnitude=params.get("perturb_magnitude", 0.005),
                                  seed=rng.integers(0, 2**31))

    # 保存
    out_img = Image.fromarray(arr)

    if params.get("add_exif", True):
        info("写入假 EXIF...")
        out_img = add_fake_exif(out_img)

    exif_data = out_img.info.get('exif')
    out_img.save(output_path, quality=95, exif=exif_data)
    print(f"\n保存: {output_path}")
    print("完成！")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python bypass_detector.py <输入图> <输出图>")
        print("示例: python bypass_detector.py ai_image.jpg output.jpg")
        sys.exit(1)

    process(sys.argv[1], sys.argv[2])
