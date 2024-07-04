import numpy as np
from typing import Any
import imageio.v3 as iio
import taichi as ti
import taichi.math as tm
import sys


np.random.seed(2231)
ti.init(arch=ti.gpu, debug=True)
assert len(sys.argv) == 2, "Please specify the input image path"
img = iio.imread(sys.argv[1]).astype(dtype=np.float32) / 255.0
h, w, c = img.shape
import biliteral_grid
print(f"{img.shape=}")

out = ti.Vector.field(3, shape=(h, w), dtype=ti.f32)
out_ = ti.Vector.field(3, shape=(h, w), dtype=ti.f32)
MAX_SIGMA_SPACE = 30
weights = ti.field(dtype=ti.f32, shape=2 * 3 * MAX_SIGMA_SPACE + 1, offset=-3 * MAX_SIGMA_SPACE)
@ti.func
def compute_weights(radius: int, sigma: float):
    total = 0.0
    assert 0 < radius <= 3 * MAX_SIGMA_SPACE, f'{radius=}'
    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        val = ti.exp(-0.5 * (i / sigma)**2)
        weights[i] = val
        total += val
        
    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        weights[i] /= total

@ti.kernel
def gaussian_filter(
    original: ti.types.ndarray(dtype=tm.vec3, ndim=2), 
    h: int, w: int,
    sigma_space: float
):
    r = ti.ceil(sigma_space * 3, int)
    compute_weights(r, sigma_space)
    for i, j in ti.ndrange(h, w):
        total_rgb = tm.vec3(0)
        for k in range(max(0, i - r), min(h, i + r + 1)):
            total_rgb += original[k, j] * weights[k - i]
        out[i, j] = total_rgb
    
    for i, j in ti.ndrange(h, w):
        total_rgb = tm.vec3(0)
        for k in range(max(0, j - r), min(w, j + r + 1)):
            total_rgb += out[i, k] * weights[k - j]
        out[i, j] = total_rgb

    for i, j in ti.ndrange(h, w):
        total_rgb = tm.vec3(0)
        for k in range(max(0, i - r), min(h, i + r + 1)):
            total_rgb += weights[k - i]
        out_[i, j] = total_rgb
    
    for i, j in ti.ndrange(h, w):
        total_rgb = tm.vec3(0)
        for k in range(max(0, j - r), min(w, j + r + 1)):
            total_rgb += out_[i, k] * weights[k - j]
        out_[i, j] = total_rgb
    
    for i, j in ti.ndrange(h, w):
        out[i, j] /= out_[i, j]

@ti.kernel
def gaussian_filter_naive(
    original: ti.types.ndarray(dtype=tm.vec3, ndim=2), 
    h: int, w: int,
    sigma_space: float
):
    r = ti.ceil(sigma_space * 3, int)
    for i, j in ti.ndrange(h, w):
        acc = tm.vec3(0)
        w_p = 0.0
        for k in range(max(0, i - r), min(h, i + r + 1)):
            for l in range(max(0, j - r), min(w, j + r + 1)):
                g = ti.exp(-((i - k) ** 2 + (j - l) ** 2) / (2 * sigma_space**2))
                w_p += g
                acc += original[k, l] * g
        # out[i, j] = acc / w_p
        out[i, j] = acc / w_p
        out_[i, j] = w_p


@ti.func
def intensity(v: tm.vec3):
    return tm.dot(v, tm.vec3(0.299, 0.587, 0.114)) * 255.0


@ti.kernel
def biliteral_filter(
    original: ti.types.ndarray(dtype=tm.vec3, ndim=2), 
    sigma_space: float,
    sigma_range: float,
):
    r = ti.ceil(sigma_space * 3)
    for i, j in ti.ndrange(h, w):
        acc = tm.vec3(0)
        w_p = 0.0
        i_p = intensity(original[i, j])
        for k in range(max(0, i - r), min(h, i + r + 1)):
            for l in range(max(0, j - r), min(w, j + r + 1)):
                g = ti.exp(
                    -((i - k) ** 2 + (j - l) ** 2) / (2 * sigma_space**2)
                    - ((i_p - intensity(original[k, l])) ** 2) / (2 * sigma_range**2)
                )
                w_p += g
                acc += original[k, l] * g
        out[i, j] = acc / w_p


window = ti.ui.Window(f"Biliteral Filter ({w}x{h})", (2 * w, 2 * h), vsync=True, fps_limit=60)
canvas = window.get_canvas()
gui = window.get_gui()
sigma_space = 1.0
sigma_range = 1.0
scale = 1.0
s_s, s_r = 16, 16
mode = 'gaussian'

while window.running:
    match mode:
        case 'gaussian':
            gaussian_filter(img, h, w,sigma_space + 1e-6)
            smoothed = out.to_numpy()
        case 'biliteral':
            biliteral_filter(img, sigma_space + 1e-6, sigma_range + 1e-6)
            smoothed = out.to_numpy()
        case 'biliteral_grid':
            smoothed = np.empty((3, h, w), dtype=np.float32)
            for c in range(3):
                channel = np.ascontiguousarray(img[:, :, c] * 255)
                biliteral_grid.biliteral_grid_filter(channel, h, w, sigma_space + 1e-6, sigma_range + 1e-6, s_s, s_r, smoothed[c])
            # print(f'{np.mean(smoothed)=}, {np.std(smoothed)=}, {np.max(smoothed)=}, {np.min(smoothed)=}')
            smoothed = smoothed.transpose(1, 2, 0) / 255.0


    smoothed = np.ascontiguousarray(np.transpose(smoothed, (1, 0, 2))[:, ::-1, :])
    smoothed = np.clip(smoothed, 0, 1)
    img_ = img.transpose(1, 0, 2)[:, ::-1, :]
    detail = img_ - smoothed + 0.5
    enhanced = np.clip(smoothed + scale * (detail - 0.5), 0, 1)
    row1 = np.concatenate([img_, smoothed])
    row2 = np.concatenate([detail, enhanced])

    canvas.set_image(np.concatenate([row2, row1], axis=1))
    with gui.sub_window("original", 0, 0, 0.2, 0.1) as subwindow:
        ...
    with gui.sub_window("smoothed", 0.5, 0, 0.4, 0.16) as subwindow:

        match subwindow.slider_int(f'mode: {mode}', {'gaussian': 0, 'biliteral': 1, 'biliteral_grid': 2}[mode], 0, 2):
            case 0:
                mode = 'gaussian'
                sigma_space = subwindow.slider_float("sigma", sigma_space, 0, MAX_SIGMA_SPACE)
                sigma_range = sigma_space
            case 1 | 2 as x:
                if x == 1:
                    mode = 'biliteral'
                else:
                    mode = 'biliteral_grid'
                    subwindow.text(f'grid_shape={biliteral_grid.grid_shape}')
                    s_s = subwindow.slider_int("s_s", s_s, 1, 32)
                    s_r = subwindow.slider_int("s_r", s_r, 1, 32)
                sigma_space = subwindow.slider_float("sigma_space", sigma_space, 0, MAX_SIGMA_SPACE)
                sigma_range = subwindow.slider_float("sigma_range", sigma_range, 0, 512)
            

    with gui.sub_window("detail = original - smoothed", 0, 0.5, 0.4, 0) as subwindow:
        ...

    with gui.sub_window(
        "enhanced = original + s * detail", 0.5, 0.5, 0.4, 0.1
    ) as subwindow:
        scale = subwindow.slider_float("s", scale, 0, 10)
    window.show()
