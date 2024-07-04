import taichi as ti
import taichi.math as tm

MAX_N = 1024
MAX_M = 1024
MAX_GRID_L = 256
grid = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_N, MAX_M, MAX_GRID_L))
grid_ = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_N, MAX_M, MAX_GRID_L))
grid_shape = ti.field(dtype=ti.i32, shape=3)
weights = ti.field(dtype=ti.f32, shape=(2, 512), offset=(0, -256))

# @ti.func
# def intensity(v: tm.vec3):
#     return tm.dot(v, tm.vec3(0.299, 0.587, 0.114)) * 255.0

@ti.func
def interpolate2d(i: float, j: float, k: int):
    g = ti.static(grid_)
    i_, j_ = int(i), int(j)
    v1 = tm.mix(g[i_, j_, k], g[i_ + 1, j_, k], tm.fract(i))
    v2 = tm.mix(g[i_, j_ + 1, k], g[i_ + 1, j_ + 1, k], tm.fract(i))
    return tm.mix(v1, v2, tm.fract(j))

@ti.func
def interpolate3d(i: float, j: float, k: float):
    v1 = interpolate2d(i, j, int(k))
    v2 = interpolate2d(i, j, int(k) + 1)
    return tm.mix(v1, v2, tm.fract(k))

@ti.func
def compute_weights(radius: int, sigma_space: float, sigma_range: float):
    total = 0.0
    assert 0 < radius < 256
    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        val = ti.exp(-0.5 * (i / sigma_space)**2)
        weights[0, i] = val
        total += val
    
    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        weights[0, i] /= total

    total = 0.0
    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        val = ti.exp(-0.5 * (i / sigma_range)**2)
        weights[1, i] = val
        total += val

    ti.loop_config(serialize=True)
    for i in range(-radius, radius + 1):
        weights[1, i] /= total

@ti.kernel
def biliteral_grid_filter(
    original: ti.types.ndarray(dtype=ti.f32, ndim=2),
    h: int, w: int,
    sigma_space: float,
    sigma_range: float,
    s_s: int,
    s_r: int,
    out: ti.types.ndarray(dtype=ti.f32, ndim=2)
):

    assert 0 < s_s and 0 <= s_r < 256
    assert h < MAX_N and w < MAX_M
    
    # downsample
    grid_n = (h + s_s - 1) // s_s # ceil
    grid_m = (w + s_s - 1) // s_s # ceil
    grid_k = (255 + s_r - 1) // s_r # ceil 
    grid_shape[0] = grid_n
    grid_shape[1] = grid_m
    grid_shape[2] = grid_k

    for i, j, k in ti.ndrange(grid_n, grid_m, grid_k):
        grid[i, j, k] = tm.vec2(0)
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_k):
        grid_[i, j, k] = tm.vec2(0)

    for i, j in ti.ndrange(h, w):
        lum = original[i, j]
        i_ = ti.round(i / s_s, ti.i32)
        j_ = ti.round(j / s_s, ti.i32)
        k_ = ti.round(lum / s_r, ti.i32)
        grid[i_, j_, k_] += tm.vec2(lum, 1)

    assert grid_k < MAX_GRID_L
    
    # biliteral filter
    r = ti.ceil(sigma_space * 3, ti.i32)
    compute_weights(r, sigma_space, sigma_range)
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_k):
        total = tm.vec2(0)
        for l in range(max(0, i - r), min(grid_n, i + r + 1)):
            total += grid[l, j, k] * weights[0, i - l]
        grid_[i, j, k] = total
    
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_k):
        total = tm.vec2(0)
        for l in range(max(0, j - r), min(grid_m, j + r + 1)):
            total += grid_[i, l, k] * weights[0, j - l]
        grid[i, j, k] = total
    
    r = ti.ceil(sigma_range * 3, ti.i32)
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_k):
        total = tm.vec2(0)
        for l in range(max(0, k - r), min(grid_k, k + r + 1)):
            total += grid[i, j, l] * weights[1, k - l]
        grid_[i, j, k] = total

    for i, j, in ti.ndrange(h, w):
        lum = original[i, j]
        sample = interpolate3d(i / s_s, j / s_s, lum / s_r)
        out[i, j] = sample[0] / sample[1]
    