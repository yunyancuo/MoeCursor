# -*- coding: utf-8 -*-
# 娘化角色整套鼠标指针 v4 —— 参考"小恐龙贴纸"风格：白底深蓝描边 + 左上角小三角
import struct, math, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

girl_src = np.asarray(Image.open('character_square.png').convert('RGBA')).astype(np.float32)
hsv = cv2.cvtColor(girl_src[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
hsv = cv2.cvtColor(girl_src[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
hsv[..., 1] = np.clip(hsv[..., 1] * 1.35, 0, 255)
rgb2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
rgb2 = np.clip((rgb2 - 128) * 1.14 + 128, 0, 255)   # 轻微对比, 抵消缩放灰化
girl = girl_src.copy(); girl[..., :3] = rgb2
# 轻度去噪: 洗掉原 JPEG 截图的压缩噪点/色块, 减少小尺寸失真感
girl[..., :3] = cv2.fastNlMeansDenoisingColored(girl[..., :3].astype(np.uint8), None, 4, 4, 7, 21).astype(np.float32)

NAVY  = (78, 118, 190, 255)
NAVYD = (45, 92, 170, 255)
WHITE = (255, 255, 255, 255)
BLUEF = (168, 210, 250, 255)
RED   = (235, 84, 96, 255)
SHC   = (15, 20, 40, 120)   # 阴影色
HANDFILL = (219, 235, 252, 255)
IBFILL   = (228, 240, 253, 255)

def premul_resize(a, size, size2=None):
    size2 = size2 or size
    r3, al = a[..., :3] * a[..., 3:] / 255.0, a[..., 3]
    inter = cv2.INTER_AREA if size2 < a.shape[0] else cv2.INTER_LANCZOS4
    r = cv2.resize(r3, (size2, size), interpolation=inter)
    m = cv2.resize(al, (size2, size), interpolation=inter)
    out = np.zeros((size, size2, 4), np.float32)
    nz = m > 4
    out[..., :3][nz] = (r[nz] / (m[nz][:, None] / 255.0)).clip(0, 255)
    out[..., 3] = m.clip(0, 255)
    out[out[..., 3] < 8] = 0
    # unsharp: 缩小会吃掉深色线稿, 锐化保住线条对比(只作用于不透明区)
    blur = cv2.GaussianBlur(out[..., :3], (0, 0), 1.0)
    mask3 = (out[..., 3:4] > 40).astype(np.float32)
    out[..., :3] = np.clip(out[..., :3] + (out[..., :3] - blur) * 1.05 * mask3, 0, 255)
    return out.astype(np.float32)

def fill_holes(mask):
    # 角色头发贴着画布左上角，必须先加外框再 flood，否则整个精灵框会被当洞填白
    padded = np.pad((~mask).astype(np.uint8), 1, constant_values=1)
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 2)
    holes = (padded == 1)[1:-1, 1:-1]
    return mask | holes

def over(base, top):
    la = top[..., 3:4] / 255.0
    base[..., :3] = top[..., :3] * la + base[..., :3] * (1 - la)
    base[..., 3:4] = top[..., 3:4] + base[..., 3:4] * (1 - la)
    return base

def add_shadow(canvas, strength=0.7):
    S = canvas.shape[0]
    a = canvas[..., 3] / 255.0
    blur = cv2.GaussianBlur(a, (0, 0), max(1.0, S / 44.0))
    sh = max(2, round(S / 40))
    shm = np.zeros_like(blur)
    if sh < S:
        shm[sh:, sh:] = blur[:-sh, :-sh]
    sa = np.clip(shm * strength, 0, 1) * (1 - a)
    canvas[..., :3] = canvas[..., :3] * a[..., None] + np.float32([20, 24, 40]) * sa[..., None]
    canvas[..., 3] = np.clip(a + sa, 0, 1) * 255
    return canvas

# 高清轮廓掩码 + 扣图清理:
# 1) 原图从白底 JPEG 抠出时, 发丝周围留着灰白毛边 —— 在 426px 高清上腐蚀掉这条带
# 2) 颜色层再羽化, 得到干净柔和的边缘
base_mask = fill_holes(girl_src[..., 3] > 100).astype(np.uint8)
er_color = cv2.erode(base_mask, np.ones((11, 11), np.uint8))   # 颜色层: 多腐蚀去毛边
er_ring = cv2.erode(base_mask, np.ones((7, 7), np.uint8))      # 白描边: 少腐蚀一点, 露出干净白边
girl[..., 3] = np.clip(cv2.GaussianBlur(er_color * 255, (0, 0), 1.6), 0, 255)  # 羽化
hi_solid = (er_ring * 255).astype(np.float32)
# 头部特写(小尺寸探头用)——保留全宽发丝/呆毛, 轮廓才有形状特征
head = girl_src[0:320, 0:426]
hi_head = fill_holes(head[..., 3] > 80).astype(np.float32)
hi_head = cv2.erode(hi_head, np.ones((2, 2), np.uint8)) * 255

def sticker_canvas(S, img, hi, gh_scale, gx_frac, gy_frac):
    """实心贴纸(填孔+白描边+深蓝细边)放在 S×S 画布上, 支持非方形素材(gh_scale=高度占比)"""
    gh = int(round(S * gh_scale))
    gw = int(round(gh * img.shape[1] / img.shape[0]))
    g = premul_resize(img, gh, gw)
    solid = cv2.resize(hi, (gw, gh), interpolation=cv2.INTER_AREA) > 80
    r_white = max(2, round(S / 24))
    ring_white = cv2.dilate(solid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_white+1, 2*r_white+1))) > 0
    gx, gy = int(round(S * gx_frac)), int(round(S * gy_frac))
    def put2(m):
        c = np.zeros((S, S), bool)
        sy0, sx0 = max(0, -gy), max(0, -gx)
        sy1, sx1 = gh - max(0, gy + gh - S), gw - max(0, gx + gw - S)
        c[max(0,gy):max(0,gy)+sy1-sy0, max(0,gx):max(0,gx)+sx1-sx0] = m[sy0:sy1, sx0:sx1]
        return c
    def put4(m4):
        c = np.zeros((S, S, 4), np.float32)
        sy0, sx0 = max(0, -gy), max(0, -gx)
        sy1, sx1 = gh - max(0, gy + gh - S), gw - max(0, gx + gw - S)
        c[max(0,gy):max(0,gy)+sy1-sy0, max(0,gx):max(0,gx)+sx1-sx0] = m4[sy0:sy1, sx0:sx1]
        return c
    wh = np.zeros((S, S, 4), np.float32); wh[..., :3] = 255; wh[..., 3] = put2(ring_white | solid) * 255
    canvas = over(np.zeros((S, S, 4), np.float32), wh)
    return over(canvas, put4(g))

def girl_canvas(S, gh_scale, gx_frac, gy_frac):
    return sticker_canvas(S, girl, hi_solid, gh_scale, gx_frac, gy_frac)

def peek_canvas(S, gh_scale, gx_frac, gy_frac):
    """完整角色贴纸(全身, 白边贴合完整轮廓); 32px 小帧不放角色保证字形干净"""
    if S < 48:
        return np.zeros((S, S, 4), np.float32)
    return girl_canvas(S, gh_scale, gx_frac, gy_frac)

def paste_at(canvas, glyph, x, y):
    gh, gw = glyph.shape[:2]
    sy0, sx0 = max(0, -y), max(0, -x)
    sy1 = gh - max(0, y + gh - canvas.shape[0]); sx1 = gw - max(0, x + gw - canvas.shape[1])
    dy0, dx0 = max(0, y), max(0, x)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    sub = glyph[sy0:sy1, sx0:sx1]
    region = canvas[dy0:dy0+sub.shape[0], dx0:dx0+sub.shape[1]].copy()
    canvas[dy0:dy0+sub.shape[0], dx0:dx0+sub.shape[1]] = over(region, sub)

def paste_center(canvas, glyph, fx, fy):
    gh, gw = glyph.shape[:2]
    paste_at(canvas, glyph, int(round(canvas.shape[1]*fx - gw/2)), int(round(canvas.shape[0]*fy - gh/2)))

def finish(canvas, strength=0.7):
    return np.clip(add_shadow(np.clip(canvas, 0, 255), strength), 0, 255).astype(np.uint8)

# ---------------- 字形(高分辨率绘制后缩小) ----------------
ARROW = [(0.0, 0.0), (0.0, 0.86), (0.19, 0.70), (0.32, 1.00), (0.41, 0.96), (0.29, 0.67), (0.54, 0.67)]

def arrow_glyph(h, ow, fill=WHITE):
    """经典指针三角箭头，可见尖端位于 glyph 的 (1,1) 附近"""
    k = 4
    W = int(h * 0.54 * k) + int(ow * k) * 2 + 10 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = int(ow * k)
    # 尖端圆弧化(可爱风): 用弧线替换尖角, 其余角保持锐利
    pts = list(ARROW)
    v = pts[0]; prv = pts[-1]; nxt = pts[1]
    d1 = (prv[0]-v[0], prv[1]-v[1]); l1 = math.hypot(*d1); d1 = (d1[0]/l1, d1[1]/l1)
    d2 = (nxt[0]-v[0], nxt[1]-v[1]); l2 = math.hypot(*d2); d2 = (d2[0]/l2, d2[1]/l2)
    rr = 0.10
    p1 = (v[0]+d1[0]*rr, v[1]+d1[1]*rr)
    p2 = (v[0]+d2[0]*rr, v[1]+d2[1]*rr)
    bx, by = d1[0]+d2[0], d1[1]+d2[1]
    bl = math.hypot(bx, by)
    mid = (v[0]+bx/bl*rr*0.35, v[1]+by/bl*rr*0.35)
    pts = [p1, mid, p2] + pts[1:]
    P = [(x * h * k + pad, y * h * k + pad) for x, y in pts]
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([(px + 2*k, py + 2*k) for px, py in P], fill=SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    d.polygon(P, fill=NAVYD)
    d.line(P + [P[0]], width=max(2, int(ow * k * 2)), fill=NAVYD, joint='curve')
    d.polygon(P, fill=fill)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    return np.asarray(im).astype(np.float32)

def star_glyph(size, ow):
    k = 4; W = int(size * k) + int(ow * k) * 4 + 8 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = 0.5 if i % 2 == 0 else 0.21
        pts.append((W/2 + r * size * k * math.cos(ang), W/2 + r * size * k * math.sin(ang)))
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([(px + 2*k, py + 2*k) for px, py in pts], fill=SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    d.polygon(pts, fill=(200, 140, 20, 255))
    d.line(pts + [pts[0]], width=max(2, int(ow * k * 2)), fill=(200, 140, 20, 255), joint='curve')
    d.polygon(pts, fill=(255, 201, 64, 255))
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

def question_glyph(size, ow):
    k = 4; W = int(size * k); pad = int(ow * k) * 2 + 8 * k
    im = Image.new('RGBA', (W + pad*2, W + pad*2), (0, 0, 0, 0))
    font = ImageFont.truetype('C:/Windows/Fonts/seguibl.ttf', int(W * 0.95))
    sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((pad + 2*k, pad + 2*k), '?', font=font, fill=SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    d = ImageDraw.Draw(im)
    d.text((pad, pad), '?', font=font, fill=WHITE, stroke_width=max(2, int(ow * k)), stroke_fill=WHITE)
    d.text((pad, pad), '?', font=font, fill=NAVYD, stroke_width=1, stroke_fill=NAVYD)
    im = im.resize((im.width // k, im.height // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

def ring_glyph(size, ow):
    k = 4; W = int(size * k) + int(ow * k) * 4 + 10 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = W / 2; r = size * k / 2
    w = max(3, int(ow * k * 2))
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([c-r+2*k, c-r+2*k, c+r+2*k, c+r+2*k], outline=(120, 20, 30, 140), width=w)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    d.ellipse([c-r, c-r, c+r, c+r], outline=RED, width=w)
    a45 = math.radians(45)
    d.line([c - r*0.72*math.cos(a45), c - r*0.72*math.sin(a45), c + r*0.72*math.cos(a45), c + r*0.72*math.sin(a45)], fill=RED, width=w)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    return np.asarray(im).astype(np.float32)

def crosshair_glyph(S, ow):
    k = 4; W = int(S * k) + int(ow * k) * 4 + 16 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = W / 2; L = S * k; th = S * 0.11 * k
    def cross(dd, off, col, extra=0):
        w = int(th + extra)
        dd.line([c - L/2 + off, c + off, c + L/2 + off, c + off], width=w, fill=col)
        dd.line([c + off, c - L/2 + off, c + off, c + L/2 + off], width=w, fill=col)
        for p in [(c-L/2, c), (c+L/2, c), (c, c-L/2), (c, c+L/2)]:
            r = w / 2
            dd.ellipse([p[0]-r+off, p[1]-r+off, p[0]+r+off, p[1]+r+off], fill=col)
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    cross(ImageDraw.Draw(sh), 2*k, SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    cross(d, 0, NAVYD, ow * k * 2); cross(d, 0, WHITE)
    r = S * 0.09 * k
    d.polygon([(c-r, c-r*0.7), (c, c), (c-r, c+r*0.7)], fill=NAVYD)
    d.polygon([(c+r, c-r*0.7), (c, c), (c+r, c+r*0.7)], fill=NAVYD)
    d.ellipse([c-r*0.28, c-r*0.28, c+r*0.28, c+r*0.28], fill=NAVYD)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    g = np.asarray(im.crop(bb)).astype(np.float32)
    H, Wd = g.shape[:2]
    M = max(H, Wd)
    sq = np.zeros((M, M, 4), np.float32)
    sq[(M-H)//2:(M-H)//2+H, (M-Wd)//2:(M-Wd)//2+Wd] = g
    return sq

def _paint_double_arrow(dd, c, ux, uy, L, th, hd, off, edge, fill):
    x1, y1 = c - ux*L/2 + off, c - uy*L/2 + off
    x2, y2 = c + ux*L/2 + off, c + uy*L/2 + off
    px, py = -uy, ux
    dd.line([x1, y1, x2, y2], width=int(th), fill=edge)
    for p in [(x1, y1), (x2, y2)]:
        r = th / 2
        dd.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=edge)
    for sgn, p in ((-1, (x1, y1)), (1, (x2, y2))):
        tip = (p[0] + ux*sgn*hd*0.62, p[1] + uy*sgn*hd*0.62)
        dd.polygon([tip, (p[0]+px*hd*0.46, p[1]+py*hd*0.46), (p[0]-px*hd*0.46, p[1]-py*hd*0.46)], fill=edge)
    if fill is not edge:
        dd.line([x1, y1, x2, y2], width=int(th*0.62), fill=fill)
        for p in [(x1, y1), (x2, y2)]:
            r = th * 0.31
            dd.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=fill)
        for sgn, p in ((-1, (x1, y1)), (1, (x2, y2))):
            tip = (p[0] + ux*sgn*hd*0.62, p[1] + uy*sgn*hd*0.62)
            dd.polygon([tip, (p[0]+px*hd*0.46, p[1]+py*hd*0.46), (p[0]-px*hd*0.46, p[1]-py*hd*0.46)], fill=fill)

def bar_with_heads(S, angle_deg, ow):
    k = 4
    L, th, hd = S * 0.30 * k, S * 0.13 * k, S * 0.40 * k
    W = int(S * 1.15 * k)
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    c = W / 2
    owk = max(2, int(ow * k))
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    _paint_double_arrow(ImageDraw.Draw(sh), c, ux, uy, L, th + owk*2, hd + owk, 2*k, SHC, SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    _paint_double_arrow(ImageDraw.Draw(im), c, ux, uy, L, th + owk*2, hd + owk, 0, NAVYD, NAVYD)
    _paint_double_arrow(ImageDraw.Draw(im), c, ux, uy, L, th, hd, 0, HANDFILL, HANDFILL)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

def four_arrow_glyph(S, ow):
    k = 4
    L, th, hd = S * 0.34 * k, S * 0.15 * k, S * 0.36 * k
    W = int(S * 1.15 * k)
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    c = W / 2
    owk = max(2, int(ow * k))
    def paint(dd, off, edge, fill):
        solid = fill is edge
        for ux, uy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x1, y1 = c + ux*S*0.08*k + off, c + uy*S*0.08*k + off
            x2, y2 = c + ux*L/2 + off, c + uy*L/2 + off
            px, py = -uy, ux
            w = th if solid else th + owk*2
            dd.line([x1, y1, x2, y2], width=int(w), fill=edge)
            dd.ellipse([x1-w/2, y1-w/2, x1+w/2, y1+w/2], fill=edge)
            tip = (x2 + ux*hd*0.62, y2 + uy*hd*0.62)
            dd.polygon([tip, (x2+(-uy)*hd*0.46, y2+ux*hd*0.46), (x2-(-uy)*hd*0.46, y2-ux*hd*0.46)], fill=edge)
        if not solid:
            for ux, uy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x1, y1 = c + ux*S*0.08*k + off, c + uy*S*0.08*k + off
                x2, y2 = c + ux*L/2 + off, c + uy*L/2 + off
                dd.line([x1, y1, x2, y2], width=int(th*0.62), fill=fill)
                r = th * 0.31
                dd.ellipse([x1-r, y1-r, x1+r, y1+r], fill=fill)
                tip = (x2 + ux*hd*0.62, y2 + uy*hd*0.62)
                dd.polygon([tip, (x2+(-uy)*hd*0.46, y2+ux*hd*0.46), (x2-(-uy)*hd*0.46, y2-ux*hd*0.46)], fill=fill)
            r = th * 0.42
            dd.ellipse([c-r, c-r, c+r, c+r], fill=fill)
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    paint(ImageDraw.Draw(sh), 2*k, SHC, SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    paint(ImageDraw.Draw(im), 0, NAVYD, NAVYD)
    paint(ImageDraw.Draw(im), 0, HANDFILL, HANDFILL)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

def ibeam_glyph(S, ow):
    k = 4; W = int(S * k); pad = int(ow * k) * 2 + 8 * k
    im = Image.new('RGBA', (W + pad*2, W + pad*2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    def ib(col, extra, off):
        x = pad + W/2 + off; top, bot = pad + off, pad + W + off
        w = int(S * 0.060 * k) + extra
        d.line([x, top + w, x, bot - w], width=w, fill=col)
        d.line([x - w*0.78, top + w*0.5, x + w*0.78, top + w*0.5], width=w, fill=col)
        d.line([x - w*0.78, bot - w*0.5, x + w*0.78, bot - w*0.5], width=w, fill=col)
        for sx in (x - w*0.78, x + w*0.78):
            for sy in (top + w*0.5, bot - w*0.5):
                d.ellipse([sx-w/2, sy-w/2, sx+w/2, sy+w/2], fill=col)
    sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).line([pad + W/2 + 2*k, pad + 2*k, pad + W/2 + 2*k, pad + W + 2*k], width=int(S*0.16*k), fill=SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    ib(NAVYD, max(2, int(ow * k * 2)), 0)
    ib(IBFILL, 0, 0)
    im = im.resize((im.width // k, im.height // k), Image.LANCZOS)
    bb = im.getbbox()
    g = np.asarray(im.crop(bb)).astype(np.float32)
    H, Wd = g.shape[:2]
    M = max(H, Wd)
    sq = np.zeros((M, M, 4), np.float32)
    sq[(M-H)//2:(M-H)//2+H, (M-Wd)//2:(M-Wd)//2+Wd] = g
    return sq

def pencil_glyph(S, ow):
    k = 4; W = int(S * k) + int(ow * k) * 4 + 12 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    a = math.radians(40)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    o = int(ow * k) * 2 + 6 * k
    tip = (o + S*0.06*k, o + S*0.06*k)
    bs = (tip[0] + dx*S*0.24*k, tip[1] + dy*S*0.24*k)
    be = (bs[0] + dx*S*0.56*k, bs[1] + dy*S*0.56*k)
    ee = (be[0] + dx*S*0.13*k, be[1] + dy*S*0.13*k)
    w = S * 0.20 * k
    def quad(p1, p2, col, extra):
        pts = [(p1[0]+nx*(w/2+extra), p1[1]+ny*(w/2+extra)), (p2[0]+nx*(w/2+extra), p2[1]+ny*(w/2+extra)),
               (p2[0]-nx*(w/2+extra), p2[1]-ny*(w/2+extra)), (p1[0]-nx*(w/2+extra), p1[1]-ny*(w/2+extra))]
        d.polygon(pts, fill=col)
    def tiptri(col, extra):
        d.polygon([tip, (bs[0]+nx*(w/2+extra), bs[1]+ny*(w/2+extra)), (bs[0]-nx*(w/2+extra), bs[1]-ny*(w/2+extra))], fill=col)
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    o2 = 2 * k
    ImageDraw.Draw(sh).polygon([(tip[0]+o2, tip[1]+o2),
                                (be[0]+nx*(w/2+2*k)+o2, be[1]+ny*(w/2+2*k)+o2),
                                (be[0]-nx*(w/2+2*k)+o2, be[1]-ny*(w/2+2*k)+o2),
                                (ee[0]+nx*(w/2+2*k)+o2, ee[1]+ny*(w/2+2*k)+o2),
                                (ee[0]-nx*(w/2+2*k)+o2, ee[1]-ny*(w/2+2*k)+o2)], fill=SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    quad(bs, be, NAVYD, 2*k); tiptri(NAVYD, 2*k); quad(be, ee, NAVYD, 2*k)
    quad(bs, be, (255, 214, 120, 255), 0); quad(be, ee, (255, 160, 170, 255), 0)
    tiptri((150, 160, 190, 255), 0)
    lw = max(2, int(ow * k))
    d.line([bs[0]+nx*w/2, bs[1]+ny*w/2, be[0]+nx*w/2, be[1]+ny*w/2], fill=NAVYD, width=lw)
    d.line([bs[0]-nx*w/2, bs[1]-ny*w/2, be[0]-nx*w/2, be[1]-ny*w/2], fill=NAVYD, width=lw)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

def up_arrow_glyph(S, ow):
    k = 4
    h = S * 0.9 * k
    W = int(h * 0.62) + int(ow * k) * 2 + 10 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = int(ow * k)
    pts = [(x * h * 0.55 + pad, y * h * 0.95 + pad) for x, y in ARROW]
    ang = math.radians(-28)
    cx, cy = pad + h*0.22, pad + h*0.33
    rot = [(cx + (px-cx)*math.cos(ang) - (py-cy)*math.sin(ang), cy + (px-cx)*math.sin(ang) + (py-cy)*math.cos(ang)) for px, py in pts]
    minx = min(p[0] for p in rot); miny = min(p[1] for p in rot)
    rot = [(p[0] - minx + pad, p[1] - miny + pad) for p in rot]
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([(p[0]+2*k, p[1]+2*k) for p in rot], fill=SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    d.polygon(rot, fill=NAVYD)
    d.line(rot + [rot[0]], width=max(2, int(ow * k * 2)), fill=NAVYD, joint='curve')
    d.polygon(rot, fill=WHITE)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

# ---------------- 动画(.ani) ----------------
ANG = [0, 4, 8, 4, 0, -4, -8, -4]
BOB = [0, 1, 2, 1, 0, -1, -2, -1]

def shift_canvas(c, dy):
    if dy == 0:
        return c
    out = np.zeros_like(c)
    if dy > 0:
        out[dy:] = c[:-dy]
    else:
        out[:dy] = c[-dy:]
    return out

def rotate_around(g, deg, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    return cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=0)

def anim_arrow_frame(S, i):
    c = shift_canvas(peek_canvas(S, 0.52, 0.40, 0.34), round(BOB[i] * S / 32))
    ow = max(2, S/34)
    aw = rotate_around(arrow_glyph(S/2.1, ow), ANG[i], 2, 2)
    paste_at(c, aw, 1 - int(ow), 1 - int(ow))
    return finish(c), (0, 0)

def anim_hand_frame(S, i):
    c = shift_canvas(peek_canvas(S, 0.42, 0.62, 0.62), round(BOB[(i+4) % 8] * S / 32))
    g, (tx, ty) = hand_glyph(S*0.60, max(2, S/34))
    gr = rotate_around(g, ANG[i], tx, ty)
    px = int(S*0.34 - gr.shape[1]/2); py = int(S*0.44 - gr.shape[0]/2)
    paste_at(c, gr, px, py)
    tip = (int(min(max(0, px + tx), S-1)), int(min(max(0, py + ty), S-1)))
    return finish(c), tip

def anim_ibeam_frame(S, i):
    c = shift_canvas(peek_canvas(S, 0.42, 0.56, 0.60), round(BOB[(i+2) % 8] * S / 32))
    g = ibeam_glyph(S*0.50, max(2, S/56))
    gr = rotate_around(g, ANG[i] * 1.5, g.shape[1]/2, g.shape[0]/2)
    paste_center(c, gr, 0.42, 0.5)
    return finish(c), (int(S*0.42), S//2)

def write_ani(path, S, frame_fn, n=8, jif=8):
    icons = []
    for i in range(n):
        px, hot = frame_fn(S, i)
        data = bmp_entry(px)
        entry = struct.pack('<BBBBHHII', S % 256, S % 256, 0, 0, hot[0], hot[1], len(data), 22)
        icons.append(struct.pack('<HHH', 0, 2, 1) + entry + data)
    def chunk(tag, payload):
        pad = b'\00' if len(payload) % 2 else b''
        return tag + struct.pack('<I', len(payload)) + payload + pad
    anih = struct.pack('<9I', 36, n, n, S, S, 0, 0, jif, 1)
    seq = b''.join(struct.pack('<I', i) for i in range(n))
    fram = b''.join(chunk(b'icon', ic) for ic in icons)
    riff = b'ACON' + chunk(b'anih', anih) + chunk(b'seq ', seq) + chunk(b'LIST', b'fram' + fram)
    open(path, 'wb').write(b'RIFF' + struct.pack('<I', len(riff)) + riff)
    print('written', path)

def heart_glyph(size, ow):
    """粉色小爱心(可爱点缀)"""
    k = 4; W = int(size * k) + int(ow * k) * 4 + 8 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = int(ow * k) * 2 + 4 * k; u = size * k
    def heart(dd, extra, col):
        dd.ellipse([pad+0.03*u-extra, pad+0.02*u-extra, pad+0.55*u+extra, pad+0.54*u+extra], fill=col)
        dd.ellipse([pad+0.45*u-extra, pad+0.02*u-extra, pad+0.97*u+extra, pad+0.54*u+extra], fill=col)
        dd.polygon([(pad+0.06*u-extra, pad+0.30*u), (pad+0.94*u+extra, pad+0.30*u), (pad+0.50*u, pad+1.02*u+extra)], fill=col)
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    heart(ImageDraw.Draw(sh), 0, SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    heart(d, int(ow*k*1.6), WHITE)
    heart(d, 0, (255, 135, 168, 255))
    d.ellipse([pad+0.16*u, pad+0.14*u, pad+0.30*u, pad+0.28*u], fill=(255, 210, 222, 255))
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

# ---------------- 各指针合成: 返回 (像素, 热点) ----------------
def compose_arrow(S):
    ow = max(2, S/30) if S < 48 else max(2, S/28)
    c = np.zeros((S, S, 4), np.float32)
    if S >= 48:
        c = peek_canvas(S, 0.60, 0.36, 0.32)
    paste_at(c, arrow_glyph(S/1.9 if S < 48 else S/1.8, ow, fill=(168, 208, 250, 255)), 1 - int(ow), 1 - int(ow))
    if S >= 48:
        paste_center(c, heart_glyph(S*0.20, max(1, S/110)), 0.74, 0.10)
    return finish(c), (0, 0)

def hand_glyph(S, ow):
    """经典链接小手(白底深蓝描边贴纸)，返回 glyph 和指尖局部坐标"""
    k = 4
    W = int(S * k) + int(ow * k) * 4 + 12 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    pad = int(ow * k) * 2 + 6 * k
    u = S * k
    d = ImageDraw.Draw(im)
    def parts(extra, col):
        # 四根手指(竖圆角条)
        d.rounded_rectangle([pad+0.15*u-extra, pad+0.03*u-extra, pad+0.31*u+extra, pad+0.64*u+extra], radius=(0.16*u)/2+extra, fill=col)
        d.rounded_rectangle([pad+0.35*u-extra, pad+0.24*u-extra, pad+0.50*u+extra, pad+0.64*u+extra], radius=(0.15*u)/2+extra, fill=col)
        d.rounded_rectangle([pad+0.54*u-extra, pad+0.28*u-extra, pad+0.68*u+extra, pad+0.64*u+extra], radius=(0.14*u)/2+extra, fill=col)
        d.rounded_rectangle([pad+0.72*u-extra, pad+0.38*u-extra, pad+0.85*u+extra, pad+0.64*u+extra], radius=(0.13*u)/2+extra, fill=col)
        # 左侧拇指鼓包
        d.ellipse([pad+0.07*u-extra, pad+0.42*u-extra, pad+0.33*u+extra, pad+0.80*u+extra], fill=col)
        # 手掌
        d.rounded_rectangle([pad+0.11*u-extra, pad+0.52*u-extra, pad+0.86*u+extra, pad+0.88*u+extra], radius=0.13*u+extra, fill=col)
        # 袖口
        d.rounded_rectangle([pad+0.24*u-extra, pad+0.86*u-extra, pad+0.78*u+extra, pad+0.99*u+extra], radius=0.09*u+extra, fill=col)
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sh)
    def shadow_parts(col):
        o = 2 * k
        ds.rounded_rectangle([pad+0.15*u+o, pad+0.03*u+o, pad+0.31*u+o, pad+0.64*u+o], radius=(0.16*u)/2, fill=col)
        ds.rounded_rectangle([pad+0.35*u+o, pad+0.24*u+o, pad+0.50*u+o, pad+0.64*u+o], radius=(0.15*u)/2, fill=col)
        ds.rounded_rectangle([pad+0.54*u+o, pad+0.28*u+o, pad+0.68*u+o, pad+0.64*u+o], radius=(0.14*u)/2, fill=col)
        ds.rounded_rectangle([pad+0.72*u+o, pad+0.38*u+o, pad+0.85*u+o, pad+0.64*u+o], radius=(0.13*u)/2, fill=col)
        ds.ellipse([pad+0.07*u+o, pad+0.42*u+o, pad+0.33*u+o, pad+0.80*u+o], fill=col)
        ds.rounded_rectangle([pad+0.11*u+o, pad+0.52*u+o, pad+0.86*u+o, pad+0.88*u+o], radius=0.13*u, fill=col)
        ds.rounded_rectangle([pad+0.24*u+o, pad+0.86*u+o, pad+0.78*u+o, pad+0.99*u+o], radius=0.09*u, fill=col)
    shadow_parts(SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    owk = int(ow * k * 2)
    parts(owk, NAVYD)
    parts(0, HANDFILL)
    # 指缝
    for x0 in (0.315, 0.52, 0.70):
        d.line([pad + x0*u, pad + 0.34*u, pad + x0*u, pad + 0.62*u], fill=NAVYD, width=max(2, int(ow * k * 0.8)))
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    g = np.asarray(im.crop(bb)).astype(np.float32)
    ys, xs = np.where(g[..., 3] > 127)
    ty = int(ys.min()); tx = float(xs[ys == ty].mean())
    return g, (tx, ty)

def hand_base(S):
    """小手+角色的公共底图，返回 (画布, 指尖热点)"""
    c = peek_canvas(S, 0.52, 0.46, 0.42)
    if S >= 48:
        paste_center(c, heart_glyph(S*0.18, max(1, S/110)), 0.88, 0.08)
    g, (tx, ty) = hand_glyph(S*0.66, max(2, S/30))
    px = int(S*0.34 - g.shape[1]/2); py = int(S*0.44 - g.shape[0]/2)
    paste_at(c, g, px, py)
    tip = (int(min(max(0, px + tx), S-1)), int(min(max(0, py + ty), S-1)))
    return c, tip

def badge_pin_glyph(S, ow):
    """定位水滴徽章(蓝)"""
    k = 4; W = int(S * k) + int(ow * k) * 4 + 12 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = int(ow * k) * 2 + 6 * k; u = S * k
    def pin(dd, extra, col):
        dd.ellipse([pad+0.17*u-extra, pad+0.02*u-extra, pad+0.83*u+extra, pad+0.68*u+extra], fill=col)
        dd.polygon([(pad+0.24*u-extra, pad+0.56*u), (pad+0.76*u+extra, pad+0.56*u), (pad+0.50*u, pad+0.99*u+extra)], fill=col)
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    pin(ImageDraw.Draw(sh), 0, SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    pin(d, int(ow*k*2.5), WHITE)
    pin(d, int(ow*k), NAVYD)
    pin(d, 0, (86, 148, 246, 255))
    d.ellipse([pad+0.38*u, pad+0.23*u, pad+0.62*u, pad+0.47*u], fill=WHITE)
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

def badge_person_glyph(S, ow):
    """人形徽章(蓝)"""
    k = 4; W = int(S * k) + int(ow * k) * 4 + 12 * k
    im = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = int(ow * k) * 2 + 6 * k; u = S * k
    def person(dd, extra, col):
        dd.ellipse([pad+0.26*u-extra, pad+0.02*u-extra, pad+0.74*u+extra, pad+0.50*u+extra], fill=col)
        dd.pieslice([pad+0.06*u-extra, pad+0.56*u-extra, pad+0.94*u+extra, pad+1.60*u+extra], 180, 360, fill=col)
    sh = Image.new('RGBA', (W, W), (0, 0, 0, 0))
    person(ImageDraw.Draw(sh), 0, SHC)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1.2 * k)))
    person(d, int(ow*k*2.5), WHITE)
    person(d, int(ow*k), NAVYD)
    person(d, 0, (110, 155, 225, 255))
    im = im.resize((W // k, W // k), Image.LANCZOS)
    bb = im.getbbox()
    return np.asarray(im.crop(bb)).astype(np.float32)

def compose_link(S):
    c, tip = hand_base(S)
    return finish(c), tip

def compose_pin(S):
    c, tip = hand_base(S)
    paste_center(c, badge_pin_glyph(S*0.28, max(2, S/52)), 0.82, 0.84)
    return finish(c), tip

def compose_person(S):
    c, tip = hand_base(S)
    paste_center(c, badge_person_glyph(S*0.30, max(2, S/52)), 0.82, 0.82)
    return finish(c), tip

def compose_help(S):
    c = peek_canvas(S, 0.58, 0.40, 0.32)
    paste_center(c, question_glyph(S*0.36, max(2, S/44)), 0.13, 0.58)
    paste_at(c, arrow_glyph(S/2.4, max(2, S/40)), 1, 1)
    return finish(c), (0, 0)

def compose_busy(S):
    c = girl_canvas(S, 0.98, 0.01, 0.01)
    if S >= 48:
        paste_center(c, heart_glyph(S*0.15, max(1, S/120)), 0.10, 0.12)
        paste_center(c, heart_glyph(S*0.10, max(1, S/150)), 0.26, 0.05)
    return finish(c), (S//2, S//2)

def compose_app(S):
    c = girl_canvas(S, 0.84, 0.08, 0.12)
    paste_center(c, star_glyph(S*0.32, max(2, S/44)), 0.86, 0.14)
    return finish(c), (S//2, S//2)

def compose_no(S):
    c = girl_canvas(S, 0.86, 0.07, 0.07)
    paste_center(c, ring_glyph(S*0.98, max(3, S/26)), 0.5, 0.5)
    return finish(c), (S//2, S//2)

def compose_cross(S):
    c = np.zeros((S, S, 4), np.float32)
    paste_center(c, crosshair_glyph(S*0.94, max(2, S/44)), 0.5, 0.5)
    return finish(c, 0.55), (S//2, S//2)

def compose_ibeam(S):
    c = peek_canvas(S, 0.52, 0.50, 0.44)
    paste_center(c, ibeam_glyph(S*0.50, max(2, S/56)), 0.42, 0.5)
    if S >= 48:
        paste_center(c, heart_glyph(S*0.18, max(1, S/110)), 0.80, 0.14)
    return finish(c), (int(S*0.42), S//2)

def compose_pen(S):
    c = peek_canvas(S, 0.52, 0.50, 0.44)
    g = pencil_glyph(S*0.84, max(2, S/40))
    px = int(S*0.38 - g.shape[1]/2); py = int(S*0.38 - g.shape[0]/2)
    paste_at(c, g, px, py)
    ys, xs = np.where(g[..., 3] > 127)
    i = int(np.argmin(xs + ys))
    hx = int(min(max(0, px + xs[i]), S-1)); hy = int(min(max(0, py + ys[i]), S-1))
    return finish(c), (hx, hy)

def compose_up(S):
    c = peek_canvas(S, 0.52, 0.52, 0.44)
    g = up_arrow_glyph(S*0.84, max(2, S/40))
    paste_at(c, g, S//2 - g.shape[1]//2, 1)
    return finish(c, 0.55), (S//2, 1)

def compose_dir(angle):
    def f(S):
        c = np.zeros((S, S, 4), np.float32)
        paste_center(c, bar_with_heads(S*0.94, angle, max(2, S/44)), 0.5, 0.5)
        return finish(c, 0.55), (S//2, S//2)
    return f

def compose_sizeall(S):
    c = np.zeros((S, S, 4), np.float32)
    paste_center(c, four_arrow_glyph(S*0.94, max(2, S/44)), 0.5, 0.5)
    return finish(c, 0.55), (S//2, S//2)

# ---------------- 写 .cur ----------------
def bmp_entry(px):
    s = px.shape[0]
    bi = struct.pack('<IiiHHIIiiII', 40, s, s*2, 1, 32, 0, s*s*4 + ((s+31)//32*4)*s, 0, 0, 0, 0)
    bgra = px[..., [2, 1, 0, 3]]
    return bi + bgra[::-1].tobytes() + b'\x00' * (((s+31)//32*4) * s)

def write_cur(path, sizes, compose):
    frames = []
    for s in sizes:
        px, hot = compose(s)
        data = bmp_entry(px)
        frames.append(((s % 256, s % 256, 0, 0, hot[0], hot[1], len(data)), data))
    offset = 6 + 16 * len(frames)
    body = b''
    for e, d in frames:
        body += struct.pack('<BBBBHHII', e[0], e[1], e[2], e[3], e[4], e[5], len(d), offset)
        offset += len(d)
    open(path, 'wb').write(struct.pack('<HHH', 0, 2, len(frames)) + body + b''.join(d for _, d in frames))
    print('written', path)

SIZE_MAIN = [256, 192, 128, 96, 64, 48, 32]
SIZE_STD  = [192, 128, 96, 64, 48, 32]

CURSORS = {
    'moe_arrow.cur':    (compose_arrow,   SIZE_MAIN),
    'moe_hand.cur':     (compose_link,    SIZE_MAIN),
    'moe_help.cur':     (compose_help,    SIZE_STD),
    'moe_busy.cur':     (compose_busy,    SIZE_MAIN),
    'moe_appstart.cur': (compose_app,     SIZE_STD),
    'moe_no.cur':       (compose_no,      SIZE_STD),
    'moe_cross.cur':    (compose_cross,   SIZE_STD),
    'moe_ibeam.cur':    (compose_ibeam,   SIZE_STD),
    'moe_pen.cur':      (compose_pen,     SIZE_STD),
    'moe_up.cur':       (compose_up,      SIZE_STD),
    'moe_sizens.cur':   (compose_dir(-90), SIZE_STD),
    'moe_sizewe.cur':   (compose_dir(0),   SIZE_STD),
    'moe_sizenwse.cur': (compose_dir(45),  SIZE_STD),
    'moe_sizenesw.cur': (compose_dir(-45), SIZE_STD),
    'moe_sizeall.cur':  (compose_sizeall, SIZE_STD),
    'moe_pin.cur':      (compose_pin,     SIZE_MAIN),
    'moe_person.cur':   (compose_person,  SIZE_MAIN),
}

only = sys.argv[1:] if len(sys.argv) > 1 else None
for name, (comp, sizes) in CURSORS.items():
    if only and name not in only:
        continue
    write_cur(name, sizes, comp)

if not only:
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    items = [('moe_arrow.cur','正常选择'), ('moe_hand.cur','链接选择'), ('moe_busy.cur','忙'),
             ('moe_appstart.cur','后台运行'), ('moe_no.cur','不可用'), ('moe_cross.cur','精确选择'),
             ('moe_ibeam.cur','文本'), ('moe_help.cur','帮助选择'), ('moe_pen.cur','手写'),
             ('moe_up.cur','备用选择'), ('moe_sizens.cur','上下调整'), ('moe_sizewe.cur','左右调整'),
             ('moe_sizenwse.cur','对角调整1'), ('moe_sizenesw.cur','对角调整2'), ('moe_sizeall.cur','移动'),
             ('moe_pin.cur','位置选择'), ('moe_person.cur','个人选择')]
    cell_w, cell_h = 220, 300
    cols = 5
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new('RGB', (cell_w*cols, cell_h*rows), (247, 247, 250))
    dd = ImageDraw.Draw(sheet)
    for i, (f, label) in enumerate(items):
        im = Image.open(f)
        fr = im.convert('RGBA').resize((170, 170), Image.LANCZOS)
        cx = (i % cols)*cell_w + 25
        cy = (i // cols)*cell_h + 25
        sheet.paste(fr, (cx, cy), fr)
        dd.text((cx + 20, cy + 185), label, fill=(60, 60, 70), font=font)
    sheet.save('效果预览.png')
    print('preview ok')
