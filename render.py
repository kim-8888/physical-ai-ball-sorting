"""5色ボール・坂・正面アーム：承認構図の再構成アニメーション。
学習済み分類器・力学シミュレーションではなく、時刻から姿勢を定める映像。
実行: python render.py --preview / python render.py
"""
from pathlib import Path
import math, argparse
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
ROOT = Path(__file__).parent
W, H, FPS = (1280, 720, 30)
R = 0.42
DURATION = 4.8
COLORS = {'red': (0.94, 0.055, 0.045, 1), 'blue': (0.025, 0.24, 0.98, 1), 'yellow': (1, 0.76, 0.025, 1), 'green': (0.035, 0.68, 0.22, 1), 'black': (0.035, 0.045, 0.065, 1)}
JP = {'red': '赤', 'blue': '青', 'yellow': '黄', 'green': '緑', 'black': '黒'}
DIR = {'red': 1, 'blue': 1, 'yellow': -1, 'green': -1, 'black': -1}
ORDERS = [['red', 'blue', 'yellow', 'green', 'black'], ['green', 'red', 'black', 'blue', 'yellow'], ['blue', 'yellow', 'green', 'black', 'red']]
font_path = str(ROOT / 'assets' / 'NotoSansJP-VF.ttf')
fonts = {n: ImageFont.truetype(font_path, n) for n in [20, 24, 28, 34]}
for font in fonts.values():
    font.set_variation_by_axes([600])

def box(half, pos, color, orn=None):
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=v, basePosition=pos, baseOrientation=orn or [0, 0, 0, 1])
sphere_shapes = {}

def sphere(radius, pos, color):
    key = (radius, tuple(color))
    if key not in sphere_shapes:
        sphere_shapes[key] = p.createVisualShape(p.GEOM_MESH, fileName=str(Path(pybullet_data.getDataPath()) / 'sphere_smooth.obj'), meshScale=[radius] * 3, rgbaColor=color) if radius > 0.1 else p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_shapes[key], basePosition=pos)

def cylinder(radius, length, pos, color):
    v = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=v, basePosition=pos)

def link(a, b, r, color):
    a, b = (np.array(a), np.array(b))
    d = b - a
    length = np.linalg.norm(d)
    z = d / length
    axis = np.cross([0, 0, 1], z)
    angle = math.acos(np.clip(z[2], -1, 1))
    q = p.getQuaternionFromAxisAngle(axis.tolist(), angle) if np.linalg.norm(axis) > 1e-06 else [0, 0, 0, 1]
    v = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=length, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=v, basePosition=((a + b) / 2).tolist(), baseOrientation=q)

def smooth(u):
    u = np.clip(u, 0, 1)
    return float(u * u * (3 - 2 * u))
p.connect(p.DIRECT)
box([100, 100, 0.05], [0, 0, -0.3], [0.91, 0.93, 0.96, 1])
box([3.0, 2.9, 0.1], [0, 0.65, -0.11], [0.68, 0.74, 0.8, 1])
theta = math.atan(0.5)
q = p.getQuaternionFromEuler([theta, 0, 0])
box([0.65, 1.7, 0.07], [0, 1.65, 0.825], [0.4, 0.53, 0.65, 1], q)
for x in [-0.72, 0.72]:
    box([0.055, 1.73, 0.15], [x, 1.65, 1.02], [0.8, 0.87, 0.93, 1], q)
for y in [1.05, 2.15, 3.1]:
    box([0.51, 0.06, max(0.05, y * 0.25)], [0, y, y * 0.25 - 0.09], [0.36, 0.43, 0.5, 1])
box([1.22, 0.65, 0.035], [0, -0.18, 0.01], [0.49, 0.61, 0.7, 1])
for x, c in [(-2.1, [0.08, 0.48, 0.52, 1]), (2.1, [0.17, 0.36, 0.64, 1])]:
    box([0.63, 0.75, 0.04], [x, -0.05, -0.005], c)
    for dx in [-0.65] if x < 0 else [0.65]:
        box([0.035, 0.75, 0.12], [x + dx, -0.05, 0.1], c)
    box([0.65, 0.035, 0.12], [x, 0.71, 0.1], c)
    box([0.65, 0.035, 0.07], [x, -0.81, 0.055], c)
cylinder(0.31, 0.32, [0, -1.65, 0.18], [0.13, 0.19, 0.25, 1])
cylinder(0.23, 0.15, [0, -1.65, 0.4], [0.95, 0.49, 0.08, 1])
view = p.computeViewMatrix([3.8, -7.2, 5.3], [0, 0.82, 0.44], [0, 0, 1])
proj = p.computeProjectionMatrixFOV(38, W / H, 0.1, 30)

def arm(end):
    shoulder = np.array([0.0, -1.65, 0.58])
    target = np.array([end[0], end[1], 0.58])
    d = target[:2] - shoulder[:2]
    l1, l2 = (1.15, 1.65)
    dist = np.linalg.norm(d)
    phi = math.atan2(d[1], d[0])
    alpha = math.acos(np.clip((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1, 1))
    elbow = shoulder + np.array([l1 * math.cos(phi - alpha), l1 * math.sin(phi - alpha), 0])
    ids = [link(shoulder, elbow, 0.125, [0.98, 0.56, 0.11, 1]), link(elbow, target, 0.105, [1, 0.65, 0.16, 1])]
    for c in [shoulder, elbow, target]:
        ids.append(cylinder(0.17, 0.14, c, [0.14, 0.2, 0.26, 1]))
    ids.append(box([0.1, 0.37, 0.28], [end[0], end[1], 0.36], [0.95, 0.57, 0.13, 1]))
    ids.append(box([0.115, 0.34, 0.2], [end[0], end[1], 0.34], [0.2, 0.25, 0.29, 1]))
    return ids

def state(t, color):
    direction = DIR[color]
    home = np.array([0.8, -1.2])
    start = np.array([-direction * (R + 0.12), -0.1])
    end = np.array([direction * 1.55, -0.1])
    if t < 1.65:
        u = t / 1.65
        y = 3.02 - 3.12 * u * u
        z = R + 0.05 + max(0, (y + 0.1) / 0.3) * (0.1 + 0.078 + R * math.sqrt(1.25) - R - 0.05) if y < 0.2 else 0.5 * y + 0.078 + R * math.sqrt(1.25)
        pos = [0, y, z]
        target = home
        phase = '坂を下っています'
    elif t < 2.2:
        pos = [0, -0.1, R + 0.05]
        target = home
        phase = 'ボールの色は？'
    elif t < 2.75:
        pos = [0, -0.1, R + 0.05]
        safe = np.array([start[0], -0.95])
        target = home + (safe - home) * smooth((t - 2.2) / 0.275) if t < 2.475 else safe + (start - safe) * smooth((t - 2.475) / 0.275)
        phase = '答え：' + JP[color]
    elif t < 4.0:
        u = smooth((t - 2.75) / 1.25)
        target = start + (end - start) * u
        pos = [target[0] + direction * (R + 0.12), -0.1, R + 0.05]
        phase = ('右へ' if direction > 0 else '左へ') + '仕分け'
    else:
        pos = [direction * 1.99, -0.1, R + 0.05]
        target = end + (home - end) * smooth((t - 4) / 0.8)
        phase = '仕分け完了'
    return (pos, target, phase)

def frame(global_t):
    idx = min(14, int(global_t / DURATION))
    t = global_t - idx * DURATION
    setno = idx // 5
    color = ORDERS[setno][idx % 5]
    pos, target, phase = state(t, color)
    ids = arm(target)
    ids.append(sphere(R, pos, COLORS[color]))
    q = p.getQuaternionFromEuler([(3.02 - pos[1]) / R, pos[0] / R, 0])
    p.resetBasePositionAndOrientation(ids[-1], pos, q)
    rot = np.asarray(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    for local in [[R * 0.98, 0, 0], [0, R * 0.98, 0], [0, 0, R * 0.98]]:
        marker = np.array(pos) + rot @ np.array(local)
        ids.append(sphere(0.026, marker.tolist(), [0.85, 0.9, 0.94, 1]))
    raw = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER, shadow=0, lightDirection=[-3, -4, 8], lightAmbientCoeff=0.65, lightDiffuseCoeff=0.65)
    img = Image.fromarray(np.asarray(raw[2], dtype=np.uint8).reshape(H, W, 4)[:, :, :3])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 94], fill='#122434')
    d.text((36, 12), '5色ボールの仕分け', font=fonts[34], fill='white')
    d.text((37, 55), '坂を下るボールを、正面のアームで左右へ', font=fonts[20], fill='#c3d3df')
    d.rounded_rectangle((1010, 20, 1244, 74), radius=16, fill='#284556')
    d.text((1030, 28), f'{setno + 1} / 3 セット · {idx % 5 + 1}/5', font=fonts[24], fill='white')
    d.rounded_rectangle((388, 638, 892, 704), radius=18, fill='#ffffff', outline='#bacbd7', width=2)
    bb = d.textbbox((0, 0), phase, font=fonts[28])
    d.text(((W - (bb[2] - bb[0])) / 2, 649), phase, font=fonts[28], fill='#172d3e')
    d.text((34, 664), '左 ← 黄・緑・黒', font=fonts[24], fill='#164d50')
    d.text((1000, 664), '赤・青 → 右', font=fonts[24], fill='#224674')
    for i in ids:
        p.removeBody(i)
    return np.asarray(img)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    if args.preview:
        times = [0.35, 1.2, 2.35, 3.35, 4.35, 12.95]
        canvas = Image.new('RGB', (W * 2, H * 3))
        for i, t in enumerate(times):
            im = Image.fromarray(frame(t))
            im.save(ROOT / f'preview_{i}.png')
            canvas.paste(im, (i % 2 * W, i // 2 * H))
        canvas.save(ROOT / 'preview_sheet.jpg')
        print('preview ready')
    else:
        writer = imageio.get_writer(ROOT / 'PhysicalAI_5色ボール_坂・正面アーム左右仕分け_3セット_20261101.mp4', fps=FPS, codec='libx264', quality=8, macro_block_size=1)
        for i in range(round(15 * DURATION * FPS)):
            writer.append_data(frame(i / FPS))
            if i % 144 == 0:
                print(f'{i}/{15 * DURATION * FPS}', flush=True)
        writer.close()
p.disconnect()
