import argparse
import os
import re
import time
import cv2
import numpy as np
import torch

from video_depth_anything.video_depth_stream import VideoDepthAnything

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process MannequinChallenge to per-frame depth and optional depth video')
    parser.add_argument('--dataset_root', type=str, default='./MannequinChallenge', help='Path to MannequinChallenge root folder')
    parser.add_argument('--splits', type=str, default='train,validation,test', help='Comma-separated splits to process')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280, help='Max longer side for processing; <=0 means no limit')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--target_fps', type=float, default=-1, help='Override fps for depth video; <=0 to use timestamps from txt')
    parser.add_argument('--fp32', action='store_true', help='Use torch.float32 (default: float16)')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale visualization (default: colorful)')
    parser.add_argument('--save_video', action='store_true', help='Also save a depth visualization video per clip')
    parser.add_argument('--skip_existing', action='store_true', help='Skip frames whose depth png already exists')
    parser.add_argument('--exts', type=str, default='.jpg,.jpeg,.png', help='Allowed frame extensions')
    return parser.parse_args()

def load_model(encoder: str, device: str, fp32: bool):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs[encoder])
    ckpt_path = f'./checkpoints/video_depth_anything_{encoder}.pth'
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    # if not fp32:
    #     model = model.half()
    # return model

_frame_pat = re.compile(r'^frame_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)

def list_frames_sorted(frames_dir: str, allowed_exts):
    files = []
    for f in os.listdir(frames_dir):
        m = _frame_pat.match(f)
        if m:
            num_str = m.group(1)
            ext = m.group(2).lower()
            if f'.{ext}' in allowed_exts:
                files.append((int(num_str), num_str, f))
    files.sort(key=lambda x: x[0])
    return files  # list of tuples: (num, num_str, filename)

def get_fps_from_txt(txt_path: str, default_fps: float = 24.0) -> float:
    if not os.path.isfile(txt_path):
        return default_fps
    timestamps = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            return default_fps
        # first line is URL, skip
        for ln in lines[1:]:
            cols = ln.split()
            if len(cols) >= 1:
                ts = int(cols[0])
                timestamps.append(ts)
        if len(timestamps) < 2:
            return default_fps
        diffs = np.diff(np.array(timestamps, dtype=np.int64))
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return default_fps
        med = np.median(diffs)  # microseconds
        fps = 1e6 / med if med > 0 else default_fps
        # Clamp to a reasonable range
        if not np.isfinite(fps) or fps <= 0:
            return default_fps
        return float(fps)
    except Exception:
        return default_fps

def compute_resize(hw, max_res):
    h, w = hw
    if max_res is not None and max_res > 0:
        longer = max(h, w)
        if longer > max_res:
            scale = max_res / longer
            nh, nw = int(round(h * scale)), int(round(w * scale))
            return nh, nw
    return h, w

def normalize_depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    # Normalize per-frame to 16-bit for saving
    dmin = np.min(depth)
    dmax = np.max(depth)
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        norm = np.zeros_like(depth, dtype=np.float32)
    else:
        norm = (depth - dmin) / (dmax - dmin)
    out = (norm * 65535.0).clip(0, 65535).astype(np.uint16)
    return out

def depth_to_vis_bgr8(depth: np.ndarray, grayscale: bool) -> np.ndarray:
    # Convert depth to 0-255 uint8 visualization for video
    d16 = normalize_depth_to_uint16(depth)
    d8 = (d16 / 257).astype(np.uint8)  # 65535/255 ≈ 257
    if grayscale:
        vis = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
    else:
        vis = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    return vis

def infer_depth(model, frame_rgb: np.ndarray, input_size: int, device: str, fp32: bool) -> np.ndarray:
    # The provided API infer_video_depth_one returns numpy depth map
    depth = model.infer_video_depth_one(frame_rgb, input_size=input_size, device=device, fp32=fp32)
    # Ensure HxW float32
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float().numpy()
    depth = np.squeeze(depth)
    depth = depth.astype(np.float32)
    return depth

def process_clip(model, split_dir: str, txt_file: str, args):
    base = os.path.splitext(os.path.basename(txt_file))[0]
    clip_dir = os.path.join(split_dir, base)
    if not os.path.isdir(clip_dir):
        print(f'[WARN] Missing folder for {txt_file}, expected: {clip_dir}')
        return

    allowed_exts = set([e.strip().lower() for e in args.exts.split(',') if e.strip()])
    frames = list_frames_sorted(clip_dir, allowed_exts)
    if not frames:
        print(f'[WARN] No frames found in {clip_dir}')
        return

    depth_dir = os.path.join(clip_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    # Determine fps for video
    txt_path = os.path.join(split_dir, base + '.txt')
    fps = args.target_fps if args.target_fps and args.target_fps > 0 else get_fps_from_txt(txt_path, default_fps=24.0)

    # Will initialize video writer on first frame if needed
    writer = None
    vis_size = None  # (w, h)
    video_path = os.path.join(clip_dir, 'depth_video.mp4') if args.save_video else None

    # Determine resize based on first frame
    first_frame_path = os.path.join(clip_dir, frames[0][2])
    img0 = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    if img0 is None:
        print(f'[WARN] Failed to read first frame: {first_frame_path}')
        return
    h0, w0 = img0.shape[:2]
    nh, nw = compute_resize((h0, w0), args.max_res)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total = len(frames)
    t0 = time.time()
    with torch.no_grad():
        for idx, (num, num_str, fname) in enumerate(frames, 1):
            in_path = os.path.join(clip_dir, fname)
            out_name = f'depth_{num_str}.png'
            out_path = os.path.join(depth_dir, out_name)
            if args.skip_existing and os.path.isfile(out_path):
                # If also saving video, read existing png to append (for consistency)
                if args.save_video:
                    vis = None
                    # Read saved depth png to BGR for video (assumes grayscale 16-bit or color 8-bit)
                    dep_img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
                    if dep_img is None:
                        pass
                    else:
                        if dep_img.dtype == np.uint16:
                            d8 = (dep_img / 257).astype(np.uint8)
                            vis = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR) if args.grayscale else cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
                        else:
                            # already 8-bit visualization
                            if dep_img.ndim == 2:
                                vis = cv2.cvtColor(dep_img, cv2.COLOR_GRAY2BGR)
                            else:
                                vis = dep_img
                    if vis is not None:
                        if writer is None and args.save_video:
                            vis_size = (vis.shape[1], vis.shape[0])
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(video_path, fourcc, fps, vis_size)
                        if args.save_video and writer is not None:
                            writer.write(vis)
                if idx % 20 == 0 or idx == total:
                    print(f'[{base}] {idx}/{total} (skipped existing)')
                continue

            img_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f'[WARN] Failed to read: {in_path}')
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if (img_rgb.shape[0], img_rgb.shape[1]) != (nh, nw):
                img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

            depth = infer_depth(model, img_rgb, args.input_size, device, args.fp32)

            # Save per-frame depth image
            # 默认保存16位灰度深度图（更利于保真）；若需要彩色可视化，则保存8位彩色图
            if args.grayscale:
                d16 = normalize_depth_to_uint16(depth)
                cv2.imwrite(out_path, d16)
                vis = cv2.cvtColor((d16 / 257).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                vis = depth_to_vis_bgr8(depth, grayscale=False)
                cv2.imwrite(out_path, vis)  # 保存彩色可视化（8位）
            # Save to video
            if args.save_video:
                if writer is None:
                    vis_size = (vis.shape[1], vis.shape[0])
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(video_path, fourcc, fps, vis_size)
                writer.write(vis)

            if idx % 20 == 0 or idx == total:
                print(f'[{base}] processed {idx}/{total}')

    if writer is not None:
        writer.release()
    dt = time.time() - t0
    print(f'[{base}] done in {dt:.2f}s, frames: {total}, fps(video): {fps:.3f}, saved depth dir: {depth_dir}' + (f', video: {video_path}' if args.save_video else ''))

def main():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    print(f'Using device: {DEVICE}')
    print('Loading model...')
    model = load_model(args.encoder, DEVICE, args.fp32)
    print('Model loaded.')

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    for split in splits:
        split_dir = os.path.join(args.dataset_root, split)
        if not os.path.isdir(split_dir):
            print(f'[WARN] Split dir not found: {split_dir}')
            continue

        # Find all .txt files
        txt_files = [f for f in os.listdir(split_dir) if f.lower().endswith('.txt')]
        txt_files.sort()
        print(f'Processing split "{split}" with {len(txt_files)} clips...')
        for i, txt in enumerate(txt_files, 1):
            txt_path = os.path.join(split_dir, txt)
            try:
                process_clip(model, split_dir, txt, args)
            except Exception as e:
                print(f'[ERROR] Failed clip {txt_path}: {e}')
        print(f'Split "{split}" done.')

if __name__ == '__main__':
    main()
