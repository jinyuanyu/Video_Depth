import argparse
import os
import time
import cv2
import numpy as np
import torch

from video_depth_anything.video_depth_stream import VideoDepthAnything


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a single video and generate its depth visualization video'
    )
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output depth video (mp4). Default: <input>_depth.mp4')
    parser.add_argument('--input_size', type=int, default=1080, help='Input size fed into the model')
    parser.add_argument('--max_res', type=int, default=1920, help='Resize longer side to at most this value; <=0 means no limit')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'], help='Model backbone')
    parser.add_argument('--target_fps', type=float, default=0, help='Override FPS for output video; <=0 to use input video FPS')
    parser.add_argument('--fp32', action='store_true', help='Use torch.float32 (default: use float16 if supported)')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale visualization (default: colorful)')
    parser.add_argument('--save_depth_frames', action='store_true', help='Also save per-frame depth images')
    parser.add_argument('--depth_frames_dir', type=str, default=None, help='Directory to save per-frame depth images (used when --save_depth_frames)')
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
    
    return model


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
    dmin = np.min(depth)
    dmax = np.max(depth)
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        norm = np.zeros_like(depth, dtype=np.float32)
    else:
        norm = (depth - dmin) / (dmax - dmin)
    out = (norm * 65535.0).clip(0, 65535).astype(np.uint16)
    return out


def depth_to_vis_bgr8(depth: np.ndarray, grayscale: bool) -> np.ndarray:
    d16 = normalize_depth_to_uint16(depth)
    d8 = (d16 / 257).astype(np.uint8)  # 65535/255 ≈ 257
    if grayscale:
        vis = cv2.cvtColor(d8, cv2.COLOR_GRAY2BGR)
    else:
        vis = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    return vis


def infer_depth(model, frame_rgb: np.ndarray, input_size: int, device: str, fp32: bool) -> np.ndarray:
    depth = model.infer_video_depth_one(frame_rgb, input_size=input_size, device=device, fp32=fp32)
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float().numpy()
    depth = np.squeeze(depth).astype(np.float32)
    return depth


def main():
    args = parse_args()

    # Device and precision setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cpu' and not args.fp32:
        print('[INFO] CPU detected; forcing fp32 for better compatibility.')
        args.fp32 = True
    torch.backends.cudnn.benchmark = True

    # Prepare paths
    in_path = args.input_video
    if not os.path.isfile(in_path):
        print(f'[ERROR] Input video not found: {in_path}')
        return

    in_dir = os.path.dirname(in_path)
    in_stem, _ = os.path.splitext(os.path.basename(in_path))
    out_video_path = args.output if args.output else os.path.join(in_dir, f'{in_stem}_depth.mp4')

    # Depth frames directory if needed
    if args.save_depth_frames:
        depth_frames_dir = args.depth_frames_dir if args.depth_frames_dir else os.path.join(in_dir, f'{in_stem}_depth_frames')
        os.makedirs(depth_frames_dir, exist_ok=True)
    else:
        depth_frames_dir = None

    # Load model
    print(f'Using device: {DEVICE}')
    print('Loading model...')
    model = load_model(args.encoder, DEVICE, args.fp32)
    print('Model loaded.')

    # Open input video
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f'[ERROR] Failed to open video: {in_path}')
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(orig_fps) or orig_fps <= 0:
        orig_fps = 24.0
    out_fps = args.target_fps if args.target_fps and args.target_fps > 0 else float(orig_fps)

    # Determine resize and writer from first frame
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        print('[ERROR] Failed to read first frame from video.')
        cap.release()
        return

    h0, w0 = frame_bgr.shape[:2]
    nh, nw = compute_resize((h0, w0), args.max_res)
    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (nw, nh))

    # Rewind to first frame for processing all frames (we already read it, so process it first)
    frame_idx = 0
    t0 = time.time()

    def process_and_write(bgr_img, idx):
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if (img_rgb.shape[0], img_rgb.shape[1]) != (nh, nw):
            img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        with torch.no_grad():
            depth = infer_depth(model, img_rgb, args.input_size, DEVICE, args.fp32)
        vis = depth_to_vis_bgr8(depth, grayscale=args.grayscale)
        writer.write(vis)

        if depth_frames_dir is not None:
            # Save per-frame depth
            if args.grayscale:
                d16 = normalize_depth_to_uint16(depth)
                out_path = os.path.join(depth_frames_dir, f'depth_{idx:06d}.png')
                cv2.imwrite(out_path, d16)  # 16-bit grayscale
            else:
                out_path = os.path.join(depth_frames_dir, f'depth_{idx:06d}.png')
                cv2.imwrite(out_path, vis)  # 8-bit colorful visualization

    # Process the first frame that was already read
    process_and_write(frame_bgr, frame_idx)
    frame_idx += 1

    # Process remaining frames
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        process_and_write(frame_bgr, frame_idx)
        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f'Processed {frame_idx} frames...')

    cap.release()
    writer.release()
    dt = time.time() - t0
    print(f'Done. Frames: {frame_idx}, time: {dt:.2f}s, output fps: {out_fps:.3f}')
    print(f'Depth video saved to: {out_video_path}')
    if depth_frames_dir is not None:
        print(f'Per-frame depth images saved to: {depth_frames_dir}')


if __name__ == '__main__':
    main()
