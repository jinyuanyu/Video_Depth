# realtime_d455.py
# 实时深度估计（Intel RealSense D455 + Video Depth Anything）
# Author: ChatGPT (2025)
# ---------------------------------------------------------------

import argparse
import numpy as np
import torch
import cv2
import pyrealsense2 as rs
import time
import os

from video_depth_anything.video_depth_stream import VideoDepthAnything

# --------------------------
# 初始化 RealSense 管道
# --------------------------
def init_realsense_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()

    tried = [
        (rs.stream.color, 1280, 720, rs.format.bgr8, 30),
        (rs.stream.color, 848, 480, rs.format.bgr8, 30),
        (rs.stream.color, 640, 480, rs.format.bgr8, 30),
    ]

    for (stream, w, h, fmt, fps) in tried:
        try:
            config.enable_stream(stream, w, h, fmt, fps)
            profile = pipeline.start(config)
            print(f"✅ 成功启动 RealSense 流: {w}x{h}@{fps} ({fmt})")
            return pipeline, profile
        except Exception as e:
            print(f"⚠️ 尝试 {w}x{h}@{fps} 失败: {e}")
            pipeline = rs.pipeline()
            config = rs.config()

    raise RuntimeError("❌ 无法启动任何 RealSense 视频流，请检查连接")

# --------------------------
# 主函数
# --------------------------
def main():
    parser = argparse.ArgumentParser(description='Real-time D455 Depth Estimation with Video Depth Anything')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--metric', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading model: {args.encoder}")

    # --------------------------
    # 加载模型
    # --------------------------
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    model = VideoDepthAnything(**model_configs[args.encoder])
    ckpt_path = f'./checkpoints/{checkpoint_name}_{args.encoder}.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 未找到模型权重：{ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # --------------------------
    # 初始化 D455 摄像头
    # --------------------------
    print("[INFO] 启动 RealSense 管道...")
    pipeline, profile = init_realsense_pipeline()
    align = rs.align(rs.stream.color)  # 对齐深度到彩色（若有深度流可拓展）

    # --------------------------
    # 实时推理循环
    # --------------------------
    print("[INFO] 开始实时推理，按 'q' 退出。")
    frame_id = 0
    t0 = time.time()
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 深度推理
        depth = model.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

        # 可视化
        if not args.grayscale:
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
        else:
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        cv2.imshow('RGB', frame)
        cv2.imshow('Depth (Video-Depth-Anything)', depth_vis)

        frame_id += 1
        if frame_id % 10 == 0:
            fps = frame_id / (time.time() - t0)
            print(f"\r[INFO] 当前 FPS: {fps:.2f}", end='')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()
    print("\n✅ 实时深度推理结束。")

# --------------------------
# 程序入口
# --------------------------
if __name__ == "__main__":
    main()
