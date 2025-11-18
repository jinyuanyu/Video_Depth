# realtime_depth_zed_dual.py
# 实时使用 ZED 相机进行流式深度估计 + 左目并排显示
# Author: ChatGPT (GPT-5)
# License: Apache 2.0

import sys
import cv2
import torch
import numpy as np
import pyzed.sl as sl
import time
import os
from video_depth_anything.video_depth_stream import VideoDepthAnything

def colorize_depth(depth):
    depth_min, depth_max = depth.min(), depth.max()
    depth_vis = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZED 双目输入 + 实时深度估计显示")
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'], help='模型类型')
    parser.add_argument('--input_size', type=int, default=384, help='输入图像尺寸')
    parser.add_argument('--metric', action='store_true', help='是否使用度量深度模型')
    parser.add_argument('--fp32', action='store_true', help='是否使用FP32精度')
    parser.add_argument('--grayscale', action='store_true', help='是否灰度显示深度图')
    parser.add_argument('--save', action='store_true', help='是否保存输出视频')
    parser.add_argument('--output', type=str, default='./outputs/zed_dual_stream.mp4', help='输出视频路径')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {DEVICE}")

    # ---------- 加载模型 ----------
    model_cfgs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
    ckpt_path = f'./checkpoints/{checkpoint_name}_{args.encoder}.pth'
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] 模型权重未找到: {ckpt_path}")
        sys.exit(1)

    print(f"[INFO] Loading model: {args.encoder}")
    model = VideoDepthAnything(**model_cfgs[args.encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # ---------- 初始化 ZED 相机 ----------
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE  # 禁用ZED内置深度
    zed = sl.Camera()

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] 无法打开 ZED 相机")
        sys.exit(1)

    runtime = sl.RuntimeParameters()
    image_left = sl.Mat()
    image_right = sl.Mat()

    print("[INFO] ZED 相机打开成功，开始实时推理...")
    print("[INFO] 按 Q 退出")

    # ---------- 视频保存 ----------
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (2560, 720))  # 两张图并排

    frame_idx = 0
    start_time = time.time()

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # 读取左右图
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            frame_left = image_left.get_data()
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGRA2BGR)

            # 深度推理
            depth = model.infer_video_depth_one(frame_left, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

            # 可视化
            if args.grayscale:
                depth_vis = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            else:
                depth_vis = colorize_depth(depth)

            # 调整大小以匹配
            h, w, _ = frame_left.shape
            depth_vis = cv2.resize(depth_vis, (w, h))

            # 拼接左右图像
            combined = np.hstack((frame_left, depth_vis))
            cv2.imshow("ZED Left + Depth Stream", combined)

            if writer:
                writer.write(combined)

            frame_idx += 1
            if frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed
                print(f"\r[INFO] 实时FPS: {fps:.2f}", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.01)

    zed.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 结束运行。")

if __name__ == "__main__":
    main()
