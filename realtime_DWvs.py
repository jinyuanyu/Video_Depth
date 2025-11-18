import os
import time
import socket
import struct
import argparse

import pyrealsense2 as rs
import numpy as np
import cv2
import torch

from video_depth_anything.video_depth_stream import VideoDepthAnything

# ----------------------------- 可选 TurboJPEG 加速 JPEG 编码 -----------------------------
try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_420
    _jpeg = TurboJPEG()
    def jpeg_encode(img, quality):
        return _jpeg.encode(img, quality=quality, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_420)
except Exception:
    _jpeg = None
    def jpeg_encode(img, quality):
        ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ret:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

# ----------------------------- UDP 分片协议定义 -----------------------------
MAGIC = b'IM'
VERSION = 1
# Header: magic(2s), version(B), stream(B), frame_id(I), timestamp_us(Q), total_chunks(H), chunk_id(H), payload_len(H)
HEADER_STRUCT = struct.Struct('!2s B B I Q H H H')
HEADER_SIZE = HEADER_STRUCT.size

def chunk_and_send(sock, addr, stream_id, frame_id, timestamp_us, data, mtu):
    """
    将数据按照给定 MTU 分片并通过 UDP 发送。
    stream_id: 0=RGB, 1=Depth
    """
    max_payload = mtu - HEADER_SIZE
    total_chunks = (len(data) + max_payload - 1) // max_payload
    view = memoryview(data)
    for chunk_id in range(total_chunks):
        start = chunk_id * max_payload
        end = min(start + max_payload, len(data))
        payload = view[start:end]
        header = HEADER_STRUCT.pack(
            MAGIC, VERSION, stream_id, frame_id, timestamp_us,
            total_chunks, chunk_id, len(payload)
        )
        sock.sendto(header + payload, addr)

# ----------------------------- 可视化辅助 -----------------------------
def colorize_depth(depth):
    depth_min, depth_max = depth.min(), depth.max()
    depth_vis = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

def main():
    parser = argparse.ArgumentParser(description="Intel RealSense D455 实时流式深度推理 + UDP 发送 RGB/Depth")
    # 模型相关
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='模型类型')
    parser.add_argument('--input_size', type=int, default=384, help='输入图像尺寸')
    parser.add_argument('--metric', action='store_true', help='是否使用度量深度模型')
    parser.add_argument('--fp32', action='store_true', help='是否使用FP32精度')
    parser.add_argument('--grayscale', action='store_true', help='灰度深度图显示')
    # 本地保存/显示
    parser.add_argument('--save', action='store_true', help='保存输出视频')
    parser.add_argument('--output', type=str, default='./outputs/realsense_depth_dual.mp4', help='输出视频路径')
    # 发送相关
    parser.add_argument('--server-ip', type=str, default='127.0.0.1', help='服务器IP(不填则不发送)')
    parser.add_argument('--port', type=int, default=5005, help='UDP端口')
    parser.add_argument('--mtu', type=int, default=1400, help='单个UDP包的最大负载(建议<=1400)')
    parser.add_argument('--jpeg_quality', type=int, default=85, help='JPEG质量(0-100)')
    parser.add_argument('--send_fps', type=float, default=30.0, help='发送帧率限制(>0启用节流; 0=不限速)')
    parser.add_argument('--pacing', type=str, default='sleep', choices=['sleep', 'drop'], help='节流策略: sleep=限速等待; drop=超速则跳过发送本帧')
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
        return

    print(f"[INFO] Loading model: {args.encoder}")
    model = VideoDepthAnything(**model_cfgs[args.encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    # ---------- 初始化 RealSense D455 ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("[INFO] 启动 RealSense 管道...")
    pipeline.start(config)

    # ---------- 视频保存 ----------
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (2560, 720))

    # ---------- UDP 初始化 ----------
    send_enabled = bool(args.server_ip)
    sock = None
    addr = None
    if send_enabled:
        addr = (args.server_ip, args.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 增大发送缓冲区并设置加速转发DSCP
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except Exception:
            pass
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0xB8)  # EF (Expedited Forwarding)
        except Exception:
            pass
        print(f"[INFO] 将发送到 {addr}, mtu={args.mtu}, jpeg_quality={args.jpeg_quality}, send_fps={args.send_fps}, pacing={args.pacing}")

    print("[INFO] RealSense 已启动，开始推理与发送，按 Q 退出")

    frame_count = 0
    frame_id = 0
    start_time = time.time()

    # 发送节流控制
    send_interval = 1.0 / args.send_fps if (args.send_fps and args.send_fps > 0) else 0.0
    last_send_time = 0.0

    try:
        while True:
            # 获取彩色帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # 深度推理
            depth = model.infer_video_depth_one(frame, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

            # 可视化（灰度或伪彩）
            if args.grayscale:
                depth_vis = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            else:
                depth_vis = colorize_depth(depth)

            # 调整大小保持一致
            h, w, _ = frame.shape
            if depth_vis.shape[:2] != (h, w):
                depth_vis = cv2.resize(depth_vis, (w, h))

            # UDP 发送：原图与深度图
            do_send = send_enabled
            if send_enabled and send_interval > 0:
                now = time.time()
                if args.pacing == 'sleep':
                    # 等待到下一个发送时隙
                    wait = (last_send_time + send_interval) - now
                    if wait > 0:
                        time.sleep(wait)
                    do_send = True
                else:
                    # drop 模式：超过速率则跳过本帧的发送，但仍显示
                    if (now - last_send_time) < send_interval:
                        do_send = False
                    else:
                        do_send = True

            if do_send and send_enabled:
                try:
                    ts_us = time.time_ns() // 1_000
                    rgb_jpg = jpeg_encode(frame, args.jpeg_quality)
                    depth_jpg = jpeg_encode(depth_vis, args.jpeg_quality)
                    # stream 0: RGB
                    chunk_and_send(sock, addr, 0, frame_id, ts_us, rgb_jpg, args.mtu)
                    # stream 1: Depth
                    chunk_and_send(sock, addr, 1, frame_id, ts_us, depth_jpg, args.mtu)
                    last_send_time = time.time()
                except Exception as e:
                    # 出错时不中断推理与显示，避免堆积
                    print(f"[WARN] 发送失败: {e}")

            # 本地并排显示（每发送一帧就显示一帧；若本帧被丢弃发送，也仍显示）
            combined = np.hstack((frame, depth_vis))
            cv2.imshow("D455 Color + Depth Stream", combined)

            if writer:
                writer.write(combined)

            frame_count += 1
            frame_id += 1

            # 实时FPS统计
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"\r[INFO] 实时FPS(推理/显示): {fps:.2f}", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        if writer:
            writer.release()
        if sock:
            sock.close()
        cv2.destroyAllWindows()
        print("\n[INFO] 已结束运行。")

if __name__ == "__main__":
    main()
