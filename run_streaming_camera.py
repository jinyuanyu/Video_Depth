import argparse
import numpy as np
import os
import torch
import time
import cv2

from video_depth_anything.video_depth_stream import VideoDepthAnything
from utils.dc_utils import save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything from Camera')
    # 移除了 --input_video 参数，因为我们将从摄像头读取
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    # 移除了 --max_len 参数，因为我们使用固定的录制时长
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    # 新增参数：摄像头录制时长
    parser.add_argument('--duration', type=int, default=5, help='duration to capture video from camera in seconds')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # =================================================================
    # 核心修改部分：从摄像头捕获视频
    # =================================================================
    
    # 使用索引 0 打开默认摄像头
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("错误：无法打开摄像头。请检查摄像头是否连接并被其他程序占用。")
        exit()

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 阶段 1: 从摄像头录制指定时长的视频
    captured_frames = []
    print(f"准备从摄像头录制 {args.duration} 秒视频... 按 'q' 键可提前停止。")
    start_time = time.time()
    while (time.time() - start_time) < args.duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将捕获的帧添加到列表中
        captured_frames.append(frame)
        
        # 显示实时预览窗口
        cv2.imshow('Recording... (Press "q" to stop early)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户提前停止录制。")
            break
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
    print(f"录制完成，共捕获 {len(captured_frames)} 帧。开始处理...")

    # =================================================================
    # 阶段 2: 处理已录制的视频帧
    # =================================================================
    
    if args.max_res > 0 and max(original_height, original_width) > args.max_res:
        scale = args.max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)
    
    # 计算实际捕获的帧率
    actual_capture_duration = time.time() - start_time
    # 避免除以零
    actual_fps = len(captured_frames) / actual_capture_duration if actual_capture_duration > 0 else original_fps
    
    # 如果用户指定了目标fps，则使用它，否则使用实际捕获的fps
    fps_for_saving = args.target_fps if args.target_fps > 0 else actual_fps
    
    depths = []
    process_start_time = time.time()
    
    # 遍历所有已捕获的帧进行处理
    for i, frame in enumerate(captured_frames):
        # 将 OpenCV 的 BGR 格式转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if args.max_res > 0 and max(original_height, original_width) > args.max_res:
            frame_rgb = cv2.resize(frame_rgb, (width, height)) # 调整帧大小

        # 推理深度
        depth = video_depth_anything.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
        depths.append(depth)
        
        if (i + 1) % 10 == 0:
            print(f"正在处理帧: {i + 1}/{len(captured_frames)}")
            
    process_end_time = time.time()
    print(f"处理总耗时: {process_end_time - process_start_time:.2f} 秒")

    # =================================================================
    # 阶段 3: 保存处理结果
    # =================================================================
    
    # 生成基于时间戳的输出文件名
    video_name = f'camera_capture_{int(time.time())}'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    depth_vis_path = os.path.join(args.output_dir, f'{video_name}_vis.mp4')
    depths = np.stack(depths, axis=0)
    save_video(depths, depth_vis_path, fps=fps_for_saving, is_depths=True, grayscale=args.grayscale)
    print(f"深度视频已保存至: {depth_vis_path}")