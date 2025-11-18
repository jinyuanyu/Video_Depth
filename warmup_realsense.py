# warmup_realsense.py
import pyrealsense2 as rs
import time

def warmup_camera():
    print("=== RealSense 相机预热 ===")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # 启动管道
        pipeline.start(config)
        print("✅ 相机启动成功")
        
        # 预热：读取一些帧
        print("预热中...", end="")
        for i in range(30):  # 读取30帧进行预热
            try:
                frames = pipeline.wait_for_frames(5000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    print(".", end="", flush=True)
                time.sleep(0.1)
            except:
                print("x", end="", flush=True)
        
        print("\n✅ 预热完成")
        pipeline.stop()
        return True
        
    except Exception as e:
        print(f"❌ 预热失败: {e}")
        return False

if __name__ == "__main__":
    warmup_camera()