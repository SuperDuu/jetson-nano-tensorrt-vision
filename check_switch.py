import Jetson.GPIO as GPIO
import subprocess
import time
import os
import sys

# ==================================================
# CONFIGURATION - KIỂM TRA KỸ ĐƯỜNG DẪN
# ==================================================
INPUT_PIN = 16  # Đã đổi sang chân 16 an toàn (Board numbering)
PROJECT_ROOT = "/home/pi/Desktop/jetson-nano-tensorrt-vision"

# TRỎ THẲNG VÀO PYTHON TRONG VENV ĐỂ NHẬN THƯ VIỆN (pyserial, cv2...)
PYTHON_EXEC = os.path.join(PROJECT_ROOT, "venv/bin/python3")
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "src/system_manager_v2.py")
CAMERA_DEVICE = "/dev/video0"  # Camera device cần giải phóng trước khi khởi động


def cleanup_previous_instances():
    """
    Dọn sạch instance cũ trước khi launch mới.
    Giải quyết lỗi 'Device /dev/video0 is busy' khi restart.
    Mirror logic từ cleanup_jetson.sh (đã xác nhận hoạt động).
    """
    print("[CLEANUP] Killing old vision processes...")

    # Kill tất cả process vision cũ
    for pattern in ["system_manager_v2.py", "system_manager.py"]:
        try:
            subprocess.run(["pkill", "-f", pattern],
                           check=False, timeout=3, capture_output=True)
        except Exception:
            pass

    # Kill zombie GStreamer processes (nguyên nhân chính gây camera busy)
    for gst_proc in ["gst-launch-1.0", "nvgstplayer"]:
        try:
            subprocess.run(["pkill", "-9", gst_proc],
                           check=False, timeout=2, capture_output=True)
        except Exception:
            pass

    # Giải phóng camera nếu vẫn bị giữ bởi process khác
    try:
        result = subprocess.run(
            ["fuser", CAMERA_DEVICE],
            check=False, timeout=2, capture_output=True, text=True
        )
        if result.stdout.strip():
            subprocess.run(["fuser", "-k", CAMERA_DEVICE],
                           check=False, timeout=3, capture_output=True)
            print(f"[CLEANUP] Released {CAMERA_DEVICE} from old processes")
    except Exception:
        pass

    # Restart nvargus-daemon: bước QUAN TRỌNG nhất trên Jetson Nano
    # Daemon này quản lý camera. Nếu bị treo → camera mãi busy
    print("[CLEANUP] Restarting nvargus-daemon...")
    try:
        subprocess.run(["systemctl", "restart", "nvargus-daemon"],
                       check=False, timeout=10, capture_output=True)
    except Exception:
        pass

    # Chờ daemon khởi động lại hoàn toàn
    time.sleep(2.0)
    print("[CLEANUP] Done. Camera should be free now.")

def main():
    # 1. Cấu hình GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    # Dùng PUD_UP vì chân 16 không có trở kéo cứng
    GPIO.setup(INPUT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print(f"\n[SYSTEM] --- Boot check on Pin {INPUT_PIN} ---")
    
    # Chờ 2 giây để điện áp ổn định và tránh dội phím
    time.sleep(2)

    # Đọc trạng thái: LOW (0) là CÔNG TẮC ĐANG BẬT (Nối chân 16 với GND)
    input_state = GPIO.input(INPUT_PIN)
    
    if input_state == GPIO.LOW:
        print(">>> Switch status: ON (LOW)")
        print(">>> Action: Launching Vision System V2...")

        # 2. Dọn process cũ trước khi launch (fix 'camera busy' mãi mãi)
        cleanup_previous_instances()

        # 3. THIẾT LẬP MÔI TRƯỜNG (Fix lỗi thiếu NVCC/CUDA)
        # Sao chép môi trường hiện tại
        full_env = os.environ.copy()
        
        # Ép thêm đường dẫn CUDA vào PATH để PyCUDA tìm thấy nvcc
        cuda_bin = "/usr/local/cuda/bin"
        cuda_lib = "/usr/local/cuda/lib64"
        
        full_env["PATH"] = f"{cuda_bin}:{full_env.get('PATH', '')}"
        full_env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{full_env.get('LD_LIBRARY_PATH', '')}"
        
        # Đảm bảo PYTHONPATH trỏ đúng vào thư mục gốc dự án
        full_env["PYTHONPATH"] = f"{PROJECT_ROOT}:{full_env.get('PYTHONPATH', '')}"

        # --- BẮT BUỘC ĐỂ HIỂN THỊ RA MÀN HÌNH KHI CHẠY LÚC STARTUP ---
        full_env["DISPLAY"] = full_env.get("DISPLAY", ":0")

        # Tìm XAUTHORITY đúng: GDM lưu auth tại /run/user/<uid>/gdm/Xauthority
        # Không phải ~/.Xauthority (cái đó là kết nối SSH)
        xauth_candidates = [
            "/run/user/1000/gdm/Xauthority",   # GDM3 (Jetson Ubuntu 18.04)
            "/run/user/1001/gdm/Xauthority",   # GDM3 (nếu uid khác)
            "/home/pi/.Xauthority",            # Fallback cổ điển
            os.path.join(os.path.expanduser("~"), ".Xauthority"),  # User hiện tại
        ]
        for xauth_path in xauth_candidates:
            if os.path.exists(xauth_path):
                full_env["XAUTHORITY"] = xauth_path
                print(f"[INFO] XAUTHORITY: {xauth_path}")
                break

        # --- CẤP QUYỀN X11 CHO ROOT PROCESS ---
        # Root (sudo) mặc định bị X server từ chối kết nối.
        # xhost +local: cho phép tất cả local process (kể cả root) hiển thị lên màn hình.
        try:
            xhost_env = {"DISPLAY": full_env["DISPLAY"]}
            if "XAUTHORITY" in full_env:
                xhost_env["XAUTHORITY"] = full_env["XAUTHORITY"]
            subprocess.run(
                ["xhost", "+local:"],
                env=xhost_env,
                check=False, timeout=3,
                capture_output=True
            )
            print("[INFO] xhost +local: granted (X11 display access enabled)")
        except Exception as e:
            print(f"[WARN] xhost failed (display may not show): {e}")

        # 4. CHẠY TIẾN TRÌNH CON
        try:
            # Chạy trực tiếp, kết quả in ra màn hình để debug
            subprocess.run(
                [PYTHON_EXEC, SCRIPT_PATH], 
                cwd=PROJECT_ROOT, 
                env=full_env, 
                check=True
            )
        except KeyboardInterrupt:
            print("\n[INFO] System stopped by user.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Vision system exited with error code {e.returncode}")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            
    else:
        print(">>> Switch status: OFF (HIGH)")
        print(">>> Action: Bypassing Vision System. Ready for manual control.")

    # Dọn dẹp GPIO trước khi thoát
    GPIO.cleanup()

if __name__ == "__main__":
    main()