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

        # 2. THIẾT LẬP MÔI TRƯỜNG (Fix lỗi thiếu NVCC/CUDA)
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
        # Khi script chạy từ systemd/rc.local, không có session đồ họa.
        # DISPLAY=:0 trỏ thẳng vào màn hình vật lý đầu tiên của Jetson Nano.
        full_env["DISPLAY"] = full_env.get("DISPLAY", ":0")
        # XAUTHORITY cho phép process truy cập X server với đúng quyền.
        xauth_path = os.path.join(os.path.expanduser("~pi"), ".Xauthority")
        if os.path.exists(xauth_path):
            full_env["XAUTHORITY"] = xauth_path
        else:
            # Thử với home directory hiện tại (phòng trường hợp user khác)
            xauth_fallback = os.path.join(os.path.expanduser("~"), ".Xauthority")
            if os.path.exists(xauth_fallback):
                full_env["XAUTHORITY"] = xauth_fallback

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

        # 3. CHẠY TIẾN TRÌNH CON
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