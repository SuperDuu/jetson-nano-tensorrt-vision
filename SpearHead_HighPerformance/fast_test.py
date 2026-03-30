import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# --- CẤU HÌNH ---
# Địa chỉ IP điện thoại của bạn
CAM_URL = "http://192.168.0.102:4747/video"
# Đường dẫn file engine bạn vừa build
ENGINE_PATH = "models/yolo.engine"

class Predictor:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.allocations, self.stream = self.setup_buffers()

    def setup_buffers(self):
        inputs, outputs, allocations = [], [], []
        for i in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(i)
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            allocations.append(int(device_mem))
            if is_input: inputs.append({'host': host_mem, 'device': device_mem})
            else: outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, allocations, cuda.Stream()

    def infer(self, img):
        # Tiền xử lý ảnh (Resize về 512x512 theo file engine của bạn)
        img_input = cv2.resize(img, (512, 512)).transpose(2, 0, 1).astype(np.float32) / 255.0
        np.copyto(self.inputs[0]['host'], img_input.ravel())
        
        # Đẩy dữ liệu lên GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.allocations, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

# --- CHƯƠNG TRÌNH CHÍNH ---
print("Đang nạp bộ não TensorRT...")
model = Predictor(ENGINE_PATH)
cap = cv2.VideoCapture(CAM_URL)

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if not ret: break

    # Chạy AI
    result = model.infer(frame)
    
    # Tính FPS thực tế
    fps = 1.0 / (time.time() - t1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("SpearHead Humanoid Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
