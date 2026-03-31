import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import logging
import threading

class TRTEngine:
    def __init__(self, engine_path, max_batch_size=1):
        self.logger = logging.getLogger(f"{__name__}.TRTEngine")
        self.logger.info(f"Loading TensorRT engine from {engine_path}")
        
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        # Store CUDA context for thread-safe push/pop
        self.cuda_ctx = cuda.Context.get_current()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self._lock = threading.Lock()
        
    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        output_shapes = []
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to bindings
            bindings.append(int(device_mem))
            
            # Append to inputs or outputs
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                output_shapes.append(tuple(shape))
        
        self.output_shapes = output_shapes
        self.logger.info(f"Output bindings: {output_shapes}")
        return inputs, outputs, bindings, stream

    def predict(self, input_data):
        # Thread-safe inference with CUDA context push/pop
        with self._lock:
            self.cuda_ctx.push()
            try:
                # 1. Copy host to device
                self.inputs[0]['host'] = np.ascontiguousarray(input_data)
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
                
                # 2. Execute
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                
                # 3. Copy device to host
                for out in self.outputs:
                    cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
                    
                # 4. Synchronize stream
                self.stream.synchronize()
                
                return [out['host'].copy().reshape(shape) for out, shape in zip(self.outputs, self.output_shapes)]
            finally:
                self.cuda_ctx.pop()

    def __del__(self):
        # Explicitly release resources if needed, though pycuda often handles it
        pass
