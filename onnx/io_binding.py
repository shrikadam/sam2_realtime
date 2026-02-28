import os
import torch
import numpy as np
import cv2
import time
import onnxruntime
import gc
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "./configs/sam2.1/sam2.1_hiera_t.yaml" 
FP16_ONNX_PATH = "sam2.1_tiny_encoder_fp16.onnx"
DEVICE = 'cuda:0'

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def benchmark_io_binding(image_path):
    print(f"--- Benchmarking FP16 CUDA: Standard vs I/O Binding ---")

    # 1. Setup PyTorch Predictor (for Decoder)
    sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # 2. Image Preprocessing (Direct to PyTorch Tensor)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image, (1024, 1024))
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    
    x_np = (img_resized - mean) / std
    x_np = x_np.transpose(2, 0, 1).astype(np.float32)[None, :, :, :]
    
    # Push input directly to GPU
    x_tensor = torch.from_numpy(x_np).contiguous().to(DEVICE)

    # 3. Initialize ONNX Runtime Session
    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = 3
    # Leaving optimizations default (enabled) as per your stable FP16 run
    
    ort_sess = onnxruntime.InferenceSession(
        FP16_ONNX_PATH, 
        sess_options=sess_options,
        providers=['CUDAExecutionProvider']
    )

    # =========================================================
    # METHOD 1: STANDARD INFERENCE (NumPy / CPU Hand-off)
    # =========================================================
    print("\n[Standard Inference] Running...")
    input_dict = {"image": x_np}
    
    # Warmup
    for _ in range(5):
        ort_sess.run(None, input_dict)
        
    start_time = time.perf_counter()
    for _ in range(10):
        numpy_features = ort_sess.run(None, input_dict)
        # The Jetson bottleneck: Forcing contiguous memory via CPU
        features = [torch.from_numpy(np.ascontiguousarray(f)).to(DEVICE, dtype=torch.float32) for f in numpy_features]
    end_time = time.perf_counter()
    
    std_time = ((end_time - start_time) / 10) * 1000
    print(f" -> Standard Avg Time: {std_time:.2f} ms ({1000/std_time:.1f} FPS)")
    
    del numpy_features, features
    clear_memory()

    # =========================================================
    # METHOD 2: I/O BINDING (Zero-Copy GPU -> GPU)
    # =========================================================
    print("\n[I/O Binding Inference] Running...")
    
    # Pre-allocate output tensors on the GPU. 
    # We force .contiguous() here so ORT writes the bytes in the perfect order for PyTorch.
    out_tensors = {
        'image_embed': torch.empty((1, 256, 64, 64), dtype=torch.float16, device=DEVICE).contiguous(),
        'high_res_0': torch.empty((1, 32, 256, 256), dtype=torch.float16, device=DEVICE).contiguous(),
        'high_res_1': torch.empty((1, 64, 128, 128), dtype=torch.float16, device=DEVICE).contiguous()
    }

    # Setup the bindings
    io_binding = ort_sess.io_binding()
    
    # Bind the input tensor's physical memory address
    io_binding.bind_input(
        name='image',
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        shape=tuple(x_tensor.shape),
        buffer_ptr=x_tensor.data_ptr()
    )

    # Bind the output tensors' physical memory addresses
    # for name, tensor in out_tensors.items():
    #     io_binding.bind_output(
    #         name=name,
    #         device_type='cuda',
    #         device_id=0,
    #         element_type=np.float16,
    #         shape=tuple(tensor.shape),
    #         buffer_ptr=tensor.data_ptr()
    #     )

    # 2. Dynamically allocate and bind outputs based on the ONNX graph
    out_tensors = {}
    
    # Map ONNX types to PyTorch and NumPy types
    type_map = {
        'tensor(float)': (torch.float32, np.float32),
        'tensor(float16)': (torch.float16, np.float16)
    }

    for output in ort_sess.get_outputs():
        name = output.name
        
        # Handle dynamic batch sizes (e.g., if ONNX says 'batch_size' instead of 1)
        shape = tuple([1 if (s is None or isinstance(s, str)) else s for s in output.shape])
        
        pt_type, np_type = type_map[output.type]
        
        # Pre-allocate exactly what ONNX wants
        tensor = torch.empty(shape, dtype=pt_type, device=DEVICE).contiguous()
        out_tensors[name] = tensor
        
        io_binding.bind_output(
            name=name,
            device_type='cuda',
            device_id=0,
            element_type=np_type,
            shape=shape,
            buffer_ptr=tensor.data_ptr()
        )

    # Warmup
    for _ in range(5):
        ort_sess.run_with_iobinding(io_binding)

    start_time = time.perf_counter()
    with torch.inference_mode(): # Good practice to wrap the loop
        for _ in range(10):
            # Run the model. The results instantly appear in `out_tensors`.
            ort_sess.run_with_iobinding(io_binding)
            
            # No conversion needed! Just pass them to the decoder dictionary.
            features_dict = {
                "image_embed": out_tensors['image_embed'].to(torch.float32),
                "high_res_feats": [out_tensors['high_res_0'].to(torch.float32), 
                                   out_tensors['high_res_1'].to(torch.float32)]
            }
    end_time = time.perf_counter()
    
    io_time = ((end_time - start_time) / 10) * 1000
    print(f" -> I/O Binding Avg Time: {io_time:.2f} ms ({1000/io_time:.1f} FPS)")

if __name__ == "__main__":
    benchmark_io_binding("image.png")
    clear_memory()