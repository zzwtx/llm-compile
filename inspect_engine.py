import tensorrt_llm
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
with open("/home/xsf/fl/edge/models/trt_0120_engines/llm/rank0.engine", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    print(f"Tensor: {name}, Mode: {mode}, Shape: {shape}, Dtype: {dtype}")
