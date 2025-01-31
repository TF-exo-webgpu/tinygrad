# from pathlib import Path
# import json
# import os
# from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
# from extra.export_model import export_model
# # from examples.llama3 import  build_transformer
# from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
# from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters
# from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm
#
# if __name__ == "__main__":
#     Device.DEFAULT = "WEBGPU"
#
#     # Set model size
#     model_size = "1B"  # Change to "1B" or "70B" if needed
#     model_quantize = None  # Change to "int8" or "bf16" if needed
#
#     MODEL_PARAMS = {
#         "1B": {"dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192},
#         "8B": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336},
#         "70B": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672},
#     }
#
#     # Define model storage directory
#     model_dir = Path(f"./llama3-{model_size.lower()}-model")
#     model_dir.mkdir(parents=True, exist_ok=True)
#
#     # Download model weights
#     if model_size == "1B":
#         fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", model_dir / "tokenizer.model")
#         model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", model_dir / "Llama-3.2-1B-Instruct-Q6_K.gguf")
#     elif model_size == "8B":
#         fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", model_dir / "tokenizer.model")
#         fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00001-of-00004.safetensors", model_dir / "model-00001-of-00004.safetensors")
#         fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00002-of-00004.safetensors", model_dir / "model-00002-of-00004.safetensors")
#         fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00003-of-00004.safetensors", model_dir / "model-00003-of-00004.safetensors")
#         fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00004-of-00004.safetensors", model_dir / "model-00004-of-00004.safetensors")
#         model_path = fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json", model_dir / "model.safetensors.index.json")
#     elif model_size == "70B":
#         subdir = "DeepSeek-R1-Distill-Llama-70B"
#         model_path = fetch("https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model.safetensors.index.json", model_dir / "model.safetensors.index.json")
#         fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", model_dir / "tokenizer.model")
#         for i in range(17):
#             fetch(f"https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model-{i+1:05d}-of-000017.safetensors", model_dir / f"model-{i+1:05d}-of-000017.safetensors")
#
#     print(f"Model weights downloaded to {model_dir}")
#
#     # model = build_transformer(model_path, model_size=model_size, quantize=model_quantize, device=Device.DEFAULT)
#
#     # Load pre-trained weights
#     # state_dict = safe_load(model_path)
#     if model_path.suffix == ".gguf":
#       print("Loading GGUF model...")
#       state_dict = gguf_load(Tensor.empty(model_path.stat().st_size, dtype=dtypes.uint8, device=f"disk:{model_path}"))[1]
#     elif model_path.suffix == ".safetensors":
#       print("Loading Safetensors model...")
#       state_dict = safe_load(model_path)
#     elif model_path.suffix in [".pth", ".pt"]:
#       print("Loading Torch model...")
#       state_dict = torch_load(model_path)
#     else:
#       raise ValueError(f"Unknown model format: {model_path.suffix}")
#
#
#     model = Transformer(**MODEL_PARAMS[model_size], linear=nn.Linear, max_context=8192, jit=True)
#     if "model.embed_tokens.weight" in state_dict:
#         # state_dict = convert_from_huggingface(state_dict, Transformer, MODEL_PARAMS[model_size]["n_heads"], MODEL_PARAMS[model_size]["n_kv_heads"])
#         state_dict = convert_from_huggingface(state_dict, model, MODEL_PARAMS[model_size]["n_heads"], MODEL_PARAMS[model_size]["n_kv_heads"])
#     state_dict = fix_bf16(state_dict)
#
#     # Initialize the model
#     llama_model = Transformer(**MODEL_PARAMS[model_size], max_context=8192, jit=True)
#     load_state_dict(llama_model, state_dict)
#
#     # Export the compiled model
#     prg, inp_sizes, out_sizes, state = export_model(
#         llama_model, Device.DEFAULT.lower(), Tensor.randn(1, 128), model_name="llama3"
#     )
#
#     # Save the compiled model and state
#     safe_save(state, (model_dir / "llama3.safetensors").as_posix())
#
#     with open(model_dir / "llama3.js", "w") as text_file:
#         text_file.write(prg)
#
#     print("Llama 3 model compiled and saved successfully!")
#

from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, torch_load, load_state_dict
from extra.models.llama import Transformer, convert_from_gguf, fix_bf16
from tinygrad.device import Device
from tinygrad.helpers import fetch
from tinygrad.nn.state import gguf_load
from tinygrad import nn

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"

    model_size = "1B"  # Change to "8B" or "70B" if needed
    model_dir = Path(f"./llama3-{model_size.lower()}-model")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_quantize = None  # Change to "int8" or "bf16" if needed

    MODEL_PARAMS = {
        "1B": {"dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192},
        "8B": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336},
        "70B": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672},
    }

    # Download the model
    model_path = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
                       model_dir / "Llama-3.2-1B-Instruct-Q6_K.gguf")

    print(f"Model weights downloaded to {model_path}")

    # Load GGUF model
    print("Loading GGUF model...")
    gguf_tensor = Tensor.empty(model_path.stat().st_size, dtype=dtypes.uint8, device=f"disk:{model_path}")
    raw_state_dict = gguf_load(gguf_tensor)[1]

    # Debugging: Print the first few keys in the loaded state_dict
    print("Available keys in state_dict:", list(raw_state_dict.keys())[:20])

    tm = Transformer(**MODEL_PARAMS[model_size], linear=nn.Linear, max_context=8192, jit=True)
    # Convert GGUF format to TinyGrad format
    state_dict = convert_from_gguf(raw_state_dict, tm)

    # Fix BF16 mismatches
    state_dict = fix_bf16(state_dict)

    # Initialize model
    MODEL_PARAMS = {
        "1B": {"dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192},
    }
    llama_model = Transformer(**MODEL_PARAMS[model_size], max_context=8192, jit=True)

    # Load model weights
    print("Loading state_dict into model...")
    load_state_dict(llama_model, state_dict)

    print("Model successfully loaded and ready for compilation!")

    # Export compiled model
    prg, inp_sizes, out_sizes, state = export_model(
        llama_model, Device.DEFAULT.lower(), Tensor.randn(1, 128), model_name="llama3"
    )

    # Save compiled model and state
    safe_save(state, (model_dir / "llama3.safetensors").as_posix())
    with open(model_dir / "llama3.js", "w") as text_file:
        text_file.write(prg)

    print("Llama 3 model compiled and saved successfully!")

