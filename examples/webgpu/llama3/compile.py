from pathlib import Path
from examples.llama3 import build_transformer
from tinygrad.helpers import fetch
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from extra.export_model import export_model
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    #Device.DEFAULT = "CUDA"

    fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model",
          "tokenizer.model", subdir="llama3-1b-instruct")
    model_name = fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct") 
    model = build_transformer(model_name, model_size="1B", quantize=None, device=Device.DEFAULT)
    #print(get_parameters(model))

    #prg, inp_sizes, out_sizes, state = export_model(model, Device.DEFAULT.lower(), Tensor.randn(1,3,416,416), model_name="yolov8")
    #dirname = Path(__file__).parent
    #safe_save(state, (dirname / "net.safetensors").as_posix())
    #with open(dirname / f"net.js", "w") as text_file:
    #   text_file.write(prg)
