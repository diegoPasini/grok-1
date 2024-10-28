from transformers import BitsAndBytesConfig
import torch

def quantize_to_4bit_normal_float(tensor: np.ndarray) -> torch.Tensor:
    tensor = torch.tensor(tensor, dtype=torch.float32)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    quantized_tensor = quant_config.quantize(tensor)
    return quantized_tensor

