from transformers import BitsAndBytesConfig
import jax.numpy as jnp

def quantize_to_4bit_normal_float(tensor: jnp.ndarray) -> jnp.ndarray:
    tensor = jnp.array(tensor, dtype=jnp.float32)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=jnp.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    quantized_tensor = quant_config.quantize(tensor)
    return quantized_tensor

def dequantize_from_4bit_normal_float(quantized_tensor: jnp.ndarray) -> jnp.ndarray:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=jnp.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    dequantized_tensor = quant_config.dequantize(quantized_tensor)
    return dequantized_tensor
