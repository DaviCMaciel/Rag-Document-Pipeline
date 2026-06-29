import torch
import logging

def get_device() -> str:
    """Retorna o melhor dispositivo disponível para processamento."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def setup_logging(level=logging.INFO):
    """Configuração central do logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )