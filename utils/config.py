import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict


logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    format="%(levelname)s [%(module)s]: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    train_data: str = field(default="data/train.jsonl")
    valid_data: str = field(default="data/valid.jsonl")
    batch_size: int = field(default=64)
    num_epochs: int = field(default=10)
    learning_rate: float = field(default=0.0001)
    model_dim: int = field(default=512)
    hidden_dims: int = field(default=64)
    num_heads: int = field(default=8)
    num_layers: int = field(default=6)
    dropout: float = field(default=0.1)
    max_seq_length: int = field(default=100)
    mixed_precision: bool = field(default=True)

    @classmethod
    def load_config_file(cls, file_path: str) -> "Config":
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The configuration file {file_path} does not exist."
            )
        with open(file_path, "r") as config_file:
            config_data: Dict[str, Any] = json.load(config_file)
            logger.info("Config loaded!")
        return cls(**config_data)


if __name__ == "__main__":
    config = Config.load_config_file("config.json")
    print(config)
