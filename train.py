import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from tqdm import tqdm
from accelerate import Accelerator

from model import Encoder, Decoder, SeqToSeqTransformer
from utils import Config

logger = logging.getLogger(__name__)

accelerator = Accelerator()
device = accelerator.device

config = Config.load_config_file(Path(__file__).parent.joinpath("config.json"))
logger.info("Configuration loaded.")

dataset = load_dataset(config.train_data, "de-en", split="train")
logger.info("Dataset loaded.")

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tokenizer.add_special_tokens({"cls_token": "<s>"})
logger.info("Tokenizer loaded with special tokens.")


def tokenize(batch):
    inputs = [example["en"] for example in batch["translation"]]
    targets = [example["de"] for example in batch["translation"]]
    model_inputs = tokenizer(inputs, max_length=config.max_seq_length, truncation=True)
    labels = tokenizer(
        text_target=targets, max_length=config.max_seq_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names
)
logger.info("Dataset tokenized.")

data_collator = DataCollatorForSeq2Seq(tokenizer)

train_data_loader = DataLoader(
    tokenized_dataset,
    batch_size=config.batch_size,
    num_workers=4,
    prefetch_factor=2,
    shuffle=True,
    collate_fn=data_collator,
)
logger.info("Train data loader ready.")

encoder = Encoder(
    tokenizer.vocab_size + 1,
    config.model_dim,
    config.hidden_dims,
    config.hidden_dims,
    config.max_seq_length,
    config.num_heads,
    config.num_layers,
    config.dropout,
)
decoder = Decoder(
    tokenizer.vocab_size + 1,
    config.model_dim,
    config.hidden_dims,
    config.hidden_dims,
    config.max_seq_length,
    config.num_heads,
    config.num_layers,
    config.dropout,
)
seq_to_seq_model = SeqToSeqTransformer(encoder, decoder)
logger.info("Model initialized.")


criteria = nn.CrossEntropyLoss(
    ignore_index=-100
)  # The tokenizer sets the targets padding to -100
optimizer = torch.optim.Adam(seq_to_seq_model.parameters(), lr=config.learning_rate)

seq_to_seq_model, optimizer, train_data_loader = accelerator.prepare(
    seq_to_seq_model, optimizer, train_data_loader
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_data_loader), epochs=config.num_epochs)

def save_model(model, optimizer, epoch, save_path="model_checkpoint.pt"):
    accelerator.save({
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def train(
    model: nn.Module,
    num_epochs: int,
    train_data_loader: DataLoader,
    criteria: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    start_token_id: int,
    pad_token_id: int,
):
    logger.info("Training started.")
    for epoch in range(num_epochs):
        logger.info(f"Epoch: {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = []
        for step, batch in enumerate(tqdm(train_data_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            encoder_input = batch["input_ids"]
            targets = batch["labels"]

            decoder_input = (
                targets.clone().detach()
            )  # Clone the targets so the original does not affected.

            decoder_input = torch.roll(
                decoder_input, shifts=1, dims=1
            )  # We have to shift the input so the model can learn what comes after each token.

            decoder_input[:, 0] = (
                start_token_id  # We have to make sure very input starts with the start token ID
            )
            decoder_input = decoder_input.masked_fill(
                decoder_input == -100, pad_token_id
            )

            encoder_mask = batch["attention_mask"]

            decoder_mask = torch.ones_like(
                decoder_input
            )  # Create a tensor like the decoder_input filled with ones
            decoder_mask = decoder_mask.masked_fill(decoder_mask == pad_token_id, 0)

            outputs = model(
                    encoder_input, decoder_input, encoder_mask, decoder_mask
                )

            loss = criteria(
                    outputs.transpose(2, 1), targets
                )  # (batch_size, seq_length, vocab_size) -> (batch_size, vocab_size, seq_length)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss.append(loss.item())
            if step % 10 == 0:  # Log every 10 steps
                logger.info(
                    f"Step {step}/{len(train_data_loader)}, Loss: {loss.item():.4f}"
                )

        avg_loss = np.mean(epoch_loss)
        accelerator.wait_for_everyone()
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            save_model(model, optimizer, epoch)
            logger.info("Model saved")


if __name__ == "__main__":
    train(
        seq_to_seq_model,
        config.num_epochs,
        train_data_loader,
        criteria,
        optimizer,
        scheduler,
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
    )
