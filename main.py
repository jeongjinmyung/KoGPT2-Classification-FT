import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
import math
import pickle
import pandas as pd
import numpy as np
from time import time
import json
import wandb

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from copy import deepcopy

import transformers
from transformers import AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import set_seed

from config import KoGPT2Config
from data import label_ids, ClassificationDataset
from model import SimpleGPT2SequenceClassifier

import gc

logger = logging.getLogger(__name__)


@hydra.main(config_name='config', version_base=None)
def main(cfg: KoGPT2Config):

    accelerator_log_kwargs = {}
    if cfg.report.with_tracking:
        wandb.login(key=cfg.report.wandb_id, relogin=True)
        accelerator_log_kwargs["log_with"] = cfg.report.name

    accelerator = Accelerator(**accelerator_log_kwargs)

    logger.info(accelerator.state)

    if cfg.param.seed is not None:
        set_seed(cfg.param.seed)

    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = SimpleGPT2SequenceClassifier(
        hidden_size=cfg.model.hidden_size, 
        num_classes=len(label_ids), 
        max_seq_len=cfg.data.max_seq_len,
        gpt_model_name=cfg.model.name, 
        tokenizer=tokenizer,
        )


    train_data = pd.read_csv(os.path.join(cfg.path.dataset_path, 'train.csv'))
    valid_data = pd.read_csv(os.path.join(cfg.path.dataset_path, 'valid.csv'))

    train_dataset = ClassificationDataset(df=train_data, tokenizer=tokenizer,label_ids=label_ids, max_length=cfg.data.max_seq_len)
    valid_dataset = ClassificationDataset(df=valid_data, tokenizer=tokenizer, label_ids=label_ids, max_length=cfg.data.max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.param.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=cfg.param.batch_size)


    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.param.lr)
    
    model = model.to(device)
    criterion = criterion.to(device)

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if cfg.report.with_tracking:
        experiment_config = vars(cfg.report)
        accelerator.init_trackers("KoGPT2_finetuning", experiment_config)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    best_acc = 0

    model.train()
    for epoch in range(cfg.param.epochs):
        
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            
            optimizer.zero_grad()

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1)==train_label).sum().item()
            total_acc_train += acc

            accelerator.backward(batch_loss)
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output = model(input_id, mask)
                
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)

            train_loss = total_loss_train/len(train_dataloader)
            train_acc = total_acc_train / len(train_dataloader)
            val_loss = total_loss_val / len(val_dataloader)
            val_acc = total_acc_val / len(val_dataloader)

            logger.info(
            f"Epochs: {epoch + 1} | Train Loss: {train_loss: .3f} \
            | Train Accuracy: {train_acc: .3f} \
            | Val Loss: {val_loss: .3f} \
            | Val Accuracy: {val_acc: .3f} \
            | Elapsed time: {elapsed_time: .3f} ms")

            is_best = val_acc > best_acc
            best_acc = min(best_acc, val_acc)

            if is_best:
                torch.save(model.state_dict(), os.path.join(cfg.path.output_dir, 'full_ft.pt'))
                with open(os.path.join(cfg.path.output_dir, 'all_results.json'), 'w+') as f:
                    json.dump({'epoch': epoch+1,
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'valid_loss': val_loss,
                            'val_acc': val_acc,
                            'elapsed time': elapsed_time}, f)

                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()