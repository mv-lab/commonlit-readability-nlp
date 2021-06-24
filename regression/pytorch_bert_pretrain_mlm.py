import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Process pytorch params.')
parser.add_argument('-model', '--model', type=str, help='Pytorch (timm) model name')
parser.add_argument('-model_dir', '--model_dir', type=str, default='test', help='Model save dir name')
parser.add_argument('-batch_size', type=int, default=32, help='batch size')
parser.add_argument('-epochs', '--epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('-debug', action='store_true', help='batch size')
parser.add_argument('-grad_accum', type=int, default=1, help='gradient accumulation step')
parser.add_argument('-max_len', type=int, default=256, help='max seq len')
parser.add_argument('-lr', type=float, default=1.5e-5, help='learning rate')

args = parser.parse_args()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

mlm_data = train[['excerpt']]
mlm_data = mlm_data.rename(columns={'excerpt':'text'})
mlm_data.to_csv('mlm_data.csv', index=False)

mlm_data_val = test[['excerpt']]
mlm_data_val = mlm_data_val.rename(columns={'excerpt':'text'})
mlm_data_val.to_csv('mlm_data_val.csv', index=False)

# %% [markdown]
# ### Import Dependencies

# %% [code] {"execution":{"iopub.status.busy":"2021-05-26T17:30:55.985669Z","iopub.execute_input":"2021-05-26T17:30:55.986269Z","iopub.status.idle":"2021-05-26T17:31:05.404016Z","shell.execute_reply.started":"2021-05-26T17:30:55.986232Z","shell.execute_reply":"2021-05-26T17:31:05.4031Z"}}
import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    CONFIG_MAPPING, 
    MODEL_MAPPING, 
    AdamW, 
    AutoConfig, 
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling, 
    SchedulerType, 
    get_scheduler, 
    set_seed
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# from pprint import pprint
# pprint(MODEL_TYPES, width=3, compact=True)

# %% [markdown]
# ### Config

# %% [code] {"execution":{"iopub.status.busy":"2021-05-26T17:31:05.405597Z","iopub.execute_input":"2021-05-26T17:31:05.40595Z","iopub.status.idle":"2021-05-26T17:31:05.415453Z","shell.execute_reply.started":"2021-05-26T17:31:05.405913Z","shell.execute_reply":"2021-05-26T17:31:05.414266Z"}}
class TrainConfig:
    train_file= 'mlm_data.csv'
    validation_file = 'mlm_data.csv'
    validation_split_percentage= 5
    pad_to_max_length= True
    model_name_or_path= args.model
    config_name= args.model
    tokenizer_name= args.model
    use_slow_tokenizer= True
    per_device_train_batch_size= args.batch_size
    per_device_eval_batch_size= args.batch_size
    learning_rate= args.lr
    weight_decay= 0.0
    num_train_epochs= args.epochs # change to 5
    max_train_steps= None
    gradient_accumulation_steps= args.grad_accum
    lr_scheduler_type= 'constant_with_warmup'
    num_warmup_steps= 0
    output_dir= args.model_dir
    seed= 2021
    model_type= 'roberta'
    max_seq_length= args.max_len
    line_by_line= False
    preprocessing_num_workers= 4
    overwrite_cache= True
    mlm_probability= 0.15

config = TrainConfig()
if 'reform' in args.model:
    config.is_decoder = False

if config.train_file is not None:
    extension = config.train_file.split(".")[-1]
    assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
if config.validation_file is not None:
    extension = config.validation_file.split(".")[-1]
    assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
if config.output_dir is not None:
    os.makedirs(config.output_dir, exist_ok=True)

# %% [markdown]
# ### Run

# %% [code] {"execution":{"iopub.status.busy":"2021-05-26T17:31:05.416962Z","iopub.execute_input":"2021-05-26T17:31:05.417338Z","iopub.status.idle":"2021-05-26T17:31:05.454587Z","shell.execute_reply.started":"2021-05-26T17:31:05.417301Z","shell.execute_reply":"2021-05-26T17:31:05.4537Z"}}
def main():
    model_args = TrainConfig()
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if model_args.seed is not None:
        set_seed(model_args.seed)

    data_files = {}
    if model_args.train_file is not None:
        data_files["train"] = model_args.train_file
    if model_args.validation_file is not None:
        data_files["validation"] = model_args.validation_file
    extension = model_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)
    
    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if model_args.model_name_or_path:
        if 'reform' in args.model:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer, mlm=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
       
    if 'reform' in args.model:
                   config.is_decoder = False

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)


    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if model_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if model_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({model_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(model_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=model_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not model_args.overwrite_cache,
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=model_args.preprocessing_num_workers,
        load_from_cache_file=not model_args.overwrite_cache,
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    if 'reform' in args.model:
       mlm_arg = False
    else:
       mlm_arg = True
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=model_args.mlm_probability, mlm=mlm_arg)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=model_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=model_args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": model_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=model_args.learning_rate)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / model_args.gradient_accumulation_steps)
    if model_args.max_train_steps is None:
        model_args.max_train_steps = model_args.num_train_epochs * num_update_steps_per_epoch
    else:
        model_args.num_train_epochs = math.ceil(model_args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=model_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=model_args.num_warmup_steps,
        num_training_steps=model_args.max_train_steps,
    )

    total_batch_size = model_args.per_device_train_batch_size * accelerator.num_processes * model_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {model_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {model_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {model_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {model_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(model_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(model_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / model_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % model_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= model_args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(model_args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        perplexity = math.exp(torch.mean(losses))

        print(f"epoch {epoch}: perplexity: {perplexity}")

    if model_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_args.output_dir, save_function=accelerator.save)

# %% [code] {"execution":{"iopub.status.busy":"2021-05-26T17:31:05.455842Z","iopub.execute_input":"2021-05-26T17:31:05.456294Z","iopub.status.idle":"2021-05-26T17:35:18.700137Z","shell.execute_reply.started":"2021-05-26T17:31:05.456253Z","shell.execute_reply":"2021-05-26T17:35:18.697848Z"}}
if __name__ == "__main__":
    main()
