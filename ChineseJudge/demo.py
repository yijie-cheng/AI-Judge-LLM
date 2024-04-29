import os
import sys
import json
import argparse
import logging
import math
import numpy as np
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_norm_
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    default_data_collator,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    MT5ForConditionalGeneration, MT5Tokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    GenerationConfig,
)
from utils import get_prompt
from utils import get_bnb_config
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    AdaLoraConfig,
    PeftType,
    PeftConfig,
    PrefixTuningConfig,
    PromptEncoderConfig, LoraConfig, PromptTuningConfig, PeftModel,
)
#from trl import SFTTrainer
#from vllm import LLM, SamplingParams

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def parse_args():
    parser = argparse.ArgumentParser(description="Language Generation")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="The path of the model.",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        default=None,
        help="The path of the adapter.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
accelerator = Accelerator()

device_map = {"":0}
tokenizer = None
model = None
if args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,  tokenizer_type='llama')

else:
    tokenizer = AutoTokenizer.from_pretrained("yentinglin/Taiwan-LLM-7B-v2.0-chat", revision="5073b2bbc1aa5519acdc865e99832857ef47f7c9", use_fast=False,  tokenizer_type='llama')
bnb_config = get_bnb_config()


if args.model_name_or_path:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16,quantization_config=bnb_config)

else:
    model = AutoModelForCausalLM.from_pretrained("yentinglin/Taiwan-LLM-7B-v2.0-chat", revision="5073b2bbc1aa5519acdc865e99832857ef47f7c9",torch_dtype=torch.bfloat16,quantization_config=bnb_config)


if args.peft_path:
    peft_config = PeftConfig.from_pretrained(args.peft_path)
    model.add_adapter(peft_config)
    model = PeftModel.from_pretrained(model, args.peft_path)
    model.to('cuda')
    model.enable_adapters()
else:
    pass
generation_config = GenerationConfig(
        temperature=1.2,
        top_p=0.9,
        top_k=10,
        num_beams=1,
        do_sample=True,
        )
model.eval()
while True:
    text = input("請輸入你需要法官判決的事件（若想退出請輸入0）：")
    if text=="0":
        break
    #inputs = text
    inputs = get_prompt(text)
    inputs = tokenizer(inputs, return_tensors="pt")
    with torch.no_grad():
        tokens = model.generate(input_ids=inputs["input_ids"].to('cuda'), generation_config=generation_config, return_dict_in_generate=True, max_new_tokens=256)
    tokens = tokens.sequences[0].cpu().numpy()
    pred = tokenizer.decode(tokens, skip_special_tokens=True)
    if len(pred.split("ASSISTANT:"))>1:
        pred = pred.split("ASSISTANT:")[1].strip()
    print("法官判決為：",pred)
    print("請嚴肅看待您的判決，並請改進。")
    print("-------------------------------")

