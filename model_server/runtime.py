"""Local 4bit model loading shared by the slim service."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRANSPORT = 'notiai-model-candidate-v1'






class Runtime:
    def __init__(self, model_dir, max_new_tokens=None, context_limit=12288, max_seconds=80, prompt_mode='compact'):
        if prompt_mode not in ('compact', 'v10-native'): raise ValueError('unknown prompt mode')
        self.prompt_mode = prompt_mode
        max_new_tokens = max_new_tokens if max_new_tokens is not None else 512 if prompt_mode == 'v10-native' else 2048
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
        self.torch = torch
        model_dir = Path(model_dir).resolve()
        config = json.loads((model_dir / 'adapter_config.json').read_text(encoding='utf-8'))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(config['base_model_name_or_path'], local_files_only=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
            device_map={'': 0}, torch_dtype=torch.float16, attn_implementation='sdpa')
        self.model = PeftModel.from_pretrained(base, str(model_dir), local_files_only=True)
        self.model.eval()
        for name in ('temperature', 'top_p', 'top_k'): setattr(self.model.generation_config, name, None)
        self.max_new_tokens, self.context_limit, self.max_seconds = max_new_tokens, context_limit, max_seconds
        self.identity = {'checkpoint': model_dir.name, 'adapter_sha256': hashlib.sha256(
            (model_dir / 'adapter_model.safetensors').read_bytes()).hexdigest(),
            'prompt_mode': prompt_mode,
            'postprocess_sha256': None,
            'training': False, 'max_new_tokens': max_new_tokens, 'context_limit': context_limit}
