"""Load a local, existing LoRA for continued fine-tuning without initializing another."""
import hashlib,json
from pathlib import Path

def adapter_spec(path):
    path=Path(path).resolve()
    files=[path/'adapter_config.json',path/'adapter_model.safetensors']
    if not all(p.is_file() for p in files):raise ValueError('Complete local adapter required')
    config=json.loads(files[0].read_text(encoding='utf-8'))
    if config.get('peft_type')!='LORA':raise ValueError('Expected LoRA adapter')
    base=Path(config['base_model_name_or_path'])
    if not (base/'config.json').is_file():raise ValueError('Adapter base model must exist locally')
    return {'path':str(path),'base':str(base),'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}

def attach_existing(model,spec):
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    import torch
    model=PeftModel.from_pretrained(model,spec['path'],is_trainable=True,local_files_only=True)
    disk=load_file(str(Path(spec['path'])/'adapter_model.safetensors'))
    loaded=get_peft_model_state_dict(model)
    if set(disk)!=set(loaded):raise ValueError('Loaded adapter keys differ from saved adapter')
    for key,value in disk.items():
        if not torch.equal(value,loaded[key].detach().cpu().to(value.dtype)):
            raise ValueError('Loaded adapter differs: '+key)
    if not any(p.requires_grad for p in model.parameters()):raise ValueError('Adapter not trainable')
    return model

def bf16_frozen_io(model):
    """Undo preparation's FP32 upcast only for frozen IO, after exact round-trip checks."""
    import torch
    result={}
    base=model.get_base_model()
    modules=[('input_embeddings',base.get_input_embeddings()),('lm_head',base.get_output_embeddings())]
    for name,module in modules:
        for p in module.parameters():
            if p.requires_grad:raise ValueError('Cannot downcast trainable IO weights')
            # Bounded temporary memory, including very large vocabulary matrices.
            for block in p.detach().split(256,dim=0):
                if not torch.equal(block,block.to(torch.bfloat16).to(block.dtype)):
                    raise ValueError('IO weight not exactly representable as BF16')
        result[name]={'before':str(next(module.parameters()).dtype),'after':'torch.bfloat16','values_exactly_preserved':True}
    for _,module in modules:
        module.to(dtype=torch.bfloat16)
    return result
