"""Default CPU preflight. Explicit --train required; no implicit old data/resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from training_data import read_rows,prepare,split_audit,AnswerOnlyCollator,audit_batches
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

def training_arguments(output_dir,epochs=3,save_model_only=False):
    from transformers import TrainingArguments
    return TrainingArguments(output_dir=str(output_dir),per_device_train_batch_size=1,
        per_device_eval_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-5,
        num_train_epochs=epochs,logging_steps=5,eval_strategy='epoch',save_strategy='epoch',
        save_total_limit=2,save_only_model=save_model_only,load_best_model_at_end=True,metric_for_best_model='eval_loss',
        greater_is_better=False,prediction_loss_only=True,bf16=True,fp16=False,
        gradient_checkpointing=True,optim='paged_adamw_8bit',report_to='none',
        warmup_ratio=0.1,weight_decay=0.01,remove_unused_columns=False,seed=42)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-data',type=Path,required=True)
    ap.add_argument('--valid-data',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,default=ROOT/'outputs/llama_v10_cbnu')
    ap.add_argument('--init-adapter',type=Path,help='Continue this existing local adapter; fresh optimizer, not exact checkpoint resume')
    ap.add_argument('--max-length',type=int,default=4096)
    ap.add_argument('--report',type=Path,default=ROOT/'outputs/archive/training_preflight/report.json')
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--chunked-head-loss-tokens',type=int,default=0,help='Opt-in loss-only lm_head+CE chunks; no full logits')
    ap.add_argument('--bf16-frozen-io',action='store_true',help='Exactly representable frozen embedding/head weights only; norms remain FP32')
    ap.add_argument('--save-model-only',action='store_true',help='Save evaluation candidates without optimizer/scheduler/RNG; exact resume unavailable')
    ap.add_argument('--memory-efficient-sdpa',action='store_true',help='Allow flash/efficient SDPA only; fail instead of math/cuDNN fallback')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--train',action='store_true')
    mode.add_argument('--prepare-only',action='store_true',help='Initialize Trainer without optimizer steps')
    args=ap.parse_args()
    from continued_adapter import adapter_spec,attach_existing
    continuation=adapter_spec(args.init_adapter) if args.init_adapter else None
    if args.chunked_head_loss_tokens<0: ap.error('Head-chunk size must be nonnegative')
    if args.max_length<2 or args.epochs<1: ap.error('Invalid length/epochs')
    runtime=args.train or args.prepare_only
    if runtime and (args.output_dir is None or args.output_dir.exists()):
        ap.error('A NEW --output-dir is required; no automatic checkpoint resume')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer),local_files_only=True)
    if args.max_length>tokenizer.model_max_length: ap.error('Length exceeds model context')
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    raw={'train':read_rows(args.train_data),'validation':read_rows(args.valid_data)}
    if args.train and any(r.get('diagnostic_only') for rows in raw.values() for r in rows):
        raise SystemExit('Diagnostic fixtures cannot be used for training')
    encoded,reports={},{}
    collator=AnswerOnlyCollator(tokenizer.pad_token_id)
    for split,rows in raw.items():
        encoded[split],reports[split]=prepare(rows,tokenizer,args.max_length)
        if reports[split]['passed']: reports[split]['actual_collator']=audit_batches(encoded[split],collator)
    separation=split_audit(raw['train'],raw['validation'])
    passed=all(r['passed'] for r in reports.values()) and separation['passed']
    decision_report=None
    report={'training_started':False,'max_length':args.max_length,'truncation':False,
        'save_only_model':args.save_model_only,
        'assistant_only_loss':True,'splits':reports,'split_audit':separation,'passed':passed,
        'ready_for_training':passed and separation['group_metadata_complete'] and not any(
            r.get('diagnostic_only') for rows in raw.values() for r in rows),
        'data_hashes':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.train_data,args.valid_data)},
        'checkpoint_selection':'eval_loss candidate only; generation quality evaluation required'}
    report['initial_adapter']=continuation
    report['chunked_head_loss_tokens']=args.chunked_head_loss_tokens
    report['bf16_frozen_io']=args.bf16_frozen_io
    report['continuation_mode']='existing_adapter_fresh_optimizer' if continuation else 'new_lora'
    report['decision_audit']=decision_report
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':passed,'report':str(args.report),
        'splits':{k:{x:v[x] for x in ('rows','max_tokens','passed')} for k,v in reports.items()},
        'split_audit':separation},ensure_ascii=False),flush=True)
    if not passed: raise SystemExit('Preflight failed. No model load or training.')
    if not runtime: return
    if not separation['group_metadata_complete']:
        raise SystemExit('source_group_ids required for every row before model preparation/training')
    import torch
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from huggingface_hub import snapshot_download
    from datasets import Dataset
    from transformers import Trainer,AutoModelForCausalLM,BitsAndBytesConfig
    base=continuation['base'] if continuation else snapshot_download('meta-llama/Llama-3.1-8B-Instruct',local_files_only=True)
    print('Loading local 4bit model for Trainer preparation; no training yet.',flush=True)
    mt=AutoTokenizer.from_pretrained(base,local_files_only=True)
    if mt.get_vocab()!=tokenizer.get_vocab(): raise ValueError('Model/tokenizer vocabulary mismatch')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This configuration requires CUDA with bf16 support')
    torch.manual_seed(42)
    if args.memory_efficient_sdpa:
        # This Windows build has no FlashAttention. Native GQA with an
        # unmasked sequence falls back to quadratic-memory math attention.
        # Use Transformers' existing repeat_kv path, as for padded batches.
        from transformers.integrations import sdpa_attention
        sdpa_attention.use_gqa_in_sdpa=lambda attention_mask,key: False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    report['memory_efficient_sdpa']=args.memory_efficient_sdpa
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True),
        device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                          gradient_checkpointing_kwargs={'use_reentrant':False})
    model=attach_existing(model,continuation) if continuation else get_peft_model(model,LoraConfig(r=32,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_alpha=64,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM'))
    if args.bf16_frozen_io:
        from continued_adapter import bf16_frozen_io
        report['io_precision_check']=bf16_frozen_io(model)
    # Standard Trainer preserves pretokenized IDs and labels; no SFT reprocessing.
    trainer_type=Trainer
    if args.chunked_head_loss_tokens:
        from chunked_head_loss import trainer_class
        trainer_type=trainer_class(args.chunked_head_loss_tokens)
    trainer=trainer_type(model=model,args=training_arguments(args.output_dir,args.epochs,args.save_model_only),
        train_dataset=Dataset.from_list(encoded['train']),eval_dataset=Dataset.from_list(encoded['validation']),
        processing_class=tokenizer,data_collator=collator)
    for dataset in (trainer.train_dataset,trainer.eval_dataset): audit_batches(list(dataset),trainer.data_collator)
    batch=next(iter(trainer.get_train_dataloader()))
    if not (batch['labels']!=-100).any(): raise ValueError('Trainer batch has no response loss')
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        device_batch={k:v.to(model.device) for k,v in batch.items()}
        loss=(trainer.compute_loss(model,device_batch) if args.chunked_head_loss_tokens else model(**device_batch).loss).item()
    if not __import__('math').isfinite(loss): raise ValueError('Non-finite preparation loss')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report['runtime_check']={'trainer_initialized':True,'optimizer_created':trainer.optimizer is not None,
                             'global_step':trainer.state.global_step,'batch_supervised_tokens':int((batch['labels']!=-100).sum().item()),
                             'forward_loss':loss,'backend':'transformers Trainer + PEFT QLoRA'}
    (args.output_dir/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.prepare_only or os.environ.get('FINAL_LLAMA8B_AUTO_V2_STOP_BEFORE_TRAIN')=='1':
        print('Trainer prepared. No optimizer steps or training performed.',flush=True)
        return
    report['training_started']=True
    (args.output_dir/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    del batch,device_batch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result=trainer.train()
    trainer.save_model(str(args.output_dir/'best_eval_loss_candidate'))
    tokenizer.save_pretrained(str(args.output_dir/'best_eval_loss_candidate'))
    completion={'status':'completed','global_step':trainer.state.global_step,
        'metrics':result.metrics,'best_model_checkpoint':trainer.state.best_model_checkpoint,
        'best_metric':trainer.state.best_metric,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'source_adapter_unchanged':continuation==adapter_spec(args.init_adapter) if continuation else None}
    (args.output_dir/'completion.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')

if __name__=='__main__': main()
