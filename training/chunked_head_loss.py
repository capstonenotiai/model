"""Loss-only LoRA path: checkpoint lm_head plus CE per token chunk, no full logits."""
def loss_only_forward(model,inputs,chunk_tokens=256,num_items_in_batch=None):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    if chunk_tokens<1:raise ValueError('Positive chunk size required')
    base=model.get_base_model()
    if base.config.model_type!='llama':raise ValueError('Only local Llama path verified')
    if any(p.requires_grad for p in base.lm_head.parameters()):raise ValueError('This path requires frozen lm_head')
    if any(getattr(c,'is_prompt_learning',False) or getattr(c,'alora_invocation_tokens',None) for c in model.peft_config.values()):
        raise ValueError('Only ordinary LoRA supported')
    hidden=base.model(input_ids=inputs['input_ids'],attention_mask=inputs.get('attention_mask'),use_cache=False,return_dict=True).last_hidden_state
    targets=inputs['labels'][...,1:].reshape(-1).to(hidden.device)
    values=hidden[...,:-1,:].reshape(-1,hidden.shape[-1])
    count=(targets!=-100).sum()
    if not count.item():raise ValueError('No supervised tokens')
    def part_loss(x,y):return F.cross_entropy(base.lm_head(x).float(),y,ignore_index=-100,reduction='sum')
    total=None
    for start in range(0,len(targets),chunk_tokens):
        x=values[start:start+chunk_tokens];y=targets[start:start+chunk_tokens]
        if not (y!=-100).any().item():continue
        loss=checkpoint(part_loss,x,y,use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() and x.requires_grad else part_loss(x,y)
        total=loss if total is None else total+loss
    divisor=count if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(divisor):divisor=divisor.to(total.device)
    return total/divisor

def trainer_class(chunk_tokens):
    from transformers import Trainer
    class LossOnlyTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            import torch
            from contextlib import nullcontext
            # Direct base.model/lm_head calls bypass Accelerate's wrapped forward.
            context=torch.autocast('cuda',dtype=torch.bfloat16) if self.args.bf16 and inputs['input_ids'].is_cuda else nullcontext()
            with context:
                loss=loss_only_forward(model,inputs,chunk_tokens,num_items_in_batch)
            return (loss,{'loss':loss}) if return_outputs else loss
    return LossOnlyTrainer
