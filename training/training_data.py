"""Untruncated chat tokenization and explicit assistant-only loss."""
import hashlib
import json
from pathlib import Path

def read_rows(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').split('\n') if x.strip()]
    if not rows: raise ValueError('Empty dataset')
    return rows

def encode_row(row, tokenizer):
    messages = row.get('messages')
    if not isinstance(messages,list) or [m.get('role') for m in messages] not in (
            ['system','user','assistant'],['user','assistant']):
        raise ValueError('Expected one user/assistant pair with optional system')
    if any(not isinstance(m.get('content'),str) or not m['content'].strip() for m in messages):
        raise ValueError('Empty/non-text message')
    full = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    ids = tokenizer(full,add_special_tokens=False,truncation=False)['input_ids']
    pre = tokenizer(prefix,add_special_tokens=False,truncation=False)['input_ids']
    if not pre or ids[:len(pre)] != pre: raise ValueError('Response token boundary mismatch')
    if len(ids)<=len(pre)+1: raise ValueError('Missing answer')
    if tokenizer.eos_token_id is None or ids[-1]!=tokenizer.eos_token_id:
        raise ValueError('Missing answer end-of-turn token')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(pre)+ids[len(pre):]}

def prepare(rows,tokenizer,max_length):
    encoded,lengths,errors = [],[],[]
    for index,row in enumerate(rows):
        try:
            e=encode_row(row,tokenizer); n=len(e['input_ids'])
            lengths.append({'row':index,'tokens':n,'answer_tokens':sum(x!=-100 for x in e['labels'])})
            if n>max_length: errors.append({'row':index,'reason':'length_exceeded','tokens':n,'limit':max_length})
            encoded.append(e)
        except (ValueError,KeyError,TypeError) as exc:
            errors.append({'row':index,'reason':str(exc)})
    return encoded,{'rows':len(rows),'max_tokens':max((x['tokens'] for x in lengths),default=0),
                    'lengths':lengths,'errors':errors,'passed':not errors}

def split_audit(train,valid):
    def inputs(rows):
        return {hashlib.sha256(json.dumps([m for m in r['messages'] if m['role']=='user'],
                    ensure_ascii=False,sort_keys=True).encode()).hexdigest() for r in rows}
    def groups(rows): return {str(g) for r in rows for g in r.get('source_group_ids',[])}
    overlap=inputs(train)&inputs(valid); group_overlap=groups(train)&groups(valid)
    return {'duplicate_inputs':len(overlap),'overlapping_groups':sorted(group_overlap),
            'group_metadata_complete':all(bool(r.get('source_group_ids')) for r in train+valid),
            'passed':not overlap and not group_overlap}

class AnswerOnlyCollator:
    """Padding uses positions, never token value: EOS may equal pad token."""
    def __init__(self,pad_token_id): self.pad_token_id=pad_token_id
    def __call__(self,features):
        import torch
        width=max(len(f['input_ids']) for f in features)
        result={k:[] for k in ('input_ids','attention_mask','labels')}
        for f in features:
            ids,labels=f['input_ids'],f['labels']
            if len(ids)!=len(labels) or len(ids)!=len(f['attention_mask']): raise ValueError('Length mismatch')
            starts=[i for i,v in enumerate(labels) if v!=-100]
            if not starts or starts[0]==0 or labels[starts[0]:]!=ids[starts[0]:]:
                raise ValueError('Missing/partial/corrupted assistant labels')
            if f['attention_mask']!=[1]*len(ids): raise ValueError('Unexpected padding')
            n=width-len(ids)
            result['input_ids'].append(ids+[self.pad_token_id]*n)
            result['attention_mask'].append([1]*len(ids)+[0]*n)
            result['labels'].append(labels+[-100]*n)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in result.items()}

def audit_batches(encoded,collator):
    total=0
    for start in range(0,len(encoded),2):
        examples=encoded[start:start+2]; batch=collator(examples)
        for i,e in enumerate(examples):
            n=len(e['input_ids'])
            assert batch['labels'][i,:n].tolist()==e['labels']
            assert batch['input_ids'][i,:n].tolist()==e['input_ids']
            assert all(v==-100 for v in batch['labels'][i,n:].tolist())
            total+=sum(v!=-100 for v in e['labels'])
    return {'rows_checked':len(encoded),'supervised_tokens':total,'passed':True}
