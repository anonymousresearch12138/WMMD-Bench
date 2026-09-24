"""7B engineering wrapper around existing SFT dataset and exact Bc objective.

Offline targets retain every vocabulary value as original BF16 bits. No top-k,
quantization, Teacher co-residency, or scientific hyperparameter selection.
"""
import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from wmmd_bench.training.hard_labels import SFTDataset, Collator, TimingCallback
from wmmd_bench.training.objective import ResponseObjective
from wmmd_bench.training.storage import budget
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def put(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(args):
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = SFTDataset(args.dataset, tok, 1024)
    assert len(ds) == 20000
    features = [ds[i] for i in range(len(ds))]
    counts = [sum(v != -100 for v in x['labels'][1:]) for x in features]
    assert min(counts) > 0, 'Zero supervised tokens after canonical truncation'
    vocab = len(tok.get_vocab())
    assert vocab == 32000
    identity = hashlib.sha256(json.dumps(features, separators=(',', ':')).encode()).hexdigest()
    stats = dict(samples=len(ds), max_length=1024, vocabulary_size=vocab,
                 supervised_tokens=sum(counts), min_supervised=min(counts), max_supervised=max(counts),
                 dtype='bfloat16', dtype_bytes=2, representation='full_vocab',
                 payload_bytes=sum(counts)*vocab*2, feature_sha256=identity,
                 dataset_file_sha256=sha(args.dataset), supervised_positions='labels[1:] != -100',
                 suggested_shards=math.ceil(len(ds)/128), samples_per_shard=128,
                 maximum_shard_bytes=max(sum(counts[i:i+128])*vocab*2 for i in range(0,len(ds),128)))
    return tok, ds, features, counts, stats


def cache(args, tok, features, counts, stats):
    target = Path(args.cache)
    assert not target.exists(), 'No cache collision or implicit regeneration'
    tt = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    assert tt.get_vocab() == tok.get_vocab(), 'Token-ID mismatch'
    for f in features[:8]:
        ids = f['input_ids']
        assert tt.decode(ids) == tok.decode(ids)
    # Direct/Paraphrase finals are already counted by disk.used at this stage.
    # Reserve only the remaining Logit final plus metadata/save overhead.
    gate = budget(*shutil.disk_usage(Path(args.dataset).parent),
                  {'remaining_logit_final':13479264256, 'metadata_and_logs':2*2**30},
                  samples=20000, supervised_tokens=stats['supervised_tokens'])
    put(Path(args.output)/'cache_disk_gate.json', gate)
    assert gate['passed'], 'Full-vocab cache disk gate failed before generation'
    target.mkdir()
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                               local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    manifest = dict(stats, teacher=args.teacher, shards=[], records=[], complete=False,
                    serialization_fix='Select canonical shifted supervised positions, then explicit BF16 cast as original Bc',
                    forward_logits_dtypes=[])
    collate = Collator(tok)
    started = time.monotonic()
    for start in range(0,len(features),128):
        end = min(start+128,len(features)); name=f'shard-{start//128:04d}.bf16'
        file=target/name; offset=0
        # Llama 4.44.2 promotes BF16 lm_head output to FP32 on return. Match
        # canonical Bc's explicit selected-target BF16 cast before serialization.
        with file.open('xb') as sink, torch.inference_mode():
            for k in range(start,end,8):
                batch={key:value.cuda() for key,value in collate(features[k:min(k+8,end)]).items()}
                labels=batch.pop('labels')
                logits=model(**batch,use_cache=False).logits[:,:-1]
                assert logits.dtype in (torch.float32, torch.bfloat16)
                if str(logits.dtype) not in manifest['forward_logits_dtypes']:
                    manifest['forward_logits_dtypes'].append(str(logits.dtype))
                for j in range(len(labels)):
                    selected=logits[j][labels[j,1:]!=-100].to(torch.bfloat16).contiguous()
                    assert selected.dtype == torch.bfloat16
                    assert len(selected)==counts[k+j] and torch.isfinite(selected).all()
                    raw=selected.cpu().view(torch.uint16).numpy().tobytes()
                    sink.write(raw)
                    manifest['records'].append(dict(index=k+j,shard=name,offset_bytes=offset,positions=len(selected)))
                    offset+=len(raw)
                del logits, batch, labels
        assert file.stat().st_size==sum(counts[start:end])*32000*2
        manifest['shards'].append(dict(path=name,bytes=file.stat().st_size,sha256=sha(file)))
        put(target/'manifest.json',manifest)
        print(json.dumps({'cache_samples':end,'total':len(features)}),flush=True)
    manifest.update(complete=True,runtime_seconds=time.monotonic()-started,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(),
                    peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    put(target/'manifest.json',manifest)
    put(Path(args.output)/'logit_manifest.json',manifest)


class IndexedDataset:
    def __init__(self,features): self.features=features
    def __len__(self): return len(self.features)
    def __getitem__(self,i): return dict(self.features[i], row_index=i)


class IndexedCollator(Collator):
    def __call__(self,features):
        result=super().__call__(features)
        result['row_index']=torch.tensor([x['row_index'] for x in features])
        return result


class MemoryTrainer(Trainer):
    """Optional microbatching WITHIN the original sampled effective batch8.

    Weight by supervised token count, not naive averaging of microbatch means.
    LR/optimizer step, clipping, and sample order remain at the original batch8.
    """
    microbatch=8

    def training_step(self,model,inputs):
        if self.microbatch==8:return super().training_step(model,inputs)
        assert self.args.gradient_accumulation_steps==1 and self.args.n_gpu==1
        model.train();inputs=self._prepare_inputs(inputs)
        total=(inputs['labels'][:,1:]!=-100).sum()
        reported=torch.zeros((),device=inputs['labels'].device)
        for start in range(0,len(inputs['labels']),self.microbatch):
            part={k:v[start:start+self.microbatch] for k,v in inputs.items()}
            weight=(part['labels'][:,1:]!=-100).sum()/total
            with self.compute_loss_context_manager():loss=self.compute_loss(model,part)*weight
            assert torch.isfinite(loss)
            self.accelerator.backward(loss)
            reported+=loss.detach()
        return reported


class OfflineTrainer(MemoryTrainer):
    def compute_loss(self,model,inputs,return_outputs=False):
        indices=inputs.pop('row_index').cpu().tolist()
        labels=inputs.pop('labels'); mask=labels[:,1:]!=-100
        outputs=model(**inputs,use_cache=False)
        student=outputs.logits[:,:-1][mask]
        arrays=[]
        for i in indices:
            record=self.cache_manifest['records'][i]
            assert record['index']==i
            with (self.cache_root/record['shard']).open('rb') as f:
                f.seek(record['offset_bytes'])
                raw=f.read(record['positions']*32000*2)
            assert len(raw)==record['positions']*32000*2
            arrays.append(torch.from_numpy(np.frombuffer(raw,dtype=np.uint16).copy()).view(torch.bfloat16).reshape(-1,32000))
        teacher=torch.cat(arrays).to(student.device)
        assert teacher.shape==student.shape
        loss,ce,kd=ResponseObjective.apply(student,teacher,labels[:,1:][mask])
        assert torch.isfinite(loss)
        return (loss,outputs) if return_outputs else loss


def train(args,tok,ds,features,stats):
    import bitsandbytes as bnb
    assert bnb.__version__=='0.50.0'
    out=Path(args.output); assert not (out/'final_model').exists()
    model=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.bfloat16,local_files_only=True)
    model.config.use_cache=False
    model=model.cuda()
    assert all(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())==6738415616
    # Same Trainer parameter grouping and Adam defaults as canonical AdamW.
    timing=TimingCallback()
    training=TrainingArguments(output_dir=str(out),num_train_epochs=3,learning_rate=1e-5,
        per_device_train_batch_size=8,gradient_accumulation_steps=1,bf16=True,fp16=False,
        save_strategy='no',logging_steps=10,report_to=[],seed=42,data_seed=42,
        remove_unused_columns=False,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_bnb_8bit',
        lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=0.,max_steps=-1)
    cls=OfflineTrainer if args.mode=='train-kd' else MemoryTrainer
    optimizer=bnb.optim.AdamW8bit(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    trainer=cls(model=model,args=training,train_dataset=IndexedDataset(features) if args.mode=='train-kd' else ds,
                data_collator=IndexedCollator(tok) if args.mode=='train-kd' else Collator(tok),callbacks=[timing],
                optimizers=(optimizer,None))
    trainer.microbatch=args.microbatch
    if args.mode=='train-kd':
        manifest=json.loads((Path(args.cache)/'manifest.json').read_text())
        assert manifest['complete'] and manifest['feature_sha256']==stats['feature_sha256']
        assert manifest['dataset_file_sha256']==stats['dataset_file_sha256']
        for shard in manifest['shards']:
            path=Path(args.cache)/shard['path']
            assert path.stat().st_size==shard['bytes'] and sha(path)==shard['sha256']
        trainer.cache_manifest=manifest; trainer.cache_root=Path(args.cache)
    trainer.create_optimizer()
    assert isinstance(trainer.optimizer,bnb.optim.AdamW8bit)
    put(out/'exact_config.json',dict(training_args=training.to_dict(),data=stats,initialization=args.base,
        revision='f5db02db724555f92da89c216ac04704f23d4590',resume=False,
        trainable_parameters=6738415616,internal_microbatch=args.microbatch,effective_batch=8,
        internal_accumulation=8//args.microbatch,normalization='supervised-token weighted over original batch8',
        optimizer_class=type(trainer.optimizer).__module__+'.'+type(trainer.optimizer).__name__,
        optimizer_version=bnb.__version__,adaptation='8-bit optimizer storage; not canonical FP32 Adam implementation equivalence',
        objective='canonical ResponseObjective T2 CE.5 KD.5' if args.mode=='train-kd' else 'canonical assistant-only CE'))
    started=time.monotonic(); result=trainer.train()
    assert trainer.state.global_step==7500
    assert all(math.isfinite(x['loss']) for x in trainer.state.log_history if 'loss' in x)
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    trainer.save_model(out/'final_model');tok.save_pretrained(out/'final_model')
    put(out/'training_summary.json',dict(steps=trainer.state.global_step,runtime_seconds=time.monotonic()-started,
        step_seconds=timing.steps,metrics=result.metrics,finite_parameters=True,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    put(out/'final_manifest.json',{'files':[dict(path=str(p.relative_to(out/'final_model')),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((out/'final_model').rglob('*')) if p.is_file()]})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['stats','cache','train','train-kd'])
    for key in ['base','dataset','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--teacher');p.add_argument('--cache');p.add_argument('--microbatch',type=int,choices=[2,4,8],default=8);args=p.parse_args()
    set_seed(42);Path(args.output).mkdir(parents=True,exist_ok=True)
    tok,ds,features,counts,stats=prepare(args);put(Path(args.output)/'supervision_stats.json',stats)
    if args.mode=='cache':cache(args,tok,features,counts,stats)
    elif args.mode in ['train','train-kd']:train(args,tok,ds,features,stats)


if __name__=='__main__':main()
