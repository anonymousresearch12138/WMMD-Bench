"""Path-explicit orchestration around the recorded formatter and loss implementations."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainerCallback,TrainingArguments,set_seed
from wmmd_bench.training.hard_labels import SFTDataset,Collator

STEPS=(0,25,50,100,250,500,1000,2000,4000,6000,7500)

def train(a,report):
    if not torch.cuda.is_available():raise RuntimeError('Formal worker requires the recorded Linux/CUDA environment')
    set_seed(42)
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    tok.pad_token=tok.pad_token or tok.eos_token
    if a.condition in ('Bb','XBb'):
        from wmmd_bench.training.transformed import SFTDataset as Data
    else:Data=SFTDataset
    ds=Data(a.dataset,tok,1024)
    teacher=None
    if a.condition=='Bc':
        tt=AutoTokenizer.from_pretrained(a.teacher_tokenizer,local_files_only=True)
        if tok.get_vocab()!=tt.get_vocab():raise ValueError('Teacher/Student tokenizer maps differ')
        x,y=json.loads(tok.backend_tokenizer.to_str()),json.loads(tt.backend_tokenizer.to_str())
        for k in ('model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens'):
            if x[k]!=y[k]:raise ValueError('Teacher/Student tokenizer backend differs: '+k)
        # Preserve historical Bc no-truncation and response-boundary preflight.
        for i,row in enumerate(ds.rows):
            user=row['instruction']+('\n\nInput:\n'+row['input'] if row['input'] else '')
            prompt=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
            full=tok.apply_chat_template([{'role':'user','content':user},{'role':'assistant','content':row['teacher_raw_answer']}],tokenize=False,add_generation_prompt=False)
            ids=tok(full,add_special_tokens=False)['input_ids'];prefix=tok(prompt,add_special_tokens=False)['input_ids']
            if not len(prefix)<len(ids)<=1024 or ids[:len(prefix)]!=prefix or ids!=tt(full,add_special_tokens=False)['input_ids']:raise ValueError('Bc serialization mismatch/truncation')
            if 'source_answer' in row and row['source_answer']!=row['teacher_raw_answer']:raise ValueError('Bc source answer changed')
        if a.method=='ctcc':
            from peft import PeftModel
            teacher_base=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,torch_dtype=torch.bfloat16)
            teacher=PeftModel.from_pretrained(teacher_base,a.teacher_path,local_files_only=True)
        else:teacher=AutoModelForCausalLM.from_pretrained(a.teacher_path,local_files_only=True,torch_dtype=torch.bfloat16)
        teacher=teacher.cuda().eval();teacher.requires_grad_(False);teacher.config.use_cache=False
    model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,torch_dtype=torch.bfloat16)
    model.config.use_cache=False;model.gradient_checkpointing_enable()
    if teacher is not None:
        for key in ('model_type','vocab_size','hidden_size','num_hidden_layers','num_attention_heads'):
            if getattr(teacher.config,key)!=getattr(model.config,key):raise ValueError('Teacher/Student architecture mismatch: '+key)
    class Checkpoints(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step in STEPS:
                dest=a.output_dir/f'checkpoint-{state.global_step}'
                kwargs['model'].save_pretrained(dest);tok.save_pretrained(dest)
    args=TrainingArguments(output_dir=str(a.output_dir),num_train_epochs=3,learning_rate=1e-5,
        per_device_train_batch_size=8,gradient_accumulation_steps=1,bf16=True,fp16=False,
        save_strategy='no' if a.trajectory or a.condition in ('XBa','XBb') or (a.condition=='Bc' and a.method=='passive_shared') else 'epoch',save_total_limit=1,
        logging_steps=1 if a.condition=='Bc' else 10,logging_nan_inf_filter=a.condition!='Bc',report_to=[],seed=42,data_seed=42,
        remove_unused_columns=False,gradient_checkpointing=True,optim='adamw_torch',lr_scheduler_type='cosine',warmup_ratio=.03,weight_decay=0.,max_steps=-1)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'protocol.json').write_text(json.dumps(dict(report,training_arguments=args.to_dict()),indent=2),encoding='utf-8')
    if a.trajectory:
        model.save_pretrained(a.output_dir/'checkpoint-0');tok.save_pretrained(a.output_dir/'checkpoint-0')
    kw=dict(model=model,args=args,train_dataset=ds,data_collator=Collator(tok),callbacks=[Checkpoints()] if a.trajectory else [])
    if teacher is None:trainer=Trainer(**kw)
    else:
        from wmmd_bench.training.online import FormalTrainer
        trainer=FormalTrainer(**kw,teacher=teacher,stage='training')
    trainer.train()
    if trainer.state.global_step!=7500:raise RuntimeError('Wrong optimizer-step count')
    if teacher is not None and (teacher.training or any(p.grad is not None for p in teacher.parameters())):raise RuntimeError('Teacher was not frozen')
    trainer.save_model(a.output_dir/'final_model');tok.save_pretrained(a.output_dir/'final_model');trainer.save_state()
