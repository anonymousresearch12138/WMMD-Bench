import torch
from transformers import Trainer
from wmmd_bench.training.objective import ResponseObjective

class KDTrainer(Trainer):

    def __init__(self, *args, teacher, stage, **kw):
        self.teacher = teacher
        self.stage = stage
        self.mem = []
        super().__init__(*args, **kw)

    def memory(self, point):
        torch.cuda.synchronize()
        self.mem.append({'step': self.state.global_step + 1, 'point': point, 'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()})

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs['labels']
        mask = labels[:, 1:] != -100
        self.memory('before_forward')
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], use_cache=False).logits
            t = teacher_logits[:, :-1][mask].to(torch.bfloat16)
            del teacher_logits
        self.memory('after_teacher_forward')
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], use_cache=False)
        self.memory('after_student_forward')
        s = outputs.logits[:, :-1][mask]
        assert s.shape == t.shape
        loss, ce, kd = ResponseObjective.apply(s, t, labels[:, 1:][mask])
        self.memory('after_KL')
        assert torch.isfinite(loss).item()
        self.last = {'ce_loss': ce.item(), 'kd_loss': kd.item(), 'scaled_kd_loss': 4 * kd.item(), 'total_loss': loss.item(), 'input_tokens': inputs['attention_mask'].sum().item(), 'response_tokens': mask.sum().item(), 'samples': len(labels), 'nan_inf': False}
        return (loss, outputs) if return_outputs else loss
