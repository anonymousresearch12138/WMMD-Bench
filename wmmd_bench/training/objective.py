"""Exact historical response-only T=2, CE=KD=0.5 objective."""
import torch
import torch.nn.functional as F
CHUNK=32

class ResponseObjective(torch.autograd.Function):
    """Exact forward KL/CE with recomputed chunk probabilities in backward.

    Inputs are only selected response positions, already causally shifted.
    No full FP32 batch*sequence*vocabulary copies are retained for backward.
    """
    @staticmethod
    def forward(ctx, student, teacher, labels):
        ctx.save_for_backward(student, teacher, labels)
        ce = student.new_zeros((), dtype=torch.float32)
        kd = ce.clone(); n = labels.numel()
        for k in range(0, n, CHUNK):
            s = student[k:k+CHUNK].float(); t = teacher[k:k+CHUNK].float()
            ce += F.cross_entropy(s, labels[k:k+CHUNK], reduction='sum')
            lt = F.log_softmax(t/2., -1); ls = F.log_softmax(s/2., -1)
            kd += (lt.exp()*(lt-ls)).sum()
        ce = ce/n; kd = kd/n
        ctx.mark_non_differentiable(ce, kd)
        return .5*ce+2.*kd, ce, kd

    @staticmethod
    def backward(ctx, scale, unused_ce, unused_kd):
        s, t, labels = ctx.saved_tensors; n = labels.numel()
        grad = torch.empty_like(s)
        for k in range(0, n, CHUNK):
            sf = s[k:k+CHUNK].float(); tf = t[k:k+CHUNK].float()
            gce = F.softmax(sf, -1)
            gce.scatter_add_(1, labels[k:k+CHUNK, None], -torch.ones((len(sf), 1), device=s.device))
            # .5*T^2 * d KL / d logits = .5*T*(p_student-p_teacher), T=2.
            g = .5*gce + F.softmax(sf/2., -1)-F.softmax(tf/2., -1)
            grad[k:k+CHUNK] = (g*(scale/n)).to(s.dtype)
        return grad, None, None

def objective_test():
    torch.manual_seed(42)
    a = torch.randn(67, 127, requires_grad=True); b = torch.randn_like(a); y = torch.randint(127, (67,))
    z, ce, kd = ResponseObjective.apply(a, b, y); z.backward(); g = a.grad.clone()
    a.grad = None
    refce = F.cross_entropy(a, y)
    lt = F.log_softmax(b/2, -1); refkd = (lt.exp()*(lt-F.log_softmax(a/2, -1))).sum()/len(y)
    ref = .5*refce+2*refkd; ref.backward()
    torch.testing.assert_close(z, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(g, a.grad, rtol=1e-4, atol=1e-7)
    return {'status':'PASS','loss_abs_error':abs(z.item()-ref.item()),'gradient_max_abs_error':(g-a.grad).abs().max().item(),'positions':67,'chunk':CHUNK}
