"""Check that the fast Gated DeltaNet kernel (flash-linear-attention on Triton) gives the same results as the
PyTorch reference used by transformers, and measure the speed-up. Run in .venv-llm on the GPU.

Exit code 0 = fast kernel active and numerically OK; 1 = do not use it (install_fast_kernels.ps1 then removes it).
"""
import sys
import time

import torch

from transformers.models.qwen3_5 import modeling_qwen3_5 as M


def main():
    assert torch.cuda.is_available(), "needs the GPU"
    try:
        import fla  # noqa: F401
    except Exception as e:
        print("fla import failed:", repr(e)); return 1
    fast = M.torch_chunk_gated_delta_rule          # resolves to fla when importable
    ref = getattr(fast, "__wrapped__", None)       # original PyTorch implementation
    if ref is None:
        print("cannot reach the PyTorch reference implementation"); return 1

    torch.manual_seed(0)
    dev = "cuda"
    # Qwen3.5-9B Gated DeltaNet shapes after head repeat: 32 value heads, key/value dim 128
    B, T, H, D = 4, 256, 32, 128
    def inputs():
        q = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        g = -(torch.rand(B, T, H, device=dev) * 0.5 + 0.01)
        beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16)
        return q, k, v, g, beta
    q, k, v, g, beta = inputs()
    w = torch.randn(B, T, H, D, device=dev)

    def run(fn):
        for t in (q, k, v):
            t.grad = None
        out, _ = fn(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        (out.float() * w).sum().backward()
        return out.detach().float(), q.grad.detach().float().clone()

    try:
        o_f, gq_f = run(fast)
    except Exception as e:
        print("fast kernel failed to run:", repr(e)); return 1
    o_r, gq_r = run(ref)
    rel_o = ((o_f - o_r).norm() / o_r.norm()).item()
    rel_g = ((gq_f - gq_r).norm() / gq_r.norm()).item()
    print(f"relative difference vs PyTorch reference: output {rel_o:.4f}, gradient {rel_g:.4f}")

    def bench(fn, n=10):
        run(fn); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            run(fn)
        torch.cuda.synchronize(); return (time.time() - t0) / n * 1000
    tf, tr = bench(fast), bench(ref)
    print(f"forward+backward per call: fast {tf:.1f} ms, reference {tr:.1f} ms, speed-up x{tr / tf:.1f}")
    ok = rel_o < 0.02 and rel_g < 0.05
    if ok and tr / tf < 1.2:
        print("RESULT: no speed-up, so transformers is probably not picking up the fla kernel"); return 1
    print("RESULT:", "OK, use the fast kernel" if ok else "MISMATCH, do not use the fast kernel")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
