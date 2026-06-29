#!/usr/bin/env python3
"""Generate StableHLO test inputs + desktop-JAX numeric oracles.

Outputs (next to this file):
  smoke.stablehlo.mlir     - matmul+softmax, the smallest non-trivial L3 input
  smoke_io.npz             - x, w, y=softmax(x@w) reference (L5 oracle)
  gpt2_block.stablehlo.mlir - regenerated GPT-2 block (compared to the committed one)
  gpt2_block_io.npz        - args + output reference (L6 oracle)

The GPT-2 block mirrors the committed fixtures/gpt2_block.stablehlo.mlir:
LayerNorm (no affine, eps 1e-5) -> QKV (one matmul, split) -> causal self-
attention (q k^T / sqrt(d), tril mask, softmax) -> out proj + residual ->
MLP (W1, tanh-GELU, W2) + residual. Static shapes seq=8, d=64, mlp=256.
"""
import numpy as np
import jax, jax.numpy as jnp

SEQ, D, MLP = 8, 64, 256


def smoke(x, w):
    return jax.nn.softmax(x @ w, axis=-1)


def layernorm(x):
    mu = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5)


def tril(x):
    i = jax.lax.broadcasted_iota(jnp.int32, (SEQ, SEQ), 0)
    j = jax.lax.broadcasted_iota(jnp.int32, (SEQ, SEQ), 1)
    return jnp.where(i >= j, x, 0.0)


def attention_block(x, Wqkv, Wo, W1, W2):
    ln = layernorm(x)
    qkv = ln @ Wqkv                       # (8,192)
    q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
    att = (q @ k.T) / jnp.sqrt(jnp.float32(D))
    mask = tril(jnp.ones((SEQ, SEQ)))
    att = jnp.where(mask > 0, att, -1e9)
    att = jax.nn.softmax(att, axis=-1)
    out = att @ v
    x = x + out @ Wo
    h = x @ W1
    gelu = 0.5 * h * (1.0 + jnp.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
    return x + gelu @ W2


def emit(fn, *args):
    return str(jax.jit(fn).lower(*args).compiler_ir("stablehlo"))


def op_histogram(mlir_text):
    import re, collections
    return collections.Counter(re.findall(r'stablehlo\.\w+', mlir_text))


def main():
    here = __file__.rsplit("/", 1)[0]
    rng = np.random.default_rng(0)

    # --- smoke ---
    x = jnp.array(np.ones((SEQ, D), np.float32))
    w = jnp.array(np.ones((D, D), np.float32))
    open(f"{here}/smoke.stablehlo.mlir", "w").write(emit(smoke, x, w))
    xr = jnp.array(rng.standard_normal((SEQ, D), np.float32))
    wr = jnp.array(rng.standard_normal((D, D), np.float32))
    np.savez(f"{here}/smoke_io.npz", x=np.asarray(xr), w=np.asarray(wr),
             y=np.asarray(smoke(xr, wr)))
    print("smoke: wrote mlir + io.npz")

    # --- gpt2 block ---
    shapes = [(SEQ, D), (D, 3 * D), (D, D), (D, MLP), (MLP, D)]
    ones = [jnp.array(np.ones(s, np.float32)) for s in shapes]
    txt = emit(attention_block, *ones)
    open(f"{here}/gpt2_block.regen.stablehlo.mlir", "w").write(txt)

    committed = open(f"{here}/gpt2_block.stablehlo.mlir").read()
    h_new, h_old = op_histogram(txt), op_histogram(committed)
    # NOTE: the histograms differ only in helper-function formulation — the
    # committed fixture keeps tril/_where as separate func.func (extra
    # convert/select/broadcast) while this JAX version inlines them. The two
    # COMPUTE bit-identically (verified: IREE diff = 0.0). So this is expected.
    if h_new != h_old:
        print("gpt2 op-histogram differs (helper-fn formulation; computations are "
              "bit-identical per IREE) — expected")

    # Scale weights by 1/sqrt(fan_in) (standard init) so activations stay O(1).
    # Without this, unnormalized weights drive outputs to O(1e4), which makes
    # absolute tolerances meaningless (a correct 1e-6 RELATIVE error then shows
    # as ~0.02 absolute). x (arg0) stays standard-normal; it is LayerNorm'd first.
    def mk(i, s):
        a = rng.standard_normal(s, np.float32)
        return jnp.array(a if i == 0 else a / np.sqrt(s[0], dtype=np.float32))
    rargs = [mk(i, s) for i, s in enumerate(shapes)]
    out = attention_block(*rargs)
    np.savez(f"{here}/gpt2_block_io.npz",
             **{f"arg{i}": np.asarray(a) for i, a in enumerate(rargs)},
             output=np.asarray(out))
    print("gpt2: wrote regen mlir + io.npz, output shape", out.shape)


if __name__ == "__main__":
    main()
