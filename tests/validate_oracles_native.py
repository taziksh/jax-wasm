#!/usr/bin/env python3
"""Independently validate the numeric oracles with IREE's native CPU backend.

Compiles the SAME StableHLO fixtures the wasm path lowers, runs them on llvm-cpu,
and compares to the desktop-JAX oracles (fixtures/*_io.npz). This is a native
stand-in for L5/L6: it confirms the lowering's numerics are correct independent of
the wasm toolchain, and that the oracles are trustworthy.

Observed: IREE-CPU vs JAX = 3.6e-7 (smoke), 1.2e-6 (full GPT-2 block).

    pip install iree-base-compiler iree-base-runtime
    python tests/validate_oracles_native.py
"""
import re, os, numpy as np
import iree.compiler as ic, iree.runtime as irt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FX = os.path.join(HERE, "fixtures")


def run(mlir_path, args):
    text = open(mlir_path).read()
    hint = re.search(r"module @(\w+)", text).group(1)
    vmfb = ic.compile_str(text, target_backends=["llvm-cpu"], input_type="stablehlo")
    ctx = irt.SystemContext(config=irt.Config("local-task"))
    ctx.add_vm_module(irt.VmModule.copy_buffer(ctx.instance, vmfb))
    return np.asarray(getattr(ctx.modules, hint).main(*args))


def check(name, mlir, npz, in_keys, tol=1e-4):
    d = np.load(os.path.join(FX, npz))
    got = run(os.path.join(FX, mlir), [d[k] for k in in_keys])
    ref = d["y" if "y" in d else "output"]
    err = float(np.max(np.abs(got - ref)))
    ok = np.allclose(got, ref, rtol=tol, atol=tol)
    print(f"{name}: max|iree-jax|={err:.2e}  allclose(rtol=atol={tol})={ok}")
    return ok


if __name__ == "__main__":
    a = check("smoke", "smoke.stablehlo.mlir", "smoke_io.npz", ["x", "w"])
    b = check("gpt2 ", "gpt2_block.stablehlo.mlir", "gpt2_block_io.npz",
              [f"arg{i}" for i in range(5)])
    raise SystemExit(0 if (a and b) else 1)
