//===- numeric_runner.cpp - codegen lowered LLVM-dialect MLIR to a wasm obj ===//
//
// L4 half of the numeric proof. Takes an ALREADY-lowered LLVM-dialect MLIR module
// (the output of pipeline_runner / pipeline.PIPELINE) and emits a wasm32 object
// file using the in-wasm LLVM WebAssembly backend — exactly eudsl's compileModule
// (WasmCompilerLinkerLoaderCAPI.cpp), which works because in a wasm-compiled LLVM
// the "native" target IS WebAssembly. The emitted kernel.o is then statically
// linked with a tiny C driver (driver.c) by host emcc and run in node, comparing
// to the desktop-JAX oracle (L5).
//
//   numeric_runner <lowered.mlir> <out_object.o>
//
#include <string>

#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdio>

static void errcb(MlirStringRef s, void *) { fwrite(s.data, 1, s.length, stderr); }

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <lowered.mlir> <out.o>\n", argv[0]); return 2; }

  // Read the lowered module text.
  FILE *f = fopen(argv[1], "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::string text(n, '\0'); if (fread(&text[0], 1, n, f) != (size_t)n) return 2; fclose(f);

  MlirContext ctx = mlirContextCreate();
  MlirDialectRegistry reg = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(reg);
  mlirContextAppendDialectRegistry(ctx, reg);
  mlirContextLoadAllAvailableDialects(ctx);
  mlirDialectRegistryDestroy(reg);

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(text.c_str()));
  if (mlirModuleIsNull(module)) { fprintf(stderr, "parse failed\n"); return 1; }

  mlir::Operation *m = unwrap(mlirModuleGetOperation(module));
  mlir::MLIRContext *mctx = m->getContext();
  mlir::registerBuiltinDialectTranslation(*mctx);
  mlir::registerLLVMDialectTranslation(*mctx);

  // Ensure the wasm32 triple is on the module so the WebAssembly backend is picked.
  if (!m->hasAttr("llvm.target_triple"))
    m->setAttr("llvm.target_triple",
               mlir::StringAttr::get(mctx, "wasm32-unknown-emscripten"));

  llvm::InitializeNativeTarget();          // == WebAssembly in a wasm-built LLVM
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::LLVMContext llctx;
  auto llvmModule = mlir::translateModuleToLLVMIR(m, llctx);
  if (!llvmModule) { fprintf(stderr, "translate to LLVM IR failed\n"); return 1; }

  std::string err;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(llvmModule->getTargetTriple(), err);
  if (!target) { fprintf(stderr, "no target: %s\n", err.c_str()); return 1; }

  llvm::TargetOptions to;
  llvm::TargetMachine *tm = target->createTargetMachine(
      llvmModule->getTargetTriple(), "", "", to, llvm::Reloc::Model::PIC_);
  tm->setOptLevel(llvm::CodeGenOptLevel::Default);
  llvmModule->setDataLayout(tm->createDataLayout());

  std::error_code ec;
  llvm::raw_fd_ostream out(argv[2], ec);
  if (ec) { fprintf(stderr, "open out: %s\n", ec.message().c_str()); return 1; }
  llvm::legacy::PassManager pm;
  if (tm->addPassesToEmitFile(pm, out, nullptr, llvm::CodeGenFileType::ObjectFile)) {
    fprintf(stderr, "backend cannot emit object\n"); return 1;
  }
  pm.run(*llvmModule);
  out.close();
  fprintf(stderr, "wrote %s\n", argv[2]);
  return 0;
}
