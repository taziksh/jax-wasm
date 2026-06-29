//===- pipeline_runner.c - in-wasm StableHLO lowering harness -------------===//
//
// Standalone MLIR C-API program. Compiled to wasm32 (emcc) and run in node, it:
//   1. registers all upstream dialects + the StableHLO/CHLO dialects,
//   2. registers all upstream passes + mlirRegisterAllStablehloPasses()  (L2),
//   3. parses an MLIR module from argv[1],
//   4. runs the pass pipeline string argv[2] on it,                       (L3)
//   5. prints the lowered module to stdout.
//
// This is the tightest proof that StableHLO cross-compiled to wasm, linked
// against eudsl-style LLVM, and that its legalize-to-linalg passes register and
// run *in wasm* — with no pyodide / python in the loop. Build it with
// scripts/05_build_pipeline_runner.sh and check the output for 0 leftover
// "stablehlo." / "linalg." / "unrealized_conversion_cast".
//
// Run (compiled with -sNODERAWFS=1 so fopen sees real host files):
//   node pipeline_runner.js <input.mlir> "<pipeline>"
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Support.h"

#include "stablehlo/integrations/c/ChloDialect.h"
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "stablehlo/integrations/c/StablehloPasses.h"

static void printToStdout(MlirStringRef s, void *userData) {
  (void)userData;
  fwrite(s.data, 1, s.length, stdout);
}

static void printToStderr(MlirStringRef s, void *userData) {
  (void)userData;
  fwrite(s.data, 1, s.length, stderr);
}

static char *readFile(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *buf = (char *)malloc(n + 1);
  if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(2); }
  buf[n] = 0; fclose(f);
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <input.mlir> <pipeline>\n", argv[0]);
    return 2;
  }
  char *input = readFile(argv[1]);
  const char *pipeline = argv[2];

  MlirContext ctx = mlirContextCreate();

  // --- dialects: upstream + stablehlo + chlo --------------------------------
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirDialectHandleInsertDialect(mlirGetDialectHandle__stablehlo__(), registry);
  mlirDialectHandleInsertDialect(mlirGetDialectHandle__chlo__(), registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirContextLoadAllAvailableDialects(ctx);
  mlirDialectRegistryDestroy(registry);

  // --- passes: upstream + stablehlo (L2) ------------------------------------
  mlirRegisterAllPasses();
  mlirRegisterAllStablehloPasses();

  // --- parse the input module ----------------------------------------------
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(input));
  if (mlirModuleIsNull(module)) {
    fprintf(stderr, "FAIL: could not parse input module\n");
    return 1;
  }

  // --- build + run the pipeline (L3) ---------------------------------------
  MlirPassManager pm = mlirPassManagerCreate(ctx);
  MlirOpPassManager opm = mlirPassManagerGetAsOpPassManager(pm);
  MlirLogicalResult parsed = mlirParsePassPipeline(
      opm, mlirStringRefCreateFromCString(pipeline), printToStderr, NULL);
  if (mlirLogicalResultIsFailure(parsed)) {
    fprintf(stderr, "FAIL: could not parse pipeline (a pass is unregistered?)\n");
    return 1;
  }

  MlirLogicalResult ran =
      mlirPassManagerRunOnOp(pm, mlirModuleGetOperation(module));
  if (mlirLogicalResultIsFailure(ran)) {
    fprintf(stderr, "FAIL: pipeline run failed\n");
    return 1;
  }

  mlirOperationPrint(mlirModuleGetOperation(module), printToStdout, NULL);

  mlirPassManagerDestroy(pm);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  free(input);
  return 0;
}
