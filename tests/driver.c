// Driver for the smoke kernel (softmax(x@w)): x[8x64], w[64x64] -> y[8x64].
// Statically linked by host emcc with kernel.o (emitted by numeric_runner from
// the lowered LLVM-dialect MLIR). Reads x.bin/w.bin (raw float32), calls the
// MLIR c-interface, writes y.bin. ABI verified from the lowered IR:
//   _mlir_ciface_main(result_sret*, x_desc*, w_desc*)
//   descriptor = {float* alloc; float* aligned; int32 offset; int32 size[2];
//                 int32 stride[2];}  (index-bitwidth=32 -> i32 fields on wasm32)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
  float *alloc;
  float *aligned;
  int32_t offset;
  int32_t size[2];
  int32_t stride[2];
} MemRef2D;

extern void _mlir_ciface_main(MemRef2D *result, MemRef2D *x, MemRef2D *w);

static float *readbin(const char *path, long count) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "open %s\n", path); exit(2); }
  float *p = (float *)malloc(count * sizeof(float));
  if (fread(p, sizeof(float), count, f) != (size_t)count) { fprintf(stderr, "short read %s\n", path); exit(2); }
  fclose(f);
  return p;
}

int main(void) {
  float *xd = readbin("x.bin", 8 * 64);
  float *wd = readbin("w.bin", 64 * 64);

  MemRef2D x = {xd, xd, 0, {8, 64}, {64, 1}};
  MemRef2D w = {wd, wd, 0, {64, 64}, {64, 1}};
  MemRef2D res = {0, 0, 0, {0, 0}, {0, 0}};   // function fills (sret, internal alloc)

  _mlir_ciface_main(&res, &x, &w);

  // result is [8x64]; read from res.aligned + res.offset, honoring strides.
  FILE *o = fopen("y.bin", "wb");
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 64; j++) {
      float v = res.aligned[res.offset + i * res.stride[0] + j * res.stride[1]];
      fwrite(&v, sizeof(float), 1, o);
    }
  fclose(o);
  fprintf(stderr, "wrote y.bin\n");
  return 0;
}
