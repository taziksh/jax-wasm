// Driver for the GPT-2 block kernel. 5 inputs -> 1 output [8x64].
//   _mlir_ciface_main(result_sret*, a0[8x64]*, a1[64x192]*, a2[64x64]*,
//                     a3[64x256]*, a4[256x64]*)   (verified from lowered IR)
// Reads arg0.bin..arg4.bin (raw float32, row-major), writes out.bin.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
  float *alloc; float *aligned; int32_t offset; int32_t size[2]; int32_t stride[2];
} MemRef2D;

extern void _mlir_ciface_main(MemRef2D *res, MemRef2D *a0, MemRef2D *a1,
                              MemRef2D *a2, MemRef2D *a3, MemRef2D *a4);

static float *rd(const char *p, long n) {
  FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(2); }
  float *b = malloc(n * sizeof(float));
  if (fread(b, sizeof(float), n, f) != (size_t)n) { fprintf(stderr, "short %s\n", p); exit(2); }
  fclose(f); return b;
}
static MemRef2D mk(float *d, int r, int c) {
  MemRef2D m = {d, d, 0, {r, c}, {c, 1}}; return m;
}

int main(void) {
  MemRef2D a0 = mk(rd("arg0.bin", 8 * 64), 8, 64);
  MemRef2D a1 = mk(rd("arg1.bin", 64 * 192), 64, 192);
  MemRef2D a2 = mk(rd("arg2.bin", 64 * 64), 64, 64);
  MemRef2D a3 = mk(rd("arg3.bin", 64 * 256), 64, 256);
  MemRef2D a4 = mk(rd("arg4.bin", 256 * 64), 256, 64);
  MemRef2D res = {0, 0, 0, {0, 0}, {0, 0}};

  _mlir_ciface_main(&res, &a0, &a1, &a2, &a3, &a4);

  FILE *o = fopen("out.bin", "wb");
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 64; j++) {
      float v = res.aligned[res.offset + i * res.stride[0] + j * res.stride[1]];
      fwrite(&v, sizeof(float), 1, o);
    }
  fclose(o);
  fprintf(stderr, "wrote out.bin\n");
  return 0;
}
