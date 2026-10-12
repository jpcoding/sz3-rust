/* Exercises sz3c.h; prints an FNV-1a hash of every stream and decompressed buffer, so the output of
 * two libSZ3c builds can be compared. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sz3c.h"

static uint64_t fnv(const void *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) h = (h ^ ((const unsigned char *)p)[i]) * 0x100000001b3ull;
    return h;
}

int main(void) {
    const size_t r1 = 70, r2 = 60, r3 = 30, n = r1 * r2 * r3;
    float *f = malloc(n * sizeof(float));
    double *d = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        size_t x = i % r1, y = i / r1 % r2, z = i / r1 / r2;
        d[i] = sin(0.1 * x) * cos(0.07 * y) + 0.3 * sin(0.05 * z * x) + 1e-3 * ((i * 2654435761u) % 1000) / 1000.0;
        f[i] = (float)d[i];
    }
    int modes[] = {ABS, REL, ABS_AND_REL, ABS_OR_REL};
    int fails = 0;
    for (int m = 0; m < 4; m++) {
        for (int t = 0; t < 2; t++) {
            void *src = t ? (void *)d : (void *)f;
            size_t es = t ? sizeof(double) : sizeof(float);
            size_t cmp_size = 0;
            unsigned char *cmp = SZ_compress_args(t ? SZ_DOUBLE : SZ_FLOAT, src, &cmp_size, modes[m], 1e-3, 1e-4, 0, 0,
                                                  0, r3, r2, r1);
            if (!cmp) { printf("mode %d type %d: compress failed\n", modes[m], t); fails++; continue; }
            void *dec = SZ_decompress(t ? SZ_DOUBLE : SZ_FLOAT, cmp, cmp_size, 0, 0, r3, r2, r1);
            if (!dec) { printf("mode %d type %d: decompress failed\n", modes[m], t); fails++; free_buf(cmp); continue; }
            double maxerr = 0;
            for (size_t i = 0; i < n; i++) {
                double e = t ? fabs(((double *)dec)[i] - d[i]) : fabs(((float *)dec)[i] - f[i]);
                if (e > maxerr) maxerr = e;
            }
            printf("mode %d type %d: %zu bytes, stream %016llx, data %016llx, max error %.3g\n", modes[m], t, cmp_size,
                   (unsigned long long)fnv(cmp, cmp_size), (unsigned long long)fnv(dec, n * es), maxerr);
            free_buf(cmp);
            free_buf(dec);
        }
    }
    /* errors come back as NULL */
    size_t s;
    if (SZ_compress_args(SZ_FLOAT, f, &s, 99, 1e-3, 0, 0, 0, 0, r3, r2, r1) != NULL) fails++;
    if (SZ_decompress(SZ_FLOAT, (unsigned char *)f, 100, 0, 0, r3, r2, r1) != NULL) fails++;
    free(f);
    free(d);
    printf("%s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
