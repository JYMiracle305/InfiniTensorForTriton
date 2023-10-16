#pragma once
namespace infini {
namespace opTimer {
double getPerfConvXdnn(int n, int c, int h, int w, int f, int r, int s,
                       int padh, int padw, int strideh, int stridew,
                       int dilationh, int dilationw, int group,
                       const char *name);
double getPerfMatmulXdnn(int b, int m, int n, int k, const char *name);
} // namespace opTimer
} // namespace infini
