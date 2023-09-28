#include "sampler.hpp"

double Sampler::mean(double* v, int sz) {
  double sum = 0;
  for (int i = 0; i < sz; ++i) {
    sum += v[i];
  }
  return sum / (double)sz;
}

double Sampler::correlation(double* v1, double* v2, int sz) {
  double meanV1 = mean(v1, sz);
  double meanV2 = mean(v2, sz);
  double sumV1V2 = 0;
  double sumV1S = 0;
  double sumV2S = 0;
  for (int i = 0; i < sz; ++i) {
    double mv1 = v1[i] - meanV1;
    double mv2 = v2[i] - meanV2;
    sumV1V2 += mv1 * mv2;
    sumV1S += mv1 * mv1;
    sumV2S += mv2 * mv2;
  }
  return sumV1V2 / sqrt(sumV1S * sumV2S);
}
