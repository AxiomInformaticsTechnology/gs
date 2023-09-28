#include "sampler.hpp"

#include <boost/math/distributions/normal.hpp>
using boost::math::normal;
#include <boost/random/normal_distribution.hpp>
using boost::random::normal_distribution;
#include <boost/random/uniform_real_distribution.hpp>
using boost::random::uniform_real_distribution;

double* Sampler::unifrnd(double a, double b, int sz) {
  uniform_real_distribution<> unifrnd(a, b);
  double* r = (double*)malloc(sz * sizeof(double));
  for (int i = 0; i < sz; ++i) {
    r[i] = unifrnd(rng);
  }
  return r;
}

double Sampler::unifrnd(double a, double b) {
  uniform_real_distribution<> unifrnd(a, b);
  return unifrnd(rng);
}

double* Sampler::normrnd(double mu, double sigma, int sz) {
  normal_distribution<> normrnd(mu, sigma);
  double* r = (double*)malloc(sz * sizeof(double));
  for (int i = 0; i < sz; ++i) {
    r[i] = normrnd(rng);
  }
  return r;
}

double Sampler::normrnd(double mu, double sigma) {
  normal_distribution<> normrnd(mu, sigma);
  return normrnd(rng);
}

double Sampler::normcdf(double x, double mu, double sigma) {
  normal n(mu, sigma);
  return cdf(n, x);
}

double Sampler::norminv(double x, double mu, double sigma) {
  normal n(mu, sigma);
  return quantile(n, x);
}

double* Sampler::normrnd(int sz) {
  normal_distribution<> normrnd(MU, SIGMA);
  double* r = (double*)malloc(sz * sizeof(double));
  for (int i = 0; i < sz; ++i) {
    r[i] = normrnd(rng);
  }
  return r;
}

double Sampler::normrnd() {
  normal_distribution<> normrnd(MU, SIGMA);
  return normrnd(rng);
}

double Sampler::normcdf(double x) {
  normal n(MU, SIGMA);
  return cdf(n, x);
}

double Sampler::norminv(double x) {
  normal n(MU, SIGMA);
  return quantile(n, x);
}
