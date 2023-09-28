#include "sampler.hpp"

Estimates Sampler::uno2p(
  Input input,
  Params params
) {
  int* y = input.y;
  double* a = input.a;
  double* g = input.g;
  double* th = input.th;
  int personCount = params.personCount;
  int itemCount = params.itemCount;
  int iterations = params.iterations;
  int burnin = params.burnin;
  int batches = params.batches;
  double mu = params.mu;
  double var = params.var;

  int bsize = (iterations - burnin) / batches;

  double* z = (double*)malloc(personCount * itemCount * sizeof(double));

  double* av = (double*)malloc(itemCount * 3 * sizeof(double));
  double* gv = (double*)malloc(itemCount * 3 * sizeof(double));
  double* thv = (double*)malloc(personCount * 3 * sizeof(double));

  for (int iteration = 0; iteration < iterations; ++iteration) {
    updateZ(y, th, a, g, z, personCount, itemCount);
    updateTH(th, a, g, z, mu, var, personCount, itemCount);
    updateAG(th, a, g, z, personCount, itemCount);
    if (iteration >= burnin) {
      trackStats(thv, av, gv, th, a, g, personCount, itemCount, iteration, burnin, bsize);
    }
  }

  free(z);

  Estimates estimates = calcEstimates(thv, av, gv, batches, bsize, personCount, itemCount);

  free(av);
  free(gv);
  free(thv);

  return estimates;
}
