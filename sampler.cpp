#include "sampler.hpp"

Sampler::Sampler(int seed) {
  Sampler::seed = seed;
  mt19937_64 rng(seed);
}

Sampler::~Sampler() {

}

Scores Sampler::generate_scores(int personCount, int itemCount) {
  struct Scores scores;
  scores.y = (int*)malloc(personCount * itemCount * sizeof(int));
  scores.as = unifrnd(0, 1, itemCount);
  scores.gs = unifrnd(0, 1, itemCount);
  scores.ths = normrnd(personCount);
  // scale ul - ll, add ll where ll = -1 and ul = 1
  for (int i = 0; i < itemCount; ++i) {
    scores.gs[i] = (scores.gs[i] * 2) - 1;
  }
  for (int p = 0; p < personCount; ++p) {
    for (int i = 0; i < itemCount; ++i) {
      scores.y[p * itemCount + i] = unifrnd(0, 1) < normcdf(scores.gs[i] - scores.as[i] * scores.ths[p]) ? 0 : 1;
    }
  }
  return scores;
}

double* Sampler::generate_array(int size, double value) {
  double* array = (double*)malloc(size * sizeof(double));
  for (int i = 0; i < size; ++i) {
    array[i] = value;
  }
  return array;
}

double* Sampler::generate_alpha(int itemCount) {
  return generate_array(itemCount, 2);
}

double* Sampler::generate_gamma(int itemCount) {
  return generate_array(itemCount, 0);
}

double* Sampler::generate_theta(int personCount) {
  return generate_array(personCount, 0);
}

double* Sampler::generate_gamma_from_scores(int* scores, int personCount, int itemCount) {
  double* gamma = (double*)malloc(itemCount * sizeof(double));
  for (int i = 0; i < itemCount; ++i) {
    double phat = 0.0;
    for (int p = 0; p < personCount; ++p) {
      phat += scores[p * itemCount + i];
    }
    gamma[i] = -norminv(phat / personCount, 0, 1) * sqrt(5);
  }
  return gamma;
}

void Sampler::updateZ(int* y, double* th, double* a, double* g, double* z, int personCount, int itemCount) {
  double lp, bb, u, tmp;
  for (int p = 0; p < personCount; ++p) {
    for (int i = 0; i < itemCount; ++i) {
      lp = th[p] * a[i] - g[i];
      bb = normcdf(0 - lp);
      u = unifrnd(0, 1);
      tmp = y[p * itemCount + i] == 0 ? bb * u : ((1 - bb) * u) + bb;
      z[p * itemCount + i] = norminv(tmp) + lp;
    }
  }
}

void Sampler::updateTH(double* th, double* a, double* g, double* z, double mu, double var, int personCount, int itemCount) {
  double mn, pmean;
  double pvar = 1 / (accumulate(a, a + itemCount, 0.0, sum_of_squares()) + (1 / var));
  double v = sqrt(pvar);
  for (int p = 0; p < personCount; ++p) {
    mn = 0;
    for (int i = 0; i < itemCount; ++i) {
      mn += a[i] * (z[p * itemCount + i] + g[i]);
    }
    pmean = (mn + mu / var) * pvar;
    th[p] = normrnd() * v + pmean;
  }
}

void Sampler::updateTH(double* lxx, double* thv, double* th, double* a, double* g, double* z, double mu, double var, int personCount, int itemCount, bool track, int burnin, int count) {
  double mn, pmean;
  double pvar = 1 / (accumulate(a, a + itemCount, 0.0, sum_of_squares()) + (1 / var));
  double v = sqrt(pvar);
  lxx[0] = lxx[1] = 0;
  for (int p = 0; p < personCount; ++p) {
    mn = 0;
    for (int i = 0; i < itemCount; ++i) {
      mn += a[i] * (z[p * itemCount + i] + g[i]);
    }
    pmean = (mn + mu / var) * pvar;
    th[p] = normrnd() * v + pmean;

    if (track) {
      thv[p * burnin + count] = th[p];
    }

    lxx[0] += th[p] * th[p];
    lxx[1] -= th[p];
  }
}

void Sampler::calcXZ(double* lxz, double* th, double* z, int personCount, int itemCount) {
  for (int i = 0; i < itemCount; ++i) {
    lxz[i * 2 + 0] = lxz[i * 2 + 1] = 0;
    for (int p = 0; p < personCount; ++p) {
      lxz[i * 2 + 0] += th[p] * z[p * itemCount + i];
      lxz[i * 2 + 1] -= z[p * itemCount + i];
    }
  }
}

void Sampler::calcAMAT(double* xx, double* ix, double* amat, int personCount) {
  double dt;

  xx[2] = xx[1];
  xx[3] = (double)personCount;

  dt = xx[0] * xx[3] - xx[1] * xx[2];

  ix[0] = xx[3] / dt;
  ix[1] = -xx[1] / dt;
  ix[2] = -xx[2] / dt;
  ix[3] = xx[0] / dt;

  amat[0] = sqrt(ix[0]);
  amat[1] = 0;
  amat[2] = ix[1] / amat[0];
  amat[3] = sqrt(ix[3] - amat[2] * amat[2]);
}

void Sampler::updateAG(double* th, double* a, double* g, double* z, int personCount, int itemCount) {
  double dt;

  double* xx = (double*)malloc(4 * sizeof(double));
  double* ix = (double*)malloc(4 * sizeof(double));
  double* amat = (double*)malloc(4 * sizeof(double));

  double* xz = (double*)malloc(2 * sizeof(double));
  double* bz = (double*)malloc(2 * sizeof(double));
  double* bi = (double*)malloc(2 * sizeof(double));

  xx[0] = xx[1] = xx[2] = xx[3] = 0;
  for (int p = 0; p < personCount; ++p) {
    xx[0] += th[p] * th[p];
    xx[1] -= th[p];
  }
  xx[2] = xx[1];
  xx[3] = (double)personCount;

  dt = xx[0] * xx[3] - xx[1] * xx[2];

  ix[0] = xx[3] / dt;
  ix[1] = -xx[1] / dt;
  ix[2] = -xx[2] / dt;
  ix[3] = xx[0] / dt;

  amat[0] = sqrt(ix[0]);
  amat[1] = 0;
  amat[2] = ix[1] / amat[0];
  amat[3] = sqrt(ix[3] - amat[2] * amat[2]);

  // update alpha and gamma
  for (int i = 0; i < itemCount; ++i) {
    xz[0] = xz[1] = 0;
    for (int p = 0; p < personCount; ++p) {
      xz[0] += th[p] * z[p * itemCount + i];
      xz[1] -= z[p * itemCount + i];
    }

    bz[0] = (ix[0] * xz[0]) + (ix[1] * xz[1]);
    bz[1] = (ix[2] * xz[0]) + (ix[3] * xz[1]);

    bi = normrnd(2);
    a[i] = ((bi[0] * amat[0]) + (bi[1] * amat[2])) + bz[0];
    g[i] = ((bi[0] * amat[1]) + (bi[1] * amat[3])) + bz[1];
  }

  free(xx);
  free(ix);
  free(amat);
  free(xz);
  free(bz);
  free(bi);
}

void Sampler::updateAG(double* av, double* gv, double* a, double* g, double* ix, double* xz, double* amat, int itemCount, bool track, int burnin, int count) {
  double* bz = (double*)malloc(2 * sizeof(double));
  double* bi = (double*)malloc(2 * sizeof(double));
  for (int i = 0; i < itemCount; ++i) {
    bz[0] = (ix[0] * xz[i * 2 + 0]) + (ix[1] * xz[i * 2 + 1]);
    bz[1] = (ix[2] * xz[i * 2 + 0]) + (ix[3] * xz[i * 2 + 1]);

    bi = normrnd(2);
    a[i] = ((bi[0] * amat[0]) + (bi[1] * amat[2])) + bz[0];
    g[i] = ((bi[0] * amat[1]) + (bi[1] * amat[3])) + bz[1];

    if (track) {
      av[i * burnin + count] = a[i];
      gv[i * burnin + count] = g[i];
    }
  }
  free(bz);
  free(bi);
}

void Sampler::trackStats(double* thv, double* av, double* gv, double* th, double* a, double* g, int personCount, int itemCount, int iteration, int burnin, int bsize) {
  if ((iteration - burnin) % bsize == 0) {
    if (iteration == burnin) {
      for (int p = 0; p < personCount; ++p) {
        thv[p * 3 + 1] = 0;
        thv[p * 3 + 2] = 0;
      }
      for (int i = 0; i < itemCount; ++i) {
        av[i * 3 + 1] = 0;
        gv[i * 3 + 1] = 0;
        av[i * 3 + 2] = 0;
        gv[i * 3 + 2] = 0;
      }
    }
    else {
      for (int p = 0; p < personCount; ++p) {
        double m = thv[p * 3 + 0] / bsize;
        thv[p * 3 + 1] += m;
        thv[p * 3 + 2] += m * m;
      }
      for (int i = 0; i < itemCount; ++i) {
        double m1 = av[i * 3 + 0] / bsize;
        double m2 = gv[i * 3 + 0] / bsize;
        av[i * 3 + 1] += m1;
        gv[i * 3 + 1] += m2;
        av[i * 3 + 2] += m1 * m1;
        gv[i * 3 + 2] += m2 * m2;
      }
    }
    for (int p = 0; p < personCount; ++p) {
      thv[p * 3 + 0] = th[p];
    }
    for (int i = 0; i < itemCount; ++i) {
      av[i * 3 + 0] = a[i];
      gv[i * 3 + 0] = g[i];
    }
  }
  else {
    for (int p = 0; p < personCount; ++p) {
      thv[p * 3 + 0] += th[p];
    }
    for (int i = 0; i < itemCount; ++i) {
      av[i * 3 + 0] += a[i];
      gv[i * 3 + 0] += g[i];
    }
  }
}

Estimates Sampler::calcEstimates(double* thv, double* av, double* gv, int batches, int bsize, int personCount, int itemCount) {
  struct Estimates estimates;
  estimates.items = (double*)malloc(itemCount * 4 * sizeof(double));
  estimates.persons = (double*)malloc(personCount * 2 * sizeof(double));

  double srb = sqrt(batches);
  int bmi = batches - 1;

  for (int i = 0; i < itemCount; ++i) {
    double m1 = (double) av[i * 3 + 0] / bsize;
    double m2 = (double) gv[i * 3 + 0] / bsize;
    av[i * 3 + 1] += m1;
    gv[i * 3 + 1] += m2;
    av[i * 3 + 2] += m1 * m1;
    gv[i * 3 + 2] += m2 * m2;

    estimates.items[i * 4 + 0] = (double) av[i * 3 + 1] / batches;
    estimates.items[i * 4 + 1] = sqrt((double) (av[i * 3 + 2] - (av[i * 3 + 1] * (double) av[i * 3 + 1] / batches)) / bmi) / srb;
    estimates.items[i * 4 + 2] = (double) gv[i * 3 + 1] / batches;
    estimates.items[i * 4 + 3] = sqrt((double) (gv[i * 3 + 2] - (gv[i * 3 + 1] * (double) gv[i * 3 + 1] / batches)) / bmi) / srb;
  }
  for (int p = 0; p < personCount; ++p) {
    double m = (double) thv[p * 3 + 0] / bsize;
    thv[p * 3 + 1] += m;
    thv[p * 3 + 2] += m * m;

    estimates.persons[p * 2 + 0] = (double) thv[p * 3 + 1] / batches;
    estimates.persons[p * 2 + 1] = sqrt((double) (thv[p * 3 + 2] - (thv[p * 3 + 1] * (double) thv[p * 3 + 1] / batches)) / bmi) / srb;
  }
  return estimates;
}
