#include "sampler.hpp"

Estimates Sampler::masterUno2p(
  Input input,
  Params params
) {
  int* y = input.y;
  double* a = input.a;
  double* g = input.g;
  double* th = input.th;
  int size = params.size;
  int personCount = params.personCount;
  int itemCount = params.itemCount;
  int iterations = params.iterations;
  int burnin = params.burnin;
  int batches = params.batches;
  double mu = params.mu;
  double var = params.var;

  int bsize = (iterations - burnin) / batches;

  // decompose
  int* personPartitions = (int*)malloc(size * sizeof(int));
  int* pi = (int*)malloc(size * sizeof(int));
  int* pid = (int*)malloc(size * sizeof(int));
  int* pxi = (int*)malloc(size * sizeof(int));
  int* pxid = (int*)malloc(size * sizeof(int));
  int* pxb = (int*)malloc(size * sizeof(int));
  int* pxbd = (int*)malloc(size * sizeof(int));

  int portion = personCount / size;
  int remainder = personCount % size;

  int stride = 0;

  for (int r = 0; r < size; r++) {
    personPartitions[r] = portion + ((remainder > 0) ? 1 : 0);

    pi[r] = personPartitions[r];
    pid[r] = stride;

    pxi[r] = personPartitions[r] * itemCount;
    pxid[r] = stride * itemCount;

    pxb[r] = personPartitions[r] * 2;
    pxbd[r] = stride * 2;

    stride += personPartitions[r];

    if (remainder > 0) {
      remainder--;
    }
  }

  int personPartitionCount = personPartitions[ROOT];

  double* z = (double*)malloc(personPartitionCount * itemCount * sizeof(double));

  double* xx = (double*)malloc(4 * sizeof(double));
  double* ix = (double*)malloc(4 * sizeof(double));
  double* amat = (double*)malloc(4 * sizeof(double));

  double* xz = (double*)malloc(itemCount * 2 * sizeof(double));

  double* bz = (double*)malloc(2 * sizeof(double));
  double* bi = (double*)malloc(2 * sizeof(double));

  double* av = (double*)malloc(itemCount * burnin * sizeof(double));
  double* gv = (double*)malloc(itemCount * burnin * sizeof(double));
  double* thv = (double*)malloc(personPartitionCount * burnin * sizeof(double));

  int count = 0;
  bool track = false;

  for (int r = 1; r < size; ++r) {
    MPI_Send(&personPartitions[r], 1, MPI_UNSIGNED, r, TAG, MPI_COMM_WORLD);
  }

  MPI_Scatterv(y, pxi, pxid, MPI_UNSIGNED, MPI_IN_PLACE, personCount * itemCount, MPI_UNSIGNED, ROOT, MPI_COMM_WORLD);

  MPI_Scatterv(th, pi, pid, MPI_DOUBLE, MPI_IN_PLACE, personCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  xx[3] = (double)personCount;
  amat[1] = 0;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    track = iteration >= burnin;

    MPI_Bcast(a, itemCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(g, itemCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    updateZ(y, th, a, g, z, personPartitionCount, itemCount);

    updateTH(xx, thv, th, a, g, z, mu, var, personPartitionCount, itemCount, track, burnin, count);

    MPI_Reduce(MPI_IN_PLACE, xx, 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    calcAMAT(xx, ix, amat, personCount);

    calcXZ(xz, th, z, personPartitionCount, itemCount);

    MPI_Reduce(MPI_IN_PLACE, xz, itemCount * 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    updateAG(av, gv, a, g, ix, xz, amat, itemCount, track, burnin, count);

    if (track) {
      count++;
    }
  }

  struct Estimates estimates;
  estimates.persons = (double*)malloc(personCount * 2 * sizeof(double));
  estimates.items = (double*)malloc(itemCount * 4 * sizeof(double));

  double srb = sqrt(batches);
  int bmi = batches - 1;

  double t1, t2, s1, s2, sum1, sum2, m1, m2;

  for (int p = 0; p < personPartitionCount; ++p) {
    count = 0;
    t1 = 0.0;
    s1 = 0.0;
    for (int b = 0; b < batches; ++b) {
      sum1 = 0.0;
      for (int bs = 0; bs < bsize; ++bs) {
        sum1 += thv[p * burnin + count];
        count++;
      }
      m1 = (double) sum1 / bsize;
      t1 += m1;
      s1 += m1 * m1;
    }

    estimates.persons[p * 2 + 0] = (double) t1 / batches;
    estimates.persons[p * 2 + 1] = sqrt((double) (s1 - (t1 * (double) t1 / batches)) / bmi) / srb;
  }

  MPI_Gatherv(MPI_IN_PLACE, personPartitionCount * 2, MPI_DOUBLE, estimates.persons, pxb, pxbd, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  for (int i = 0; i < itemCount; ++i) {
    count = 0;
    t1 = 0.0;
    t2 = 0.0;
    s1 = 0.0;
    s2 = 0.0;
    for (int b = 0; b < batches; ++b) {
      sum1 = 0.0;
      sum2 = 0.0;
      for (int bs = 0; bs < bsize; ++bs) {
        sum1 += av[i * burnin + count];
        sum2 += gv[i * burnin + count];
        count++;
      }
      m1 = (double) sum1 / bsize;
      m2 = (double) sum2 / bsize;
      t1 += m1;
      t2 += m2;
      s1 += m1 * m1;
      s2 += m2 * m2;
    }

    estimates.items[i * 4 + 0] = (double) t1 / batches;
    estimates.items[i * 4 + 1] = sqrt((double) (s1 - (t1 * (double) t1 / batches)) / bmi) / srb;
    estimates.items[i * 4 + 2] = (double) t2 / batches;
    estimates.items[i * 4 + 3] = sqrt((double) (s2 - (t2 * (double) t2 / batches)) / bmi) / srb;
  }

  free(personPartitions);
  free(pi);
  free(pid);
  free(pxi);
  free(pxid);
  free(pxb);
  free(pxbd);

  free(z);
  free(xx);
  free(ix);
  free(amat);
  free(xz);

  free(bz);
  free(bi);

  free(av);
  free(gv);
  free(thv);

  return estimates;
}
