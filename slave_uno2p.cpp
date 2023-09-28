#include "sampler.hpp"

void Sampler::slaveUno2p(
  Params params
) {
  int personCount = params.personCount;
  int itemCount = params.itemCount;
  int iterations = params.iterations;
  int burnin = params.burnin;
  int batches = params.batches;
  double mu = params.mu;
  double var = params.var;

  int bsize = (iterations - burnin) / batches;

  int personPartitionCount;

  MPI_Recv(&personPartitionCount, 1, MPI_UNSIGNED, ROOT, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int* y = (int*)malloc(personPartitionCount * itemCount * sizeof(int));

  double* z = (double*)malloc(personPartitionCount * itemCount * sizeof(double));

  double* a = (double*)malloc(itemCount * sizeof(double));
  double* g = (double*)malloc(itemCount * sizeof(double));

  double* th = (double*)malloc(personPartitionCount * sizeof(double));

  double* xx = (double*)malloc(4 * sizeof(double));
  double* ix = (double*)malloc(4 * sizeof(double));
  double* amat = (double*)malloc(4 * sizeof(double));

  double* xz = (double*)malloc(itemCount * 2 * sizeof(double));

  double* thv = (double*)malloc(personPartitionCount * burnin * sizeof(double));

  int count = 0;
  bool track = false;

  MPI_Scatterv(NULL, NULL, NULL, MPI_UNSIGNED, y, personPartitionCount * itemCount, MPI_UNSIGNED, ROOT, MPI_COMM_WORLD);

  MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, th, personPartitionCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  xx[3] = (double)personCount;
  amat[1] = 0;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    track = iteration >= burnin;

    MPI_Bcast(a, itemCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(g, itemCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    updateZ(y, th, a, g, z, personPartitionCount, itemCount);

    updateTH(xx, thv, th, a, g, z, mu, var, personPartitionCount, itemCount, track, burnin, count);

    MPI_Reduce(xx, xx, 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    calcAMAT(xx, ix, amat, personCount);

    calcXZ(xz, th, z, personPartitionCount, itemCount);

    MPI_Reduce(xz, xz, itemCount * 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (track) {
      count++;
    }
  }

  double* persons = (double*)malloc(personPartitionCount * 2 * sizeof(double));

  double srb = sqrt(batches);
  int bmi = batches - 1;

  double t1, s1, sum1, m1;

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

    persons[p * 2 + 0] = (double) t1 / batches;
    persons[p * 2 + 1] = sqrt((double) (s1 - (t1 * (double) t1 / batches)) / bmi) / srb;
  }

  MPI_Gatherv(persons, personPartitionCount * 2, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  free(y);
  free(z);
  free(a);
  free(g);
  free(th);
  free(xx);
  free(ix);
  free(amat);
  free(xz);

  free(thv);

  free(persons);
}
