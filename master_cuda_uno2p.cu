#include "cuda_sampler.cuh"

void Sampler::initDevice() {
  char* localRankStr = NULL;
  int rank = 0, deviceCount = 0;

  if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL) {
    rank = atoi(localRankStr);
  }

  cudaGetDeviceCount(&deviceCount);
  cudaSetDevice(rank % deviceCount);
}

Estimates Sampler::masterCudaUno2p(
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

  for (int r = 0; r < size; ++r) {
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

  int nextEvenPersonPartitionCount = personPartitionCount % 2 == 0 ? personPartitionCount : personPartitionCount + 1;

  double* av = (double*)malloc(itemCount * 3 * sizeof(double));
  double* gv = (double*)malloc(itemCount * 3 * sizeof(double));
  double* thv = (double*)malloc(personPartitionCount * 3 * sizeof(double));

  double* ones = (double*)malloc(personPartitionCount * sizeof(double));
  std::fill(ones, ones + personPartitionCount, -1.0);

  for (int r = 1; r < size; ++r) {
    MPI_Send(&personPartitions[r], 1, MPI_UNSIGNED, r, TAG, MPI_COMM_WORLD);
  }

  MPI_Scatterv(y, pxi, pxid, MPI_UNSIGNED, MPI_IN_PLACE, personCount * itemCount, MPI_UNSIGNED, ROOT, MPI_COMM_WORLD);

  MPI_Scatterv(th, pi, pid, MPI_DOUBLE, MPI_IN_PLACE, personCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  dim3 zBlock(PERSON_THREADS, ITEM_THREADS, 1);
  dim3 zGrid((personPartitionCount + zBlock.x - 1) / zBlock.x, (itemCount + zBlock.y - 1) / zBlock.y, 1);

  const int personBlockDim = THREADS;
  const int personGridDim = (personPartitionCount + THREADS - 1) / THREADS;

  const int itemBlockDim = THREADS;
  const int itemGridDim = (itemCount + THREADS - 1) / THREADS;

  const int reducePersonGridDim = THREADS * 2;

  curandGenerator_t rng;

  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, seed);

  cublasHandle_t cbh;
  cublasCreate(&cbh);

  int* devY;

  double* devU, * devN, * devN2;

  double* devZ, * devA, * devG, * devTH;

  double* devOnes;

  double* devPVAR, * devV;

  double* devIX, * devAMAT;

  double2* devXXB;

  double* devXX, * devXZ;

  double* devAV, * devGV, * devTHV;

  cudaStream_t streamTHV, streamAGV;
  cudaStreamCreate(&streamTHV);
  cudaStreamCreate(&streamAGV);

  cudaMalloc((void**)&devU, personPartitionCount * itemCount * sizeof(double));
  cudaMalloc((void**)&devY, personPartitionCount * itemCount * sizeof(int));
  cudaMalloc((void**)&devZ, personPartitionCount * itemCount * sizeof(double));

  cudaMalloc((void**)&devN, nextEvenPersonPartitionCount * sizeof(double));

  cudaMalloc((void**)&devTH, personPartitionCount * sizeof(double));

  cudaMalloc((void**)&devOnes, personPartitionCount * sizeof(double));

  cudaMalloc((void**)&devN2, itemCount * 2 * sizeof(double));

  cudaMalloc((void**)&devA, itemCount * sizeof(double));
  cudaMalloc((void**)&devG, itemCount * sizeof(double));

  cudaMalloc((void**)&devV, sizeof(double));
  cudaMalloc((void**)&devPVAR, sizeof(double));

  cudaMalloc((void**)&devXXB, reducePersonGridDim * sizeof(double2));

  cudaMalloc((void**)&devXX, 2 * sizeof(double));
  cudaMalloc((void**)&devXZ, itemCount * 2 * sizeof(double));

  cudaMalloc((void**)&devIX, 4 * sizeof(double));
  cudaMalloc((void**)&devAMAT, 4 * sizeof(double));

  cudaMalloc((void**)&devAV, itemCount * 3 * sizeof(double));
  cudaMalloc((void**)&devGV, itemCount * 3 * sizeof(double));

  cudaMalloc((void**)&devTHV, personPartitionCount * 3 * sizeof(double));

  cudaMemcpyToSymbolAsync(DEV_TOTAL_PERSON_COUNT, &personCount, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbolAsync(DEV_PERSON_COUNT, &personPartitionCount, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbolAsync(DEV_ITEM_COUNT, &itemCount, sizeof(int), 0, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbolAsync(DEV_BSIZE, &bsize, sizeof(int), 0, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbolAsync(DEV_MU, &mu, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbolAsync(DEV_VAR, &var, sizeof(double), 0, cudaMemcpyHostToDevice);

  cudaMemcpyAsync(devOnes, ones, personPartitionCount * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpyAsync(devY, y, personPartitionCount * itemCount * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpyAsync(devTH, th, personPartitionCount * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(devA, a, itemCount * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(devG, g, itemCount * sizeof(double), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    MPI_Bcast(devA, itemCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(devG, itemCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    curandGenerateUniformDouble(rng, devU, personPartitionCount * itemCount);
    cudaUpdateZ(devU, devY, devTH, devA, devG, devZ, zBlock, zGrid);

    curandGenerateNormalDouble(rng, devN, nextEvenPersonPartitionCount, 0.0, 1.0);
    cudaUpdateTH(devN, devTH, devA, devG, devZ, devPVAR, devV, personGridDim, personBlockDim);

    cudaCalcXX(devTH, devXX, devXXB, reducePersonGridDim, personBlockDim);
    cudaCalcXZ(cbh, devOnes, devTH, devZ, devXZ, itemCount, personPartitionCount);

    MPI_Reduce(MPI_IN_PLACE, devXX, 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, devXZ, itemCount * 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    curandGenerateNormalDouble(rng, devN2, itemCount * 2, 0.0, 1.0);
    cudaUpdateAG(devN2, devTH, devA, devG, devZ, devXX, devXZ, devIX, devAMAT, itemGridDim, itemBlockDim);

    if (iteration >= burnin) {
      cudaTrackStats(streamTHV, streamAGV, devTHV, devAV, devGV, devTH, devA, devG, personGridDim, personBlockDim, itemGridDim, itemBlockDim, iteration, burnin, bsize);
    }
  }

  cudaMemcpy(thv, devTHV, personPartitionCount * 3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(av, devAV, itemCount * 3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(gv, devGV, itemCount * 3 * sizeof(double), cudaMemcpyDeviceToHost);

  struct Estimates estimates;
  estimates.persons = (double*)malloc(personCount * 2 * sizeof(double));
  estimates.items = (double*)malloc(itemCount * 4 * sizeof(double));

  double srb = sqrt(batches);
  int bmi = batches - 1;

  for (int p = 0; p < personPartitionCount; ++p) {
    double m = (double)thv[p * 3 + 0] / bsize;
    thv[p * 3 + 1] += m;
    thv[p * 3 + 2] += m * m;

    estimates.persons[p * 2 + 0] = (double)thv[p * 3 + 1] / batches;
    estimates.persons[p * 2 + 1] = sqrt((double)(thv[p * 3 + 2] - (thv[p * 3 + 1] * (double)thv[p * 3 + 1] / batches)) / bmi) / srb;
  }

  MPI_Gatherv(MPI_IN_PLACE, personPartitionCount * 2, MPI_DOUBLE, estimates.persons, pxb, pxbd, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  for (int i = 0; i < itemCount; ++i) {
    double m1 = (double)av[i * 3 + 0] / bsize;
    double m2 = (double)gv[i * 3 + 0] / bsize;
    av[i * 3 + 1] += m1;
    gv[i * 3 + 1] += m2;
    av[i * 3 + 2] += m1 * m1;
    gv[i * 3 + 2] += m2 * m2;

    estimates.items[i * 4 + 0] = (double)av[i * 3 + 1] / batches;
    estimates.items[i * 4 + 1] = sqrt((double)(av[i * 3 + 2] - (av[i * 3 + 1] * (double)av[i * 3 + 1] / batches)) / bmi) / srb;
    estimates.items[i * 4 + 2] = (double)gv[i * 3 + 1] / batches;
    estimates.items[i * 4 + 3] = sqrt((double)(gv[i * 3 + 2] - (gv[i * 3 + 1] * (double)gv[i * 3 + 1] / batches)) / bmi) / srb;
  }

  free(personPartitions);
  free(pi);
  free(pid);
  free(pxi);
  free(pxid);
  free(pxb);
  free(pxbd);

  free(ones);

  free(av);
  free(gv);
  free(thv);

  cudaFree(devU);
  cudaFree(devN);
  cudaFree(devN2);

  cudaFree(devY);
  cudaFree(devZ);
  cudaFree(devA);
  cudaFree(devG);
  cudaFree(devTH);

  cudaFree(devOnes);

  cudaFree(devV);
  cudaFree(devPVAR);

  cudaFree(devXXB);

  cudaFree(devXX);
  cudaFree(devXZ);

  cudaFree(devIX);
  cudaFree(devAMAT);

  cudaFree(devAV);
  cudaFree(devGV);
  cudaFree(devTHV);

  cudaStreamDestroy(streamTHV);
  cudaStreamDestroy(streamAGV);

  cublasDestroy(cbh);

  curandDestroyGenerator(rng);

  return estimates;
}
