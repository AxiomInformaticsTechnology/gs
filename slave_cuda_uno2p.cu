#include "cuda_sampler.cuh"

void Sampler::slaveCudaUno2p(
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

  int nextEvenPersonPartitionCount = personPartitionCount % 2 == 0 ? personPartitionCount : personPartitionCount + 1;

  int* y = (int*)malloc(personPartitionCount * itemCount * sizeof(int));

  double* th = (double*)malloc(personPartitionCount * sizeof(double));

  double* thv = (double*)malloc(personPartitionCount * 3 * sizeof(double));

  double* ones = (double*)malloc(personPartitionCount * sizeof(double));
  std::fill(ones, ones + personPartitionCount, -1.0);

  MPI_Scatterv(NULL, NULL, NULL, MPI_UNSIGNED, y, personPartitionCount * itemCount, MPI_UNSIGNED, ROOT, MPI_COMM_WORLD);

  MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, th, personPartitionCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  dim3 zBlock(PERSON_THREADS, ITEM_THREADS, 1);
  dim3 zGrid((personPartitionCount + zBlock.x - 1) / zBlock.x, (itemCount + zBlock.y - 1) / zBlock.y, 1);

  const int personBlockDim = THREADS;
  const int personGridDim = (personPartitionCount + THREADS - 1) / THREADS;

  const int reducePersonGridDim = THREADS * 2;

  curandGenerator_t rng;

  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, seed);

  cublasHandle_t cbh;
  cublasCreate(&cbh);

  int* devY;

  double* devU, * devN;

  double* devZ, * devA, * devG, * devTH;

  double* devOnes;

  double* devPVAR, * devV;

  double2* devXXB;

  double* devXX, * devXZ;

  double* devTHV;

  cudaStream_t streamTHV;
  cudaStreamCreate(&streamTHV);

  cudaMalloc((void**)&devU, personPartitionCount * itemCount * sizeof(double));
  cudaMalloc((void**)&devY, personPartitionCount * itemCount * sizeof(int));
  cudaMalloc((void**)&devZ, personPartitionCount * itemCount * sizeof(double));

  cudaMalloc((void**)&devN, nextEvenPersonPartitionCount * sizeof(double));

  cudaMalloc((void**)&devTH, personPartitionCount * sizeof(double));

  cudaMalloc((void**)&devOnes, personPartitionCount * sizeof(double));

  cudaMalloc((void**)&devA, itemCount * sizeof(double));
  cudaMalloc((void**)&devG, itemCount * sizeof(double));

  cudaMalloc((void**)&devV, sizeof(double));
  cudaMalloc((void**)&devPVAR, sizeof(double));

  cudaMalloc((void**)&devXXB, reducePersonGridDim * sizeof(double2));

  cudaMalloc((void**)&devXX, 2 * sizeof(double));
  cudaMalloc((void**)&devXZ, itemCount * 2 * sizeof(double));

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

    MPI_Reduce(devXX, devXX, 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(devXZ, devXZ, itemCount * 2, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (iteration >= burnin) {
      cudaTrackTHStats(streamTHV, devTHV, devTH, personGridDim, personBlockDim, iteration, burnin, bsize);
    }
  }

  cudaMemcpy(thv, devTHV, personPartitionCount * 3 * sizeof(double), cudaMemcpyDeviceToHost);

  double* persons = (double*)malloc(personPartitionCount * 2 * sizeof(double));

  double srb = sqrt(batches);
  int bmi = batches - 1;

  for (int p = 0; p < personPartitionCount; ++p) {
    double m = (double)thv[p * 3 + 0] / bsize;
    thv[p * 3 + 1] += m;
    thv[p * 3 + 2] += m * m;

    persons[p * 2 + 0] = (double)thv[p * 3 + 1] / batches;
    persons[p * 2 + 1] = sqrt((double)(thv[p * 3 + 2] - (thv[p * 3 + 1] * (double)thv[p * 3 + 1] / batches)) / bmi) / srb;
  }

  MPI_Gatherv(persons, personPartitionCount * 2, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

  free(y);
  free(th);

  free(ones);

  free(thv);

  free(persons);

  cudaFree(devU);
  cudaFree(devN);

  cudaFree(devY);
  cudaFree(devZ);
  cudaFree(devA);
  cudaFree(devG);
  cudaFree(devTH);

  cudaFree(devOnes);

  cudaFree(devPVAR);
  cudaFree(devV);

  cudaFree(devXXB);

  cudaFree(devXX);
  cudaFree(devXZ);

  cudaFree(devTHV);

  cudaStreamDestroy(streamTHV);

  cublasDestroy(cbh);

  curandDestroyGenerator(rng);
}
