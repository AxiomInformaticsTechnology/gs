#include "cuda_sampler.cuh"

Estimates Sampler::cudaUno2p(
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

  int nextEvenPersonCount = personCount % 2 == 0 ? personCount : personCount + 1;

  int bsize = (iterations - burnin) / batches;

  double* av = (double*)malloc(itemCount * 3 * sizeof(double));
  double* gv = (double*)malloc(itemCount * 3 * sizeof(double));
  double* thv = (double*)malloc(personCount * 3 * sizeof(double));

  double* ones = (double*)malloc(personCount * sizeof(double));
  std::fill(ones, ones + personCount, -1.0);

  dim3 zBlock(PERSON_THREADS, ITEM_THREADS, 1);
  dim3 zGrid((personCount + zBlock.x - 1) / zBlock.x, (itemCount + zBlock.y - 1) / zBlock.y, 1);

  const int personBlockDim = THREADS;
  const int personGridDim = (personCount + THREADS - 1) / THREADS;

  const int itemBlockDim = THREADS;
  const int itemGridDim = (itemCount + THREADS - 1) / THREADS;

  const int reducePersonGridDim = THREADS * 2;

  cudaSetDevice(0);

  curandGenerator_t rng;

  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(rng, seed);

  cublasHandle_t cbh;
  cublasCreate(&cbh);

  int* devY;

  double* devU, * devN, * devN2;

  double* devZ, * devA, * devG, * devTH;
  
  double * devOnes;

  double* devPVAR, * devV;

  double* devIX, * devAMAT;

  double2* devXXB;

  double* devXX, * devXZ;

  double* devAV, * devGV, * devTHV;

  cudaStream_t streamTHV, streamAGV;

  cudaStreamCreate(&streamTHV);
  cudaStreamCreate(&streamAGV);

  cudaMalloc((void**)&devU, personCount * itemCount * sizeof(double));
  cudaMalloc((void**)&devN, nextEvenPersonCount * sizeof(double));
  cudaMalloc((void**)&devN2, itemCount * 2 * sizeof(double));

  cudaMalloc((void**)&devY, personCount * itemCount * sizeof(int));
  cudaMalloc((void**)&devZ, personCount * itemCount * sizeof(double));
  cudaMalloc((void**)&devA, itemCount * sizeof(double));
  cudaMalloc((void**)&devG, itemCount * sizeof(double));
  cudaMalloc((void**)&devTH, personCount * sizeof(double));

  cudaMalloc((void**)&devOnes, personCount * sizeof(double));

  cudaMalloc((void**)&devV, sizeof(double));
  cudaMalloc((void**)&devPVAR, sizeof(double));

  cudaMalloc((void**)&devXXB, reducePersonGridDim * sizeof(double2));

  cudaMalloc((void**)&devXX, 2 * sizeof(double));
  cudaMalloc((void**)&devXZ, itemCount * 2 * sizeof(double));

  cudaMalloc((void**)&devIX, 4 * sizeof(double));
  cudaMalloc((void**)&devAMAT, 4 * sizeof(double));

  cudaMalloc((void**)&devAV, itemCount * 3 * sizeof(double));
  cudaMalloc((void**)&devGV, itemCount * 3 * sizeof(double));
  cudaMalloc((void**)&devTHV, personCount * 3 * sizeof(double));

  cudaMemcpyToSymbolAsync(DEV_TOTAL_PERSON_COUNT, &personCount, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbolAsync(DEV_PERSON_COUNT, &personCount, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbolAsync(DEV_ITEM_COUNT, &itemCount, sizeof(int), 0, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbolAsync(DEV_BSIZE, &bsize, sizeof(int), 0, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbolAsync(DEV_MU, &mu, sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbolAsync(DEV_VAR, &var, sizeof(double), 0, cudaMemcpyHostToDevice);

  cudaMemcpyAsync(devOnes, ones, personCount * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpyAsync(devY, y, personCount * itemCount * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpyAsync(devTH, th, personCount * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(devA, a, itemCount * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(devG, g, itemCount * sizeof(double), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    curandGenerateUniformDouble(rng, devU, personCount * itemCount);
    cudaUpdateZ(devU, devY, devTH, devA, devG, devZ, zBlock, zGrid);

    curandGenerateNormalDouble(rng, devN, nextEvenPersonCount, 0.0, 1.0);
    cudaUpdateTH(devN, devTH, devA, devG, devZ, devPVAR, devV, personGridDim, personBlockDim);

    cudaCalcXX(devTH, devXX, devXXB, reducePersonGridDim, personBlockDim);

    curandGenerateNormalDouble(rng, devN2, itemCount * 2, 0.0, 1.0);
    cudaUpdateAG(cbh, devN2, devOnes, devTH, devA, devG, devZ, devXX, devXZ, devIX, devAMAT, itemGridDim, itemBlockDim, itemCount, personCount);

    if (iteration >= burnin) {
      cudaTrackStats(streamTHV, streamAGV, devTHV, devAV, devGV, devTH, devA, devG, personGridDim, personBlockDim, itemGridDim, itemBlockDim, iteration, burnin, bsize);
    }
  }

  cudaMemcpy(thv, devTHV, personCount * 3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(av, devAV, itemCount * 3 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(gv, devGV, itemCount * 3 * sizeof(double), cudaMemcpyDeviceToHost);

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

  cudaDeviceReset();

  struct Estimates estimates;
  estimates.items = (double*)malloc(itemCount * 4 * sizeof(double));
  estimates.persons = (double*)malloc(personCount * 2 * sizeof(double));

  double srb = sqrt(batches);
  int bmi = batches - 1;

  for (int i = 0; i < itemCount; ++i) {
    double m1 = av[i * 3 + 0] / bsize;
    double m2 = gv[i * 3 + 0] / bsize;
    av[i * 3 + 1] += m1;
    gv[i * 3 + 1] += m2;
    av[i * 3 + 2] += m1 * m1;
    gv[i * 3 + 2] += m2 * m2;

    estimates.items[i * 4 + 0] = av[i * 3 + 1] / batches;
    estimates.items[i * 4 + 1] = sqrt((av[i * 3 + 2] - (av[i * 3 + 1] * av[i * 3 + 1] / batches)) / bmi) / srb;
    estimates.items[i * 4 + 2] = gv[i * 3 + 1] / batches;
    estimates.items[i * 4 + 3] = sqrt((gv[i * 3 + 2] - (gv[i * 3 + 1] * gv[i * 3 + 1] / batches)) / bmi) / srb;
  }
  for (int p = 0; p < personCount; ++p) {
    double m = thv[p * 3 + 0] / bsize;
    thv[p * 3 + 1] += m;
    thv[p * 3 + 2] += m * m;

    estimates.persons[p * 2 + 0] = thv[p * 3 + 1] / batches;
    estimates.persons[p * 2 + 1] = sqrt((thv[p * 3 + 2] - (thv[p * 3 + 1] * thv[p * 3 + 1] / batches)) / bmi) / srb;
  }

  free(av);
  free(gv);
  free(thv);

  free(ones);

  return estimates;
}
