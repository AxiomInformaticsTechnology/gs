#include "sampler.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#ifdef MVAMPICH
#define ENV_LOCAL_RANK "MV2_COMM_WORLD_LOCAL_RANK"
#else
#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"
#endif

#ifndef PERSON_THREADS
#define PERSON_THREADS 32
#endif

#ifndef ITEM_THREADS
#define ITEM_THREADS 16
#endif

#ifndef THREADS
#define THREADS 64
#endif

const double alpha = 1.0;
const double beta = 0.0;

static __device__ __constant__ int DEV_TOTAL_PERSON_COUNT;
static __device__ __constant__ int DEV_PERSON_COUNT;
static __device__ __constant__ int DEV_ITEM_COUNT;

static __device__ __constant__ int DEV_BSIZE;

static __device__ __constant__ double DEV_MU;
static __device__ __constant__ double DEV_VAR;

static __global__ void kernelUpdateZ(double* devU, int* devY, double* devTH, double* devA, double* devG, double* devZ)
{
  int person = blockIdx.x * blockDim.x + threadIdx.x;
  int item = blockIdx.y * blockDim.y + threadIdx.y;
  if (person < DEV_PERSON_COUNT && item < DEV_ITEM_COUNT) {
    int index = person * DEV_ITEM_COUNT + item;
    double lp = devTH[person] * devA[item] - devG[item];
    double bb = normcdf(0.0 - lp);
    double u = devU[index];
    double tmp = devY[index] == 0 ? bb * u : ((1.0 - bb) * u) + bb;
    devZ[index] = normcdfinv(tmp) + lp;
  }
}
static void cudaUpdateZ(double* devU, int* devY, double* devTH, double* devA, double* devG, double* devZ, dim3 block, dim3 grid)
{
  kernelUpdateZ<<<grid, block>>>(devU, devY, devTH, devA, devG, devZ);
}

static __global__ void kernelCalcV(double* devA, double* devPVAR, double* devV)
{
  devV[0] = 0.0;
  for (int item = 0; item < DEV_ITEM_COUNT; ++item) {
    devV[0] += devA[item] * devA[item];
  }
  devPVAR[0] = 1.0 / (devV[0] + 1.0 / DEV_VAR);
  devV[0] = sqrt(devPVAR[0]);
}
static __global__ void kernelUpdateTH(double* devN, double* devTH, double* devA, double* devG, double* devZ, double* devPVAR, double* devV)
{
  int person = blockIdx.x * blockDim.x + threadIdx.x;
  if (person < DEV_PERSON_COUNT) {
    double mn = 0.0;
    int personRow = person * DEV_ITEM_COUNT;
    for (int item = 0; item < DEV_ITEM_COUNT; ++item) {
      mn += devA[item] * (devZ[personRow + item] + devG[item]);
    }
    double pmean = (mn + DEV_MU / DEV_VAR) * devPVAR[0];
    devTH[person] = devN[person] * devV[0] + pmean;
  }
}
static void cudaUpdateTH(double* devN, double* devTH, double* devA, double* devG, double* devZ, double* devPVAR, double* devV, int personGridDim, int personBlockDim)
{
  kernelCalcV<<<1, 1>>>(devA, devPVAR, devV);
  kernelUpdateTH<<<personGridDim, personBlockDim>>>(devN, devTH, devA, devG, devZ, devPVAR, devV);
}

__inline__ __device__ double2 warpReduceSumDouble2(double2 val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down_sync(0xffffffff, val.x, offset);
    val.y += __shfl_down_sync(0xffffffff, val.y, offset);
  }
  return val;
}
__inline__ __device__ double2 blockReduceSumDouble2(double2 val)
{
  __shared__ double2 shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSumDouble2(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  double2 zero; zero.x = zero.y = 0.0;
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;
  if (wid == 0) {
    val = warpReduceSumDouble2(val);
  }

  return val;
}
static __global__ void kernelBlockReduceStableXX(double2* devXXB, double* devTH)
{
  double2 sum; sum.x = sum.y = 0.0;
  for (int person = blockIdx.x * blockDim.x + threadIdx.x; person < DEV_PERSON_COUNT; person += blockDim.x * gridDim.x) {
    sum.x += devTH[person] * devTH[person];
    sum.y -= devTH[person];
  }
  sum = blockReduceSumDouble2(sum);
  if (threadIdx.x == 0) {
    devXXB[blockIdx.x] = sum;
  }
}
static __global__ void kernelReduceStableXX(double* devXX, double2* devXXB, int reduceGridDim)
{
  double2 sum; sum.x = sum.y = 0.0;
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < reduceGridDim; b += blockDim.x * gridDim.x) {
    sum.x += devXXB[b].x;
    sum.y += devXXB[b].y;
  }
  sum = blockReduceSumDouble2(sum);
  if (threadIdx.x == 0) {
    devXX[0] = sum.x;
    devXX[1] = sum.y;
  }
}
static void cudaCalcXX(double* devTH, double* devXX, double2* devXXB, int reduceGridDim, int reduceBlockDim)
{
  kernelBlockReduceStableXX<<<reduceGridDim, reduceBlockDim>>>(devXXB, devTH);
  kernelReduceStableXX<<<1, reduceGridDim>>>(devXX, devXXB, reduceGridDim);
}

static __global__ void kernelCalcAMAT(double* devTH, double* devXX, double* devIX, double* devAMAT)
{
  double dt;
  double4 xx;

  xx.x = devXX[0];
  xx.y = devXX[1];
  xx.z = xx.y;
  xx.w = (double)DEV_TOTAL_PERSON_COUNT;

  dt = xx.x * xx.w - xx.y * xx.z;

  devIX[0] = xx.w / dt;
  devIX[1] = -xx.y / dt;
  devIX[2] = -xx.z / dt;
  devIX[3] = xx.x / dt;

  devAMAT[0] = sqrt(devIX[0]);
  devAMAT[1] = 0.0;
  devAMAT[2] = devIX[1] / devAMAT[0];
  devAMAT[3] = sqrt(devIX[3] - devAMAT[2] * devAMAT[2]);
}
static __global__ void kernelCudaUpdateAG(double* devN2, double* devTH, double* devA, double* devG, double* devZ, double* devXZ, double* devIX, double* devAMAT)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item < DEV_ITEM_COUNT) {
    double2 xz, bz, bi;

    xz.x = devXZ[item * 2 + 0];
    xz.y = devXZ[item * 2 + 1];

    bz.x = (devIX[0] * xz.x) + (devIX[1] * xz.y);
    bz.y = (devIX[2] * xz.x) + (devIX[3] * xz.y);

    bi.x = devN2[item * 2 + 0];
    bi.y = devN2[item * 2 + 1];

    devA[item] = ((bi.x * devAMAT[0]) + (bi.y * devAMAT[2])) + bz.x;
    devG[item] = ((bi.x * devAMAT[1]) + (bi.y * devAMAT[3])) + bz.y;
  }
}
static void cudaCalcXZ(cublasHandle_t handle, double* devOnes, double* devTH, double* devZ, double* devXZ, int itemCount, int personCount)
{
  cublasDgemv(handle, CUBLAS_OP_N, itemCount, personCount, &alpha, devZ, itemCount, devTH, 1, &beta, &devXZ[0], 2);
  cublasDgemv(handle, CUBLAS_OP_N, itemCount, personCount, &alpha, devZ, itemCount, devOnes, 1, &beta, &devXZ[1], 2);
}
static void cudaUpdateAG(cublasHandle_t handle, double* devN2, double* devOnes, double* devTH, double* devA, double* devG, double* devZ, double* devXX, double* devXZ, double* devIX, double* devAMAT, int itemGridDim, int itemBlockDim, int itemCount, int personCount)
{ 
  kernelCalcAMAT<<<1, 1>>>(devTH, devXX, devIX, devAMAT);
  cudaCalcXZ(handle, devOnes, devTH, devZ, devXZ, itemCount, personCount);
  kernelCudaUpdateAG<<<itemGridDim, itemBlockDim>>>(devN2, devTH, devA, devG, devZ, devXZ, devIX, devAMAT);
}

static void cudaUpdateAG(double* devN2, double* devTH, double* devA, double* devG, double* devZ, double* devXX, double* devXZ, double* devIX, double* devAMAT, int itemGridDim, int itemBlockDim)
{
  kernelCalcAMAT<<<1, 1>>>(devTH, devXX, devIX, devAMAT);
  kernelCudaUpdateAG<<<itemGridDim, itemBlockDim>>>(devN2, devTH, devA, devG, devZ, devXZ, devIX, devAMAT);
}

static __global__ void kernelInitTH(double* devTHV)
{
  int person = blockIdx.x * blockDim.x + threadIdx.x;
  if (person < DEV_PERSON_COUNT) {
    devTHV[person * 3 + 1] = 0.0;
    devTHV[person * 3 + 2] = 0.0;
  }
}
static __global__ void kernelInitAG(double* devAV, double* devGV)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item < DEV_ITEM_COUNT) {
    devAV[item * 3 + 1] = 0.0;
    devAV[item * 3 + 2] = 0.0;
    devGV[item * 3 + 1] = 0.0;
    devGV[item * 3 + 2] = 0.0;
  }
}
static void cudaInitTHEstimates(cudaStream_t stream, double* devTHV, int personGridDim, int personBlockDim)
{
  kernelInitTH<<<personGridDim, personBlockDim, 0, stream>>>(devTHV);
}
static void cudaInitAGEstimates(cudaStream_t stream, double* devAV, double* devGV, int itemGridDim, int itemBlockDim)
{
  kernelInitAG<<<itemGridDim, itemBlockDim, 0, stream>>>(devAV, devGV);
}

static __global__ void kernelTrackTH(double* devTHV)
{
  int person = blockIdx.x * blockDim.x + threadIdx.x;
  if (person < DEV_PERSON_COUNT) {
    double m1 = (double)devTHV[person * 3 + 0] / DEV_BSIZE;
    devTHV[person * 3 + 1] += m1;
    devTHV[person * 3 + 2] += m1 * m1;
  }
}
static __global__ void kernelTrackAG(double* devAV, double* devGV)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item < DEV_ITEM_COUNT) {
    double m1 = (double)devAV[item * 3 + 0] / DEV_BSIZE;
    double m2 = (double)devGV[item * 3 + 0] / DEV_BSIZE;
    devAV[item * 3 + 1] += m1;
    devGV[item * 3 + 1] += m2;
    devAV[item * 3 + 2] += m1 * m1;
    devGV[item * 3 + 2] += m2 * m2;
  }
}
static void cudaTrackTHEstimates(cudaStream_t stream, double* devTHV, int personGridDim, int personBlockDim)
{
  kernelTrackTH<<<personGridDim, personBlockDim, 0, stream>>>(devTHV);
}
static void cudaTrackAGEstimates(cudaStream_t stream, double* devAV, double* devGV, int itemGridDim, int itemBlockDim)
{
  kernelTrackAG<<<itemGridDim, itemBlockDim, 0, stream>>>(devAV, devGV);
}

static __global__ void kernelCopyTH(double* devTHV, double* devTH)
{
  int person = blockIdx.x * blockDim.x + threadIdx.x;
  if (person < DEV_PERSON_COUNT) {
    devTHV[person * 3 + 0] = devTH[person];
  }
}
static __global__ void kernelCopyAG(double* devAV, double* devGV, double* devA, double* devG)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item < DEV_ITEM_COUNT) {
    devAV[item * 3 + 0] = devA[item];
    devGV[item * 3 + 0] = devG[item];
  }
}
static void cudaCopyTHEstimates(cudaStream_t stream, double* devTHV, double* devTH, int personGridDim, int personBlockDim)
{
  kernelCopyTH<<<personGridDim, personBlockDim, 0, stream>>>(devTHV, devTH);
}
static void cudaCopyAGEstimates(cudaStream_t stream, double* devAV, double* devGV, double* devA, double* devG, int itemGridDim, int itemBlockDim)
{
  kernelCopyAG<<<itemGridDim, itemBlockDim, 0, stream>>>(devAV, devGV, devA, devG);
}

static __global__ void kernelSumTH(double* devTHV, double* devTH)
{
  int person = blockIdx.x * blockDim.x + threadIdx.x;
  if (person < DEV_PERSON_COUNT) {
    devTHV[person * 3 + 0] += devTH[person];
  }
}
static __global__ void kernelSumAG(double* devAV, double* devGV, double* devA, double* devG)
{
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item < DEV_ITEM_COUNT) {
    devAV[item * 3 + 0] += devA[item];
    devGV[item * 3 + 0] += devG[item];
  }
}
static void cudaSumTHEstimates(cudaStream_t stream, double* devTHV, double* devTH, int personGridDim, int personBlockDim)
{ 
  kernelSumTH<<<personGridDim, personBlockDim, 0, stream>>>(devTHV, devTH);
}
static void cudaSumAGEstimates(cudaStream_t stream, double* devAV, double* devGV, double* devA, double* devG, int itemGridDim, int itemBlockDim)
{ 
  kernelSumAG<<<itemGridDim, itemBlockDim, 0, stream>>>(devAV, devGV, devA, devG);
}

static void cudaTrackStats(cudaStream_t streamTHV, cudaStream_t streamAGV, double* devTHV, double* devAV, double* devGV, double* devTH, double* devA, double* devG, int personGridDim, int personBlockDim, int itemGridDim, int itemBlockDim, int iteration, int burnin, int bsize)
{
  if (((iteration - burnin) % bsize) == 0) {
    if (iteration == burnin) {
      cudaInitTHEstimates(streamTHV, devTHV, personGridDim, personBlockDim);
      cudaInitAGEstimates(streamAGV, devAV, devGV, itemGridDim, itemBlockDim);
    }
    else {
      cudaTrackTHEstimates(streamTHV, devTHV, personGridDim, personBlockDim);
      cudaTrackAGEstimates(streamAGV, devAV, devGV, itemGridDim, itemBlockDim);
    }
    cudaCopyTHEstimates(streamTHV, devTHV, devTH, personGridDim, personBlockDim);
    cudaCopyAGEstimates(streamAGV, devAV, devGV, devA, devG, itemGridDim, itemBlockDim);
  }
  else {
    cudaSumTHEstimates(streamTHV, devTHV, devTH, personGridDim, personBlockDim);
    cudaSumAGEstimates(streamAGV, devAV, devGV, devA, devG, itemGridDim, itemBlockDim);
  }
}

static void cudaTrackTHStats(cudaStream_t streamTHV, double* devTHV, double* devTH, int personGridDim, int personBlockDim, int iteration, int burnin, int bsize)
{
  if (((iteration - burnin) % bsize) == 0) {
    if (iteration == burnin) {
      cudaInitTHEstimates(streamTHV, devTHV, personGridDim, personBlockDim);
    }
    else {
      cudaTrackTHEstimates(streamTHV, devTHV, personGridDim, personBlockDim);
    }
    cudaCopyTHEstimates(streamTHV, devTHV, devTH, personGridDim, personBlockDim);
  }
  else {
    cudaSumTHEstimates(streamTHV, devTHV, devTH, personGridDim, personBlockDim);
  }
}
