#pragma once

#include <array>
using std::array;

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

#include <iostream>
using std::cout;
using std::endl;
using std::string;

#include <iomanip>
using std::setw;

#include <numeric>
using std::accumulate;
using std::max;
using std::min;

#include <boost/random/mersenne_twister.hpp>
using boost::random::mt19937_64;

#include <mpi.h>

struct sum_of_squares {
  double operator()(double sum, double v) const {
    return sum + v * v;
  }
};

#define TAG 0
#define ROOT 0

#define MU 0.0
#define SIGMA 1.0

struct Input {
  int* y;
  double* a;
  double* g;
  double* th;
};

struct Params {
  string mode;
  int seed;

  int personCount;
  int itemCount;
  int iterations;
  int burnin;
  int batches;
  double mu;
  double var;
  int unif;

  int rank;
  int size;
};

struct Scores {
  int* y;
  double* as;
  double* gs;
  double* ths;
};

struct Estimates {
  double* items;
  double* persons;
};

class Sampler {
private:
  int seed;
  mt19937_64 rng;
public:
  Sampler(int seed);
  ~Sampler();

  Scores generate_scores(int personCount, int itemCount);

  double* generate_array(int size, double value);
  double* generate_alpha(int itemCount);
  double* generate_gamma(int itemCount);
  double* generate_theta(int personCount);

  double* generate_gamma_from_scores(int* scores, int personCount, int itemCount);

  double* unifrnd(double a, double b, int sz1);
  double unifrnd(double a, double b);
  double* normrnd(double mu, double sigma, int sz1);
  double normrnd(double mu, double sigma);
  double normcdf(double x, double mu, double sigma);
  double norminv(double x, double mu, double sigma);

  double* normrnd(int sz1);
  double normrnd();
  double normcdf(double x);
  double norminv(double x);

  double mean(double* v, int sz);
  double correlation(double* v1, double* v2, int sz);

  void printMatrix(double* m, int sz1, int sz2, string name, int cw);

  void printArray(double* m, int sz, string name, int cw);

  void printScores(int* y, int personCount, int itemCount);

  void printItems(double* as, double* gs, double* items, int itemCount);

  void printPersons(double* ths, double* persons, int personCount);

  Estimates uno2p(
    Input input,
    Params params
  );

  Estimates cudaUno2p(
    Input input,
    Params params
  );

  Estimates masterUno2p(
    Input input,
    Params params
  );

  void slaveUno2p(
    Params params
  );

  void initDevice();

  Estimates masterCudaUno2p(
    Input input,
    Params params
  );

  void slaveCudaUno2p(
    Params params
  );

  // common
  void updateZ(int* y, double* th, double* a, double* g, double* z, int personCount, int itemCount);

  // serial
  void updateTH(double* th, double* a, double* g, double* z, double mu, double var, int personCount, int itemCount);
  void updateAG(double* th, double* a, double* g, double* z, int personCount, int itemCount);

  // mpi
  void updateTH(double* lxx, double* thv, double* th, double* a, double* g, double* z, double mu, double var, int personCount, int itemCount, bool track, int burnin, int count);
  void calcXZ(double* lxz, double* th, double* z, int personCount, int itemCount);
  void calcAMAT(double* xx, double* ix, double* amat, int personCount);
  void updateAG(double* av, double* gv, double* a, double* g, double* ix, double* xz, double* amat, int itemCount, bool track, int burnin, int count);
  void trackStats(double* thv, double* av, double* gv, double* th, double* a, double* g, int personCount, int itemCount, int iteration, int burnin, int bsize);

  Estimates calcEstimates(double* thv, double* av, double* gv, int batches, int bsize, int personCount, int itemCount);

};
