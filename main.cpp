#include "sampler.hpp"

#ifdef LINUX
#include <sys/resource.h>
#endif

void print(Params params) {
  int bsize = (params.iterations - params.burnin) / params.batches;

  cout << "model: 2pno" << endl;
  cout << "seed: " << params.seed << endl;
  cout << "person count: " << params.personCount << endl;
  cout << "item count: " << params.itemCount << endl;
  cout << "iterations: " << params.iterations << endl;
  cout << "burnin: " << params.burnin << endl;
  cout << "batch size: " << bsize << endl;
  cout << "mean: " << params.mu << endl;
  cout << "variance: " << params.var << endl;
  cout << "unif: " << params.unif << endl;
}

void serial(Params params) {
  cout << "mode: serial" << endl;
  print(params);

  int personCount = params.personCount;
  int itemCount = params.itemCount;

  Sampler sampler(params.seed);
  Scores scores = sampler.generate_scores(personCount, itemCount);

  struct Input input;
  input.y = scores.y;
  input.a = sampler.generate_alpha(itemCount);
  input.g = sampler.generate_gamma(itemCount);
  input.th = sampler.generate_theta(personCount);

  auto start = high_resolution_clock::now();

  Estimates estimates = sampler.uno2p(input, params);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << endl << "time: " << (duration.count() / (double)1000) << " seconds" << endl << endl;

  free(input.a);
  free(input.g);
  free(input.th);

  double* aEst = (double*)malloc(itemCount * sizeof(double));
  double* gEst = (double*)malloc(itemCount * sizeof(double));
  for (int i = 0; i < itemCount; ++i) {
    aEst[i] = estimates.items[i * 4 + 0];
    gEst[i] = estimates.items[i * 4 + 2];
  }

  cout << "a: " << sampler.correlation(aEst, scores.as, itemCount) << endl;
  cout << "g: " << sampler.correlation(gEst, scores.gs, itemCount) << endl;

  free(aEst);
  free(gEst);

  double* thEst = (double*)malloc(personCount * sizeof(double));
  for (int p = 0; p < personCount; ++p) {
    thEst[p] = estimates.persons[p * 2 + 0];
  }

  cout << "th: " << sampler.correlation(thEst, scores.ths, personCount) << endl << endl;

  free(thEst);

  // sampler.printScores(scores.y, personCount, itemCount);
  sampler.printItems(scores.as, scores.gs, estimates.items, itemCount);
  // sampler.printPersons(scores.ths, estimates.persons, personCount);

  free(scores.y);
  free(scores.as);
  free(scores.gs);
  free(scores.ths);

  free(estimates.items);
  free(estimates.persons);
}

void cuda(Params params) {
  cout << "mode: cuda" << endl;
  print(params);

  int personCount = params.personCount;
  int itemCount = params.itemCount;

  Sampler sampler(params.seed);
  Scores scores = sampler.generate_scores(personCount, itemCount);

  struct Input input;
  input.y = scores.y;
  input.a = sampler.generate_alpha(itemCount);
  input.g = sampler.generate_gamma(itemCount);
  input.th = sampler.generate_theta(personCount);

  auto start = high_resolution_clock::now();

  Estimates estimates = sampler.cudaUno2p(input, params);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << endl << "time: " << (duration.count() / (double)1000) << " seconds" << endl << endl;

  free(input.a);
  free(input.g);
  free(input.th);

  double* aEst = (double*)malloc(itemCount * sizeof(double));
  double* gEst = (double*)malloc(itemCount * sizeof(double));
  for (int i = 0; i < itemCount; ++i) {
    aEst[i] = estimates.items[i * 4 + 0];
    gEst[i] = estimates.items[i * 4 + 2];
  }

  cout << "a: " << sampler.correlation(aEst, scores.as, itemCount) << endl;
  cout << "g: " << sampler.correlation(gEst, scores.gs, itemCount) << endl;

  free(aEst);
  free(gEst);

  double* thEst = (double*)malloc(personCount * sizeof(double));
  for (int p = 0; p < personCount; ++p) {
    thEst[p] = estimates.persons[p * 2 + 0];
  }

  cout << "th: " << sampler.correlation(thEst, scores.ths, personCount) << endl << endl;

  free(thEst);

  // sampler.printScores(scores.y, personCount, itemCount);
  sampler.printItems(scores.as, scores.gs, estimates.items, itemCount);
  // sampler.printPersons(scores.ths, estimates.persons, personCount);

  free(scores.y);
  free(scores.as);
  free(scores.gs);
  free(scores.ths);

  free(estimates.items);
  free(estimates.persons);
}

void mpi(Params params) {
  Sampler sampler(params.seed + 7 * params.rank);

  if (params.rank == ROOT) {
    int personCount = params.personCount;
    int itemCount = params.itemCount;

    cout << "mode: mpi" << endl;
    print(params);

    cout << "size: " << params.size << endl;

    Scores scores = sampler.generate_scores(personCount, itemCount);

    struct Input input;
    input.y = scores.y;
    input.a = sampler.generate_alpha(itemCount);
    input.g = sampler.generate_gamma(itemCount);
    input.th = sampler.generate_theta(personCount);

    auto start = high_resolution_clock::now();

    Estimates estimates = sampler.masterUno2p(input, params);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << endl << "time: " << (duration.count() / (double)1000) << " seconds" << endl << endl;

    free(input.a);
    free(input.g);
    free(input.th);

    double* aEst = (double*)malloc(itemCount * sizeof(double));
    double* gEst = (double*)malloc(itemCount * sizeof(double));
    for (int i = 0; i < itemCount; ++i) {
      aEst[i] = estimates.items[i * 4 + 0];
      gEst[i] = estimates.items[i * 4 + 2];
    }

    cout << "a: " << sampler.correlation(aEst, scores.as, itemCount) << endl;
    cout << "g: " << sampler.correlation(gEst, scores.gs, itemCount) << endl;

    free(aEst);
    free(gEst);

    double* thEst = (double*)malloc(personCount * sizeof(double));
    for (int p = 0; p < personCount; ++p) {
      thEst[p] = estimates.persons[p * 2 + 0];
    }

    cout << "th: " << sampler.correlation(thEst, scores.ths, personCount) << endl << endl;

    free(thEst);

    // sampler.printScores(scores.y, personCount, itemCount);
    sampler.printItems(scores.as, scores.gs, estimates.items, itemCount);
    // sampler.printPersons(scores.ths, estimates.persons, personCount);

    free(scores.y);
    free(scores.as);
    free(scores.gs);
    free(scores.ths);

    free(estimates.items);
    free(estimates.persons);

  } else {
    sampler.slaveUno2p(params);
  }
}

void cuda_mpi(Params params) {
  Sampler sampler(params.seed + 7 * params.rank);

  sampler.initDevice();

  if (params.rank == ROOT) {
    int personCount = params.personCount;
    int itemCount = params.itemCount;

    cout << "mode: cuda-aware mpi" << endl;
    print(params);

    cout << "size: " << params.size << endl;

    Scores scores = sampler.generate_scores(personCount, itemCount);

    struct Input input;
    input.y = scores.y;
    input.a = sampler.generate_alpha(itemCount);
    input.g = sampler.generate_gamma(itemCount);
    input.th = sampler.generate_theta(personCount);

    auto start = high_resolution_clock::now();

    Estimates estimates = sampler.masterCudaUno2p(input, params);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << endl << "time: " << (duration.count() / (double)1000) << " seconds" << endl << endl;

    free(input.a);
    free(input.g);
    free(input.th);

    double* aEst = (double*)malloc(itemCount * sizeof(double));
    double* gEst = (double*)malloc(itemCount * sizeof(double));
    for (int i = 0; i < itemCount; ++i) {
      aEst[i] = estimates.items[i * 4 + 0];
      gEst[i] = estimates.items[i * 4 + 2];
    }

    cout << "a: " << sampler.correlation(aEst, scores.as, itemCount) << endl;
    cout << "g: " << sampler.correlation(gEst, scores.gs, itemCount) << endl;

    free(aEst);
    free(gEst);

    double* thEst = (double*)malloc(personCount * sizeof(double));
    for (int p = 0; p < personCount; ++p) {
      thEst[p] = estimates.persons[p * 2 + 0];
    }

    cout << "th: " << sampler.correlation(thEst, scores.ths, personCount) << endl << endl;

    free(thEst);

    // sampler.printScores(scores.y, personCount, itemCount);
    sampler.printItems(scores.as, scores.gs, estimates.items, itemCount);
    // sampler.printPersons(scores.ths, estimates.persons, personCount);

    free(scores.y);
    free(scores.as);
    free(scores.gs);
    free(scores.ths);

    free(estimates.items);
    free(estimates.persons);

  } else {
    sampler.slaveCudaUno2p(params);
  }
}

void run(Params params) {
  if (params.mode == "serial") {
    serial(params);
  } else if (params.mode == "cuda") {
    cuda(params);
  } else if (params.mode == "mpi") {
    mpi(params);
  } else if (params.mode == "cuda-mpi") {
    cuda_mpi(params);
  } else {
    cout << params.mode << " is not an acceptable mode. [serial|cuda|mpi|cuda-mpi]";
  }
}

int main(int argc, char** argv) {
#ifdef LINUX
  struct rlimit lim = { 2000000000, 4000000000 };
  if (setrlimit(RLIMIT_STACK, &lim) == -1) {
    return 1;
  }
#endif

  string mode = (argc > 1) ? argv[1] : "serial";
  int personCount = (argc > 2) ? atoi(argv[2]) : 500;
  int itemCount = (argc > 3) ? atoi(argv[3]) : 20;
  int iterations = (argc > 4) ? atoi(argv[4]) : 10000;
  int burnin = (argc > 5) ? atoi(argv[5]) : 5000;
  int batches = (argc > 6) ? atoi(argv[6]) : 500;
  double mu = (argc > 7) ? atoi(argv[7]) : 0;
  double var = (argc > 8) ? atoi(argv[8]) : 1;
  int unif = (argc > 9) ? atoi(argv[9]) : 1;
  int seed = (argc > 10) ? atoi(argv[10]) : 1493659;

  struct Params params;
  params.mode = mode;
  params.seed = seed;
  params.personCount = personCount;
  params.itemCount = itemCount;
  params.iterations = iterations;
  params.burnin = burnin;
  params.batches = batches;
  params.mu = mu;
  params.var = var;
  params.unif = unif;

  if (mode == "mpi" || mode == "cuda-mpi") {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    params.rank = rank;
    params.size = size;
  }

  if (argc == 2) {
    array<int, 12> personCounts = { 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 2000000, 5000000 };
    array<int, 4> itemCounts = { 20, 50, 100, 200 };
    for (size_t i = 0; i < personCounts.size(); ++i ) {
      params.personCount = personCounts[i];
      for (size_t j = 0; j < itemCounts.size(); ++j ) {
        params.itemCount = itemCounts[j];
        for (unsigned int k = 1; k <= 3; ++k) {
          if (mode == "cuda" && params.personCount == 5000000 && params.itemCount == 200) {
            continue;
          }
          if ((mode == "serial" || mode == "mpi") && k > 1) {
            continue;
          }
          if ((mode == "serial" || mode == "cuda") || ((mode == "mpi" || mode == "cuda-mpi") && params.rank == ROOT)) {
            cout << "run: " << k << endl;
          }
          run(params);
          if ((mode == "serial" || mode == "cuda") || ((mode == "mpi" || mode == "cuda-mpi") && params.rank == ROOT)) {
            cout << endl;
          }
        }
      }
    }
  } else {
    run(params);
  }

  if (mode == "mpi" || mode == "cuda-mpi") {
    MPI_Finalize();
  }

  return 0;
}
