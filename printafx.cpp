#include "sampler.hpp"

void Sampler::printMatrix(double* m, int sz1, int sz2, string name, int cw) {
  cout << name << endl;
  for (int i = 0; i < sz1; ++i) {
    for (int j = 0; j < sz2; ++j) {
      cout << setw(cw) << m[i * sz2 + j];
      if (j < sz2 - 1) {
        cout << ",";
      }
    }
    cout << endl;
  }
  cout << endl;
}

void Sampler::printArray(double* a, int sz, string name, int cw) {
  cout << name << endl;
  for (int i = 0; i < sz; ++i) {
    cout << setw(cw) << a[i] << endl;
  }
  cout << endl;
}

void Sampler::printScores(int* y, int personCount, int itemCount) {
  cout << "y" << endl;
  for (int p = 0; p < personCount; ++p) {
    for (int i = 0; i < itemCount; ++i) {
      cout << y[p * itemCount + i];
      if (i < itemCount - 1) {
        cout << ",";
      }
    }
    cout << endl;
  }
  cout << endl;
}

void Sampler::printItems(double* as, double* gs, double* items, int itemCount) {
  cout << "items" << endl;
  cout << setw(16) << "as" << "," << setw(16) << "estA" << "," << setw(16) << "seA" << ",";
  cout << setw(16) << "gs" << "," << setw(16) << "estG" << "," << setw(16) << "seG" << endl;
  for (int i = 0; i < itemCount; ++i) {
    cout << setw(16) << as[i] << "," << setw(16) << items[i * 4 + 0] << "," << setw(16) << items[i * 4 + 1] << ",";
    cout << setw(16) << gs[i] << "," << setw(16) << items[i * 4 + 2] << "," << setw(16) << items[i * 4 + 3] << endl;
  }
  cout << endl;
}

void Sampler::printPersons(double* ths, double* persons, int personCount) {
  cout << "persons" << endl;
  cout << setw(16) << "ths" << "," << setw(16) << "estTH" << "," << setw(16) << "seTH" << endl;
  for (int p = 0; p < personCount; ++p) {
    cout << setw(16) << ths[p] << "," << setw(16) << persons[p * 2 + 0] << "," << setw(16) << persons[p * 2 + 1] << endl;
  }
  cout << endl;
}
