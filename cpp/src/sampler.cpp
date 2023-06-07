#include <random>
#include <iostream>
#include <vector>
#include <valarray>
#include <limits>

#include "rectangular_lsap.h"

std::mt19937 gen;

template <typename T>
using List = std::vector<T>;

template <typename T>
using Vector = std::valarray<T>;

Vector<double>
get_uniform_int_weight_matrix(int size){
  Vector<double> weights (0.0, size*size);
  std::uniform_int_distribution<> distr(1, size);
  for (size_t row = 0; row < size; row++)
    for (size_t col = 0; col < size; col++)
      weights[row * size + col] = distr(gen);
  return weights;
}


/**
 * sample_from - a vector of N size where i-th item is 1 iff col i is to be added
*/
template <typename T>
Vector<T> getSubWeightMatrix(const Vector<T> &matrix, const int N, const Vector<int> &sample_from, const int j) {
  const int s = N - j;
  Vector<double> sub_matrix(0.0, s*s);
  for (size_t row = j; row < N; row++) {
    int k = 0;
    for (size_t col = 0; col < N; col++) {
      if (sample_from[col]) {
        sub_matrix[s*(row-j) + k] = matrix[N * row + col];
        k++;
      }
    }
  }
  return sub_matrix;
}


double
get_upper_bound(const Vector<double> &log_weights, const int N) {
  const int M = 50;
  double sum = 0;


  std::uniform_real_distribution<> uniform(0,1);
  Vector<double> uniformNoise(0.0, log_weights.size());

  std::cout << "start for loop ..." << std::endl;
  for (size_t i = 0; i < M; i++) {
    std::cout << "\ti=" << i << std::endl;
    std::cout << "\tsample uniform noise" << std::endl;

    for (size_t i = 0; i < uniformNoise.size(); i++)
      uniformNoise[i] = uniform(gen);

    std::cout << "\ttransform to gumbel noise" << std::endl;
    const Vector<double> gumbelNoise = std::log(-std::log(uniformNoise));
    std::cout << "\tadding noise" << std::endl;
    auto perturbed_weights = log_weights + gumbelNoise;

    double* costs = (double*) malloc(sizeof(double)*N*N);
    int64_t* left = (int64_t*) malloc(sizeof(int64_t)*N);
    int64_t* right = (int64_t*) malloc(sizeof(int64_t)*N);

    for (size_t row = 0; row < N; row++) {
      for (size_t col = 0; col < N; col++) {
        const int index = N * row + col;
        costs[index] = log_weights[index] + gumbelNoise[index];
      }
    }

    std::cout << "\tsolve_rectangular_linear_sum_assignment (round=" << i;
    solve_rectangular_linear_sum_assignment(N, N, costs, true, left, right);

    double cost = 0;
    for (size_t i = 0; i < N; i++)
      cost += costs[N*i + right[i]];

    std::cout << ") = " << cost << std::endl;
    sum += cost;

    free(costs);
    free(left);
    free(right);
  }

  return sum / M;
}


List<double>
calculate_distribution(const Vector<double> &log_weights, const int N, const Vector<int> samples, const int j) {
  std::cout << "Called calculate_distribution j=" << j << "." << std::endl;

  std::cout << "Samples: [ ";
  for (size_t i = 0; i < N; i++) {
    if (samples[i])
      std::cout << i << " ";
  }
  std::cout << "]" << std::endl;


  List<double> P(N+1, 0.0);
  std::cout << "P: ";
  for (size_t i = 0; i < P.size(); i++) {
    std::cout << P[i] << " ";
  }
  std::cout << std::endl;

  Vector<int> sample_from(samples);
  for (size_t i = 0; i < N; i++)
      sample_from[i] = !samples[i];

  std::cout << "sample from " << sample_from.size() << " items" << std::endl;
  std::cout << "getSubWeightMatrix ..." << std::endl;

  const auto sub_weight_matrix = getSubWeightMatrix(log_weights, N, sample_from, j);
  std::cout << "sub weight matrix size " << sub_weight_matrix.size() << std::endl;

  std::cout << "get_upper_bound ..." << std::endl;
  const double prev_bound = get_upper_bound(sub_weight_matrix, N);
  std::cout << "Calculated upper bound: " << prev_bound << std::endl;

  double sum = 0;
  for (size_t v = 0; v < N; v++) {
    if (!sample_from[v])
      continue;

    std::cout << "--Next item is to try is " << v << std::endl;

    Vector<int> from(samples);
    for (size_t i = 0; i < N; i++)
        from[i] = !samples[i];
    from[v] = 0;

    auto weight_v = log_weights[N*j + v];
    std::cout << "Its log weight is " << weight_v << std::endl;

    const auto submatrix = getSubWeightMatrix(log_weights, N, sample_from, j+1);
    std::cout << "Its sub_weight_matrix is " << submatrix.size() << std::endl;
    auto upper_bound = get_upper_bound(submatrix, N);
    std::cout << "its upper bound is: " << upper_bound << std::endl;
    const auto p = std::exp(upper_bound + weight_v - prev_bound);
    std::cout << "its probability is: " << p << std::endl;
    P[v] = p;
    std::cout << "P[" << v << "] is " << P[v] << std::endl;
    sum += p;
    std::cout << "Acc sum is " << sum << std::endl;
  }

  std::cout << "Done calc of P " << sum << std::endl;

  P[N] = 1 - sum;
  return P;
}


Vector<int> bipartite_matching_sampler(const Vector<double> &W, const int N) {
  std::cout << "Called bipartite_matching_sampler" << std::endl;

  Vector<int> samples(0, N);
  long int rejections = 0;
  Vector<double> log_weights = std::log(W);
  int reject_symbol = N;

  int j = 0;
  while (j < N) {
    std::cout << "===while..." << std::endl;
    List<double> P = calculate_distribution(log_weights, N, samples, j);
    std::discrete_distribution<> d(P.begin(), P.end());

    std::cout << "sampling..." << std::endl;
    int sampleId = (P[P.size() - 1] < 0)? reject_symbol : d(gen);
    std::cout << "sampled " << sampleId << std::endl;
    if (sampleId == reject_symbol) {
      rejections += 1;
      j = 0;
      for (size_t i = 0; i < N; i++)
        samples[i] = 0;
    }
    else {
      j += 1;
      samples[sampleId] = 1;
    }
  }

  return samples;
}


template <typename T>
void print_matrix(const Vector<T> &matrix, const int N) {
  for (size_t row = 0; row < N; row++) {
    for (size_t col = 0; col < N; col++) {
      std::cout << matrix[N*row + col] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


int main(int argc, char *argv[]) {
  int N = 7;
  const auto weights = get_uniform_int_weight_matrix(N);
  const auto log_weights = std::log(weights);
  // auto log_weights = get_log_weights(weights);

  std::cout << "Random weight matrix" << std::endl;
  print_matrix(weights, N);

  // const Vector<int> sample_from = {0,0,1,1,0,1,1};
  // const int n = sample_from.sum();
  // const auto submatrix = getSubWeightMatrix(weights, N, sample_from, N-n);
  // std::cout << "Random sub matrix" << std::endl;
  // print_matrix(submatrix, n);

  std::cout << "Sampled items" << std::endl;
  auto samples = bipartite_matching_sampler(weights, N);
  for (size_t i = 0; i < N; i++) {
      std::cout << samples[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
