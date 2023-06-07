std::mt19937 gen;

std::valarray<double>
gumbelCDF(const std::valarray<double>& x, const double mu, const double beta) {
    return std::exp(-std::exp(-(x-mu)/beta));
}

std::valarray<double>
gumbelPDF(const std::valarray<double>& x, const double mu, const double beta) {
  const std::valarray<double> z = (x - mu) /beta;
  return (1.0)/beta * std::exp(-(z+std::exp(-z)));
}

std::valarray<double>
gumbelInverseCDF(const std::valarray<double>& uniformNoise, const double mu=0, const double beta=1) {
    return mu - beta * std::log(-std::log(uniformNoise));
}

std::valarray<double>
sampleUniformNoise(const int M) {
  std::uniform_real_distribution<> uniform;
  std::valarray<double> uniformNoise(M);

  for (size_t i = 0; i < M; i++)
    uniformNoise[i] = uniform(gen);

  return uniformNoise;
}

int argmax(const std::valarray<double>& values) {
  if (values.size() == 0) return -1;

  int argmax = 0;
  double max = values[0];

  for (size_t i = 1; i < values.size(); i++) {
    if (values[i] > max) {
      max = values[i];
      argmax = i;
    }
  }

  return argmax;
}

int sampleWithGumbelNoise(const std::valarray<double>& logits) {
  const std::valarray<double> gumbelNoise = gumbelInverseCDF(sampleUniformNoise(logits.size()));
  const std::valarray<double> pl = logits + gumbelNoise;
  return argmax(logits + gumbelNoise);
}

std::valarray<double> sampleGumbelNoise(const int size) {
  return gumbelInverseCDF(sampleUniformNoise(size));
}