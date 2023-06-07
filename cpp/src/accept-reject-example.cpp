#include <random>
#include <iostream>
#include <tuple>

#include <sciplot/sciplot.hpp>

const double minRange = -3;
const double maxRange = 3;
std::mt19937 gen;

const double M = 3;

template <typename T>
T g(T &x)
{
  return std::exp(-std::pow(x, 2) / 2);
}

template <typename T>
T fStar(T &x)
{
  const auto Mg = std::exp(-std::pow(x, 2) / 2);
  const auto cs2 = std::pow(std::cos(x), 2);
  const auto sn2 = std::pow(std::sin(4 * x), 2);
  return Mg * (1 + 2 * cs2 * sn2);
}

// std::valarray<double> (*f)(std::valarray<double>),
// std::valarray<double> (*g)(std::valarray<double>),
// double M
std::tuple<std::valarray<double>, std::valarray<double>>
sample(int N)
{
  std::uniform_real_distribution<double> uniform;
  std::normal_distribution<double> normal;
  std::valarray<double> Z(N);
  std::valarray<double> U(N);
  for (size_t i = 0; i < N; i++)
  {
    while (true)
    {
      double u = uniform(gen);
      double z = normal(gen);
      if (u * (M * g(z)) < fStar(z))
      {
        U[i] = u;
        Z[i] = z;
        break;
      }
    }
  }
  const std::tuple t(U, Z);
  return t;
}

void savePlot(sciplot::Plot2D &plot, const std::string &title, const std::string &filename)
{
  sciplot::Figure fig = {{plot}};
  sciplot::Canvas canvas = {{fig}};
  canvas.title(title);
  canvas.save(filename);
  canvas.show();
  plot.clear();
}

void setupCanvas(sciplot::Plot2D &plot)
{
  plot.xlabel("x");
  plot.ylabel("y");
  // plot.xrange(minRange, maxRange);

  plot.legend()
      .atOutsideBottom()
      .displayHorizontal()
      .displayExpandWidthBy(2);
}

void plot()
{
  sciplot::Plot2D plot;
  setupCanvas(plot);
  std::valarray<double> x = sciplot::linspace(minRange, maxRange, 200);
  plot.drawCurve(x, fStar(x));
  plot.drawCurve(x, M * g(x));

  const int N = 100;
  const auto t = sample(N);
  const auto [z, fz] = t;
  // std::valarray<double> p = counts / (counts.sum());
  plot.drawPoints(z, fz);

  savePlot(plot, "f* & Mg & samples", "f-star-plot.pdf");
}

int main(int argc, char *argv[])
{
  plot();

  std::cout << "Done plotting" << std::endl;
  return 0;
}
