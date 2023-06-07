#include <random>
#include <iostream>

#include <boost/program_options.hpp>
#include <sciplot/sciplot.hpp>

#include "gumbel.hpp"

namespace opt = boost::program_options;

const double minRange = -5;
const double maxRange = 20;
// std::mt19937 gen;

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
  plot.xrange(minRange, maxRange);

  plot.legend()
      .atOutsideBottom()
      .displayHorizontal()
      .displayExpandWidthBy(2);
}

void plotGumbelCDF(sciplot::Plot2D &plot, double mu = 0, double beta = 1)
{
  setupCanvas(plot);

  sciplot::Vec x = sciplot::linspace(minRange, maxRange, 200);
  plot.drawCurve(x, gumbelCDF(x, mu, beta));

  savePlot(plot, "Gumbel CDF", "example-plot-gumbel-cdf.pdf");
}

void plotGumbelPDF(sciplot::Plot2D &plot, double mu, double beta)
{
  setupCanvas(plot);

  sciplot::Vec x = sciplot::linspace(minRange, maxRange, 200);
  plot.drawCurve(x, gumbelPDF(x, mu, beta));

  savePlot(plot, "Gumbel PDF", "example-plot-gumbel-pdf.pdf");
}

void plotHistogram(sciplot::Plot2D &plot)
{
  const int N = 7;
  const int M = 1000;

  plot.xlabel("x");
  plot.ylabel("y");
  plot.xrange(0.5, N + 0.5);

  plot.legend()
      .atOutsideBottom()
      .displayHorizontal()
      .displayExpandWidthBy(2);

  std::uniform_int_distribution<> distrib(1, N);
  gen.seed(0);

  std::valarray<double> counts{1, 3, 4, 5, 2, 1, 6};
  // for (size_t i = 0; i < counts.size(); i++) counts[i] = 0;

  // for (size_t i = 0; i < M; i++)
  // {
  //   counts[distrib(gen)]++;
  // }

  std::valarray<double> p = counts / (counts.sum());
  plot.drawHistogram(p);
  savePlot(plot, "Histogram", "example-plot-histogram-pdf.pdf");

  std::valarray<double> logits = std::log(counts);
  std::valarray<double> samples(N + 1);
  for (size_t i = 0; i < samples.size(); i++)
    samples[i] = 0;

  for (size_t i = 0; i < M; i++)
  {
    const int sample = sampleWithGumbelNoise(logits);
    samples[sample]++;
  }

  p = samples / ((double)M);
  plot.drawHistogram(p);
  savePlot(plot, "Histogram - sampleWithGumbelNoise", "example-plot-histogram-gumbel-noise-pdf.pdf");
}

int main(int argc, char *argv[])
{
  opt::options_description desc("Plot Gumbel CDF/PDF");

  desc.add_options()("mu", opt::value<double>(), "$mu$")("beta", opt::value<double>(), "$beta$")("help", "show this help message");

  opt::variables_map vm;
  opt::store(opt::parse_command_line(argc, argv, desc), vm);
  opt::notify(vm);

  if (vm.size() == 0 || vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  // Create a Plot object
  sciplot::Plot2D plot;

  const double mu = vm["mu"].as<double>();
  const double beta = vm["beta"].as<double>();

  // plotGumbelCDF(plot, mu, beta);
  // plotGumbelPDF(plot, mu, beta);

  // 1. Plot bipartite
  // 2. Sample bipartite match
  // => calculate Z, theta
  // 3. Sample bipartite match w algorithm
  // 4. Accept/reject ratio
  // 5. Comparing

  plotHistogram(plot);

  std::cout << "Done plotting" << std::endl;
  return 0;
}
