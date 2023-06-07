#include <iostream>
#include <boost/program_options.hpp>
namespace opt = boost::program_options;

template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
long double permanent(std::vector<T> v)
{
  long long n = v.size();
  return n;
}

int main(int argc, char *argv[])
{
  opt::options_description desc("All options");

  desc.add_options()("help", "show this help message");

  opt::variables_map vm;
  opt::store(opt::parse_command_line(argc, argv, desc), vm);
  opt::notify(vm);

  const Matrix<double> v = {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};

  if (vm.size() == 0 || vm.count("help"))
  {
    std::cout << desc << "\n";
    std::cout << "permanent(x): " << permanent(v) << "\n";
    return 1;
  }

  return 0;
}
