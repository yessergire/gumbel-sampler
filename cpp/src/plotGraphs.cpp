#include <iostream>                  // for std::cout
#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include <boost/program_options.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

namespace opt = boost::program_options;

int main(int argc, char *argv[])
{
  opt::options_description desc("All options");

  desc.add_options()
    ("help", "show this help message");

  opt::variables_map vm;
  opt::store(opt::parse_command_line(argc, argv, desc), vm);
  opt::notify(vm);

  enum { A, B, C, D, E, N };
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph; // create a typedef for the Graph type
  Graph g(N);

  typedef std::pair<int, int> Edge;

  Edge edge_array[] =
  { Edge(A,B), Edge(C,D), Edge(B,E),  };

  const int num_edges = sizeof(edge_array)/sizeof(edge_array[0]);

  // add the edges to the graph object
  for (int i = 0; i < num_edges; ++i) {
    add_edge(edge_array[i].first, edge_array[i].second, g);
  }

  std::stringstream dotCode;

  boost::write_graphviz(dotCode, g);

  std::cout << dotCode.str() << "\n";

  if (vm.size() == 0 || vm.count("help")) {
    //std::cout << desc << "\n";
    return 1;
  }

  return 0;
}
