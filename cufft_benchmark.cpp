#include "FFTSolver.cuh"
#include "Generator.h"
#include "Parameters.h"

int main(int argc, const char* argv[])
{
  Parameters& params = Parameters::getInstance();

  params.init(argc, argv);
  params.printSetup();

  Generator g;
  g.generate();

  FFTSolver* solver = FFTSolver::getSolver();
  const void* input = g.getData();

  solver->compute(input);

  delete solver;

  return 0;
}