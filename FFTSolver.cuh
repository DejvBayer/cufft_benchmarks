#ifndef __FFT_SOLVER_CUH__
#define __FFT_SOLVER_CUH__

#include <cufft.h>
#include <cufftXt.h>

#include "Parameters.h"

class FFTSolver
{
protected:
  Parameters& mParams;

public:
  static FFTSolver* getSolver();

public:
  FFTSolver();
  virtual void compute(const void* input);
};

class cufftSolver : public FFTSolver
{
  cufftHandle mPlan;
  void*       mInput;
  void*       mOutput;

public:
  cufftSolver();
  void compute(const void* input);
  ~cufftSolver();
};

class cufftXtSolver : public FFTSolver
{
  cufftHandle    mPlan;
  cudaLibXtDesc* mInput;
  cudaLibXtDesc* mOutput;

public:
  cufftXtSolver();
  void compute(const void* input);
  ~cufftXtSolver();
};

#endif  //  __FFT_SOLVER_CUH__
