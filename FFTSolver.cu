#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cufft.h>
#include <cufftXt.h>

#include "gpu_utils.cuh"
#include "FFTSolver.cuh"

FFTSolver* FFTSolver::getSolver()
{
  Parameters& params = Parameters::getInstance();

  return params.getNGPUs() == 1
    ? reinterpret_cast<FFTSolver*>(new cufftSolver())
    : reinterpret_cast<FFTSolver*>(new cufftXtSolver());
}

FFTSolver::FFTSolver()
: mParams(Parameters::getInstance())
{}

void FFTSolver::compute(const void* input)
{}

cufftSolver::cufftSolver()
: FFTSolver(), mPlan(0), mInput(nullptr), mOutput(nullptr)
{
  switch (mParams.getNDims())
  {
  case Dimensions::d2:
    checkCufftError(cufftPlan2d(&mPlan, mParams.getXDim(), mParams.getYDim(),
                    mParams.getCufftType()));
    break;
  case Dimensions::d3:
    checkCufftError(cufftPlan3d(&mPlan, mParams.getXDim(), mParams.getYDim(),
                    mParams.getZDim(), mParams.getCufftType()));
    break;
  }

  checkCudaError(cudaMalloc(&mInput, mParams.getInputSize()));
  checkCudaError(cudaMalloc(&mOutput, mParams.getOutputSize()));
}

void cufftSolver::compute(const void* input)
{
  checkCudaError(cudaMemcpy(mInput, input, mParams.getInputSize(), cudaMemcpyHostToDevice));

  auto start = std::chrono::high_resolution_clock::now();

  switch (mParams.getFFTType())
  {
  case FFTType::C2C:
    checkCufftError(cufftExecC2C(mPlan, reinterpret_cast<cufftComplex*>(mInput),
                    reinterpret_cast<cufftComplex*>(mOutput), CUFFT_FORWARD));
    break;
  case FFTType::C2R:
    checkCufftError(cufftExecC2R(mPlan, reinterpret_cast<cufftComplex*>(mInput),
                    reinterpret_cast<cufftReal*>(mOutput)));
    break;
  case FFTType::R2C:
    checkCufftError(cufftExecR2C(mPlan, reinterpret_cast<cufftReal*>(mInput),
                    reinterpret_cast<cufftComplex*>(mOutput)));
    break;
  }
  checkCudaError(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);

  std::cout << duration.count() << " s" << std::endl;
}

cufftSolver::~cufftSolver()
{
  cufftDestroy(mPlan);
  cudaFree(mInput);
  cudaFree(mOutput);
}

cufftXtSolver::cufftXtSolver()
: FFTSolver(), mPlan(0), mInput(nullptr), mOutput(nullptr)
{
  std::vector<int> whichGPUs(mParams.getNGPUs());
  std::vector<size_t> workSize(mParams.getNGPUs());

  std::iota(whichGPUs.begin(), whichGPUs.end(), 0);

  checkCufftError(cufftCreate(&mPlan));
  checkCufftError(cufftXtSetGPUs(mPlan, mParams.getNGPUs(), whichGPUs.data()));

  switch (mParams.getNDims())
  {
  case Dimensions::d2:
    checkCufftError(cufftMakePlan2d(mPlan, mParams.getXDim(), mParams.getYDim(),
                    mParams.getCufftType(), workSize.data()));
    break;
  case Dimensions::d3:
    checkCufftError(cufftMakePlan3d(mPlan, mParams.getXDim(), mParams.getYDim(),
                    mParams.getZDim(), mParams.getCufftType(), workSize.data()));
    break;
  }

  checkCufftError(cufftXtMalloc(mPlan, &mInput, CUFFT_XT_FORMAT_INPLACE));
  checkCufftError(cufftXtMalloc(mPlan, &mOutput, CUFFT_XT_FORMAT_INPLACE));
}

void cufftXtSolver::compute(const void* input)
{
  checkCufftError(cufftXtMemcpy(mPlan, mInput, const_cast<void*>(input), CUFFT_COPY_HOST_TO_DEVICE));

  auto start = std::chrono::high_resolution_clock::now();

  switch (mParams.getFFTType())
  {
  case FFTType::C2C:
    checkCufftError(cufftXtExecDescriptorC2C(mPlan, mInput, mOutput, CUFFT_FORWARD));
    break;
  case FFTType::C2R:
    checkCufftError(cufftXtExecDescriptorC2R(mPlan, mInput, mOutput));
    break;
  case FFTType::R2C:
    checkCufftError(cufftXtExecDescriptorR2C(mPlan, mInput, mOutput));
    break;
  }

  checkCudaError(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);

  std::cout << duration.count() << " s" << std::endl;
}

cufftXtSolver::~cufftXtSolver()
{
  cufftDestroy(mPlan);
  cufftXtFree(mInput);
  cufftXtFree(mOutput);
}
