#include "Generator.h"
#include "gpu_utils.cuh"

Generator::Generator()
: mParams(Parameters::getInstance()), mGenerator(nullptr), mDevMem(nullptr), mHostMem(nullptr)
{
  checkCuRandError(curandCreateGenerator(&mGenerator, CURAND_RNG_PSEUDO_DEFAULT));
  checkCuRandError(curandSetPseudoRandomGeneratorSeed(mGenerator, sSeed));
}

void Generator::generate()
{
  if (mDevMem == nullptr)
  {
    size_t inputSize = mParams.getInputSize();
  
    checkCudaError(cudaMalloc(&mDevMem, inputSize));
    checkCudaError(cudaHostAlloc(&mHostMem, inputSize, cudaHostAllocMapped));

    checkCuRandError(curandGenerateUniform(mGenerator, reinterpret_cast<float*>(mDevMem), mParams.getInputN()));

    checkCudaError(cudaMemcpy(mHostMem, mDevMem, inputSize, cudaMemcpyDeviceToHost));    
  }
}

Generator::~Generator()
{
  curandDestroyGenerator(mGenerator);
  cudaFree(mDevMem);
  cudaFreeHost(mHostMem);
}
