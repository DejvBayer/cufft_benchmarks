#include <chrono>
#include <iostream>
#include <string>

#include <cufft.h>
#include <curand.h>
#include <cuda_runtime_api.h>

enum class FFTDims : uint {
  d1 = 1,
  d2 = 2,
  d3 = 3,
};

struct Parameters
{
  FFTDims dims;
  dim3 sizes;
};

Parameters parseArgs(int argc, const char* argv[])
{
  if (argc != 5)
  {
    std::cerr << "Invalid numer of arguments" << std::endl;
    exit(1);
  }

  Parameters p;

  p.dims = static_cast<FFTDims>(std::stoi(argv[1]));
  p.sizes = {
    static_cast<uint>(std::stoi(argv[2])),
    static_cast<uint>(std::stoi(argv[3])),
    static_cast<uint>(std::stoi(argv[4]))
  };

  return p;
}

size_t getRSize(dim3 sizes)
{
  return sizes.x * sizes.y * sizes.z * sizeof(cufftReal);
}

size_t getCSize(dim3 sizes)
{
  return (sizes.x * sizes.y * sizes.z / 2 + 1) * sizeof(cufftComplex);
}

cufftHandle getPlan(Parameters &params)
{
  cufftHandle plan;

  switch (params.dims)
  {
  case FFTDims::d1:
    cufftPlan1d(&plan, params.sizes.x, CUFFT_R2C, 1);
    break;
  case FFTDims::d2:
    cufftPlan2d(&plan, params.sizes.x, params.sizes.y, CUFFT_R2C);
    break;
  case FFTDims::d3:
    cufftPlan3d(&plan, params.sizes.x, params.sizes.y, params.sizes.z, CUFFT_R2C);
    break;
  }

  return plan;
}

void initData(cufftReal* dRs, dim3 sizes)
{
  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator, 0ull);

  curandGenerateUniform(generator, dRs, sizes.x * sizes.y * sizes.z);
}

void checkOutput(cufftComplex* hCs)
{

}

int main(int argc, const char* argv[])
{
  Parameters params = parseArgs(argc, argv);

  //  Allocate memory

  cufftReal* hRs;
  cufftComplex* hCs;

  cufftReal* dRs;
  cufftComplex* dCs;

  cudaHostAlloc(&hRs, getRSize(params.sizes), cudaHostAllocMapped);
  cudaHostAlloc(&hCs, getCSize(params.sizes), cudaHostAllocMapped);

  cudaMalloc(&dRs,  getRSize(params.sizes));
  cudaMalloc(&dCs, getCSize(params.sizes));

  //  Create FFT plan

  cufftHandle plan = getPlan(params);

  //  Initialize data

  initData(dRs, params.sizes);

  //  Copy initialized data from host to device

  cudaMemcpy(hRs, dRs, getRSize(params.sizes), cudaMemcpyDeviceToHost);
  

  //  Execute computation

  auto start = std::chrono::high_resolution_clock::now();

  cufftExecR2C(plan, dRs, dCs);
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);

  std::cout << duration.count() << " s" << std::endl;

  //  Copy results from device to host

  cudaMemcpy(hCs, dCs, getCSize(params.sizes), cudaMemcpyDeviceToHost);

  //  Verify results

  checkOutput(dCs);

  //  Clean up

  cufftDestroy(plan);
  cudaFree(dRs);
  cudaFree(dCs);
  cudaFreeHost(hRs);
  cudaFreeHost(hCs);

  return 0;
}
