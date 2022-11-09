#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cufftXt.h>
#include <curand.h>
#include <cuda_runtime_api.h>

#define checkCudaError(error)                                      \
  if ((error) != cudaSuccess)                                      \
  {                                                                \
    std::cout << "CUDA Error: " << cudaGetLastError()         \
              << " (File" << __FILE__ << ", line" << __LINE__ << ")" << std::endl; \
  }
#define checkCuFFTError(error)                                                 \
  if ((error) != CUFFT_SUCCESS)                                                \
  {                                                                            \
    std::cout << "cuFFT Error: " << std::hex << static_cast<int>(error) << std::dec << " (File" << __FILE__ << ", line" << __LINE__ << ")" << std::endl; \
  }

enum class FFTDims : uint {
  d1 = 1,
  d2 = 2,
  d3 = 3,
};

struct Parameters
{
  uint nGpus;
  FFTDims dims;
  dim3 sizes;

  void printSetup()
  {
    std::cout << "nGPUs: "     << nGpus
              << "\ndims: "    << static_cast<int>(dims)
              << "\nsizes.x: " << sizes.x
              << "\nsizes.y: " << sizes.y
              << "\nsizes.z: " << sizes.z
              <<  std::endl;
  }
};

Parameters parseArgs(int argc, const char* argv[])
{
  if (argc != 6)
  {
    std::cerr << "Invalid number of arguments" << std::endl;
    exit(1);
  }

  Parameters p;

  p.nGpus = static_cast<uint>(std::stoi(argv[1]));
  p.dims = static_cast<FFTDims>(std::stoi(argv[2]));
  p.sizes = {
    static_cast<uint>(std::stoi(argv[3])),
    static_cast<uint>(std::stoi(argv[4])),
    static_cast<uint>(std::stoi(argv[5]))
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

void init(const Parameters &p)
{
  int gpuCount;
  checkCudaError(cudaGetDeviceCount(&gpuCount));

  if (gpuCount < p.nGpus)
  {
    std::cerr << "Not enough gpus" << std::endl;
    exit(1);
  }

  std::cout << "GPUs: " << p.nGpus << "/" << gpuCount << std::endl;
}

void initData(cufftReal* dRs, dim3 sizes)
{
  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator, 0ull);

  curandGenerateUniform(generator, dRs, sizes.x * sizes.y * sizes.z);
}

cufftHandle getPlan(const Parameters &params)
{
  std::vector<int> whichGPUs(params.nGpus);
  std::vector<size_t> workSize(params.nGpus);
  std::iota(whichGPUs.begin(), whichGPUs.end(), 0);

  cufftHandle plan;
  checkCuFFTError(cufftCreate(&plan));

  checkCuFFTError(cufftXtSetGPUs(plan, params.nGpus, whichGPUs.data()));

  switch (params.dims)
  {
  case FFTDims::d1:
    checkCuFFTError(cufftMakePlan1d(plan, params.sizes.x, CUFFT_R2C, 1, workSize.data()));
    break;
  case FFTDims::d2:
    checkCuFFTError(cufftMakePlan2d(plan, params.sizes.x, params.sizes.y, CUFFT_R2C, workSize.data()));
    break;
  case FFTDims::d3:
    checkCuFFTError(cufftMakePlan3d(plan, params.sizes.x, params.sizes.y, params.sizes.z, CUFFT_R2C, workSize.data()));
    break;
  }

  std::cout << "Work sizes[" << workSize.size() << "]: ";
  for (auto i : workSize)
  {
    std::cout << i << ", ";
  }
  std::cout << std::endl;

  return plan;
}

void checkOutput(cufftComplex* hCs)
{

}

int main(int argc, const char* argv[])
{
  Parameters params = parseArgs(argc, argv);
  params.printSetup();

  init(params);

  //  Allocate memory

  cufftReal* hRs;
  cufftComplex* hCs;

  cufftReal* dRs;
  cufftComplex* dCs;

  checkCudaError(cudaHostAlloc(&hRs, getRSize(params.sizes), cudaHostAllocMapped));
  checkCudaError(cudaHostAlloc(&hCs, getCSize(params.sizes), cudaHostAllocMapped));

  checkCudaError(cudaMalloc(&dRs, getRSize(params.sizes)));
  checkCudaError(cudaMalloc(&dCs, getCSize(params.sizes)));

  //  Create FFT plan

  cufftHandle plan = getPlan(params);

  cudaLibXtDesc* dRsXt;
  cudaLibXtDesc* dCsXt;

  checkCuFFTError(cufftXtMalloc(plan, &dRsXt, CUFFT_XT_FORMAT_INPUT));
  checkCuFFTError(cufftXtMalloc(plan, &dCsXt, CUFFT_XT_FORMAT_OUTPUT));

  //  Initialize data

  initData(dRs, params.sizes);

  //  Copy initialized data from host to device

  checkCudaError(cudaMemcpy(hRs, dRs, getRSize(params.sizes), cudaMemcpyDeviceToHost));
  checkCuFFTError(cufftXtMemcpy(plan, dRsXt, hRs, CUFFT_COPY_HOST_TO_DEVICE));

  std::cout << "Data copied!" << std::endl;

  //  Execute computation

  auto start = std::chrono::high_resolution_clock::now();

  checkCuFFTError(cufftXtExecDescriptorR2C(plan, dRsXt, dCsXt));
  checkCudaError(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);

  std::cout << duration.count() << " s" << std::endl;

  //  Copy results from device to host

  // checkCudaError(cudaMemcpy(hCs, dCs, getCSize(params.sizes), cudaMemcpyDeviceToHost));
  checkCuFFTError(cufftXtMemcpy(plan, hCs, dCsXt, CUFFT_COPY_DEVICE_TO_HOST));

  //  Verify results

  checkOutput(dCs);

  //  Clean up

  checkCuFFTError(cufftXtFree(dRsXt));
  checkCuFFTError(cufftXtFree(dCsXt));
  checkCuFFTError(cufftDestroy(plan));
  checkCudaError(cudaFree(dRs));
  checkCudaError(cudaFree(dCs));
  checkCudaError(cudaFreeHost(hRs));
  checkCudaError(cudaFreeHost(hCs));

  return 0;
}
