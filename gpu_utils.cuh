#ifndef __GPU_UTILS_CUH__
#define __GPU_UTILS_CUH__

#include <iostream>

#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaError(cudaCall)                                 \
  do                                                             \
  {                                                              \
    cudaError_t __error = cudaCall;                              \
    if (__error != cudaSuccess)                                  \
    {                                                            \
      std::cout << "CUDA Error: " << cudaGetErrorString(__error) \
                << " (File " << __FILE__                         \
                << ", line " << __LINE__ << ")"                  \
                << std::endl;                                    \
    }                                                            \
  } while(0)

#define checkCuRandError(curandCall)            \
  do                                            \
  {                                             \
    curandStatus_t __status = curandCall;       \
    if (__status != CURAND_STATUS_SUCCESS)      \
    {                                           \
      std::cout << "curand Error: " << __status \
                << " (File " << __FILE__        \
                << ", line " << __LINE__ << ")" \
                << std::endl;                   \
    }                                           \
  } while(0)

#define checkCufftError(cufftCall)                                       \
  do                                                                     \
  {                                                                      \
    cufftResult __result = cufftCall;                                    \
    if (__result != CUFFT_SUCCESS)                                       \
    {                                                                    \
      std::cout << "cufft Error: 0x" << std::hex << __result << std::dec \
                << " (File " << __FILE__                                 \
                << ", line " << __LINE__ << ")"                          \
                << std::endl;                                            \
    }                                                                    \
  } while(0)

std::ostream& operator<<(std::ostream& os, const dim3& dims);

#endif  //  __GPU_UTILS_CUH__
