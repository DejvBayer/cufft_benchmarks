#include <iostream>

#include <cufft.h>

#include "gpu_utils.cuh"
#include "Parameters.h"

const std::string Parameters::sUsage =
    "Usage: cufft_benchmark <nGPUs> <type> <nDims> <nX> <nY> [<nZ>]";

const std::map<std::pair<Dimensions, FFTType>, std::function<size_t(const dim3&)>> Parameters::sInputDataLenghtTable = 
{
  { {Dimensions::d2, FFTType::C2C}, [](const dim3& dim) { return dim.x * dim.y; }                   },
  { {Dimensions::d2, FFTType::C2R}, [](const dim3& dim) { return dim.x * (dim.y / 2 + 1); }         },
  { {Dimensions::d2, FFTType::R2C}, [](const dim3& dim) { return dim.x * dim.y; }                   },
  { {Dimensions::d3, FFTType::C2C}, [](const dim3& dim) { return dim.x * dim.y * dim.z; }           },
  { {Dimensions::d3, FFTType::C2R}, [](const dim3& dim) { return dim.x * dim.y * (dim.z / 2 + 1); } },
  { {Dimensions::d3, FFTType::R2C}, [](const dim3& dim) { return dim.x * dim.y * dim.z; }           },
};

const std::map<std::pair<Dimensions, FFTType>, std::function<size_t(const dim3&)>> Parameters::sOutputDataLenghtTable = 
{
  { {Dimensions::d2, FFTType::C2C}, [](const dim3& dim) { return dim.x * dim.y; }                   },
  { {Dimensions::d2, FFTType::C2R}, [](const dim3& dim) { return dim.x * dim.y; }                   },
  { {Dimensions::d2, FFTType::R2C}, [](const dim3& dim) { return dim.x * (dim.y / 2 + 1); }         },
  { {Dimensions::d3, FFTType::C2C}, [](const dim3& dim) { return dim.x * dim.y * dim.z; }           },
  { {Dimensions::d3, FFTType::C2R}, [](const dim3& dim) { return dim.x * dim.y * dim.z; }           },
  { {Dimensions::d3, FFTType::R2C}, [](const dim3& dim) { return dim.x * dim.y * (dim.z / 2 + 1); } },
};

Parameters& Parameters::getInstance()
{
  static Parameters p;
  return p;
}

Parameters::Parameters()
: mNGPUs(1u), mNDims(Dimensions::d2), mDims(dim3(0u, 0u, 0u))
{}

void Parameters::init(int argc, const char* argv[])
{
  if (argc < 6 || argc > 7)
  {
    std::cout << sUsage << std::endl;
    exit(0);
  }

  Parameters& p = getInstance();

  p.mNGPUs = static_cast<uint>(std::stoul(argv[1]));
  p.mType  = static_cast<FFTType>(std::stoul(argv[2]));
  p.mNDims = static_cast<Dimensions>(std::stoul(argv[3]));
  p.mDims.x = static_cast<uint>(std::stoul(argv[4]));
  p.mDims.y = static_cast<uint>(std::stoul(argv[5]));

  if(p.mNDims == Dimensions::d2)
  {    
    p.mDims.z = 1u;
  }
  else if (argc == 7 && p.mNDims == Dimensions::d3)
  {
    p.mDims.z = static_cast<uint>(std::stoul(argv[6]));
  }
  else
  {
    std::cerr << "Invalid combination of parameters" << std::endl;
    exit(1);
  }
}

void Parameters::printSetup()
{
  std::cout << "\tBenchmark Setup"
            << "\nnGPUs: " << mNGPUs
            << "\nType: "  << static_cast<uint>(mType)
            << "\nnDims: " << static_cast<uint>(mNDims)
            << "\ndims: "  << mDims
            << std::endl;
}

cufftType Parameters::getCufftType()
{
  switch (mType)
  {
  case FFTType::C2C:
    return CUFFT_C2C;
  case FFTType::C2R:
    return CUFFT_C2R;
  case FFTType::R2C:
    return CUFFT_R2C;
  }
}

size_t Parameters::getInputTypeSize()
{
  switch (mType)
  {
  case FFTType::C2C:
  case FFTType::C2R:
    return sizeof(cufftComplex);
  case FFTType::R2C:
    return sizeof(cufftReal);
  }
}

size_t Parameters::getOutputTypeSize()
{
  switch (mType)
  {
  case FFTType::C2C:
  case FFTType::R2C:
    return sizeof(cufftComplex);
  case FFTType::C2R:
    return sizeof(cufftReal);
  }
}
