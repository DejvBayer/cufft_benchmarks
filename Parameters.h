#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <functional>
#include <map>
#include <string>

#include <cufft.h>

enum class FFTType : uint {
  C2C = 0,
  C2R = 1,
  R2C = 2,
};

enum class Dimensions : uint {
  d2 = 2,
  d3 = 3,
};

class Parameters
{
  static const std::string sUsage; /// Usage message
  static const std::map<std::pair<Dimensions, FFTType>, std::function<size_t(const dim3&)>> sInputDataLenghtTable;
  static const std::map<std::pair<Dimensions, FFTType>, std::function<size_t(const dim3&)>> sOutputDataLenghtTable;

  uint       mNGPUs;
  FFTType    mType;
  Dimensions mNDims;
  dim3       mDims;

public:
  static Parameters& getInstance();

private:
  Parameters();

public:
  Parameters(const Parameters&) = delete;
  Parameters operator=(const Parameters&) = delete;

  void init(int argc, const char* argv[]);
  void printSetup();

  uint getNGPUs() { return mNGPUs; }
  FFTType getFFTType() { return mType; }
  cufftType getCufftType();
  Dimensions getNDims() { return mNDims; }
  uint getXDim() { return mDims.x; }
  uint getYDim() { return mDims.y; }
  uint getZDim() { return mDims.z; }

  size_t getInputTypeSize();
  size_t getInputN() { return sInputDataLenghtTable.at(std::make_pair(mNDims, mType))(mDims); };
  size_t getInputSize() { return getInputN() * getInputTypeSize(); }

  size_t getOutputTypeSize();
  size_t getOutputN() { return sOutputDataLenghtTable.at(std::make_pair(mNDims, mType))(mDims); };
  size_t getOutputSize() { return getOutputN() * getOutputTypeSize(); }
};

#endif  //  __PARAMETERS_H__
