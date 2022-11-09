#ifndef __GENERATOR_H__
#define __GENERATOR_H__

#include <curand.h>

#include "Parameters.h"

class Generator
{
  static const unsigned long long sSeed = 0ull;

  Parameters&       mParams;
  curandGenerator_t mGenerator;
  void*             mDevMem;
  void*             mHostMem;

public:
  Generator();
  void generate();
  const void* getData() { return mHostMem; }
  ~Generator();
};

#endif  //  __GENERATOR_H__
