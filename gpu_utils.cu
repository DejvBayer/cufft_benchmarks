#include "gpu_utils.cuh"

std::ostream& operator<<(std::ostream& os, const dim3& dims)
{
  os << dims.x << ", " << dims.y << ", " << dims.z;

  return os;
}
