// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MeshMotionInfo_h
#define MeshMotionInfo_h

// standard c++
#include <string>
#include <vector>

namespace sierra {
namespace nalu {

class MeshMotionInfo
{
public:
  MeshMotionInfo(
    std::vector<std::string> meshMotionBlock,
    const double omega,
    std::vector<double> centroid,
    std::vector<double> unitVec,
    const bool computeCentroid);

  ~MeshMotionInfo();

  std::vector<std::string> meshMotionBlock_;
  const double omega_;
  std::vector<double> centroid_;
  std::vector<double> unitVec_;
  const double computeCentroid_;
  double computeCentroidCompleted_;
};

} // namespace nalu
} // namespace sierra

#endif
