// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SurfaceForceAndMomentAlgorithmDriver_h
#define SurfaceForceAndMomentAlgorithmDriver_h

#include <AlgorithmDriver.h>
#include <string>
#include <vector>

namespace sierra {
namespace nalu {

class Realm;

class SurfaceForceAndMomentAlgorithmDriver : public AlgorithmDriver
{
public:
  SurfaceForceAndMomentAlgorithmDriver(Realm& realm);
  ~SurfaceForceAndMomentAlgorithmDriver();

  std::vector<Algorithm*> algVec_;

  void execute();

  void zero_fields();
  void parallel_assemble_area();
  void parallel_assemble_fields();
};

} // namespace nalu
} // namespace sierra

#endif
