// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef ComputeMdotAlgorithmDriver_h
#define ComputeMdotAlgorithmDriver_h

#include <AlgorithmDriver.h>
#include <string>

namespace sierra{
namespace nalu{

class Realm;
class SolutionOptions;

class ComputeMdotAlgorithmDriver : public AlgorithmDriver
{
public:

  ComputeMdotAlgorithmDriver(
    Realm &realm);

  ~ComputeMdotAlgorithmDriver();

  double compute_accumulation();
  void correct_open_mdot(const double finalCorrection);
  void provide_output();

  SolutionOptions &solnOpts_;
  bool hasMass_;
  bool lumpedMass_;

  void pre_work();
  void post_work();
};
  

} // namespace nalu
} // namespace Sierra

#endif
