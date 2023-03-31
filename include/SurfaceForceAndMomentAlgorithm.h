// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SurfaceForceAndMomentAlgorithm_h
#define SurfaceForceAndMomentAlgorithm_h

#include <Algorithm.h>
#include <FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

class Realm;

class SurfaceForceAndMomentAlgorithm : public Algorithm
{
public:
  SurfaceForceAndMomentAlgorithm(
    Realm& realm,
    stk::mesh::PartVector& partVec,
    const std::string& outputFileName,
    const int& frequency_,
    const std::vector<double>& parameters,
    const bool& useShifted);
  ~SurfaceForceAndMomentAlgorithm();

  void execute();

  void pre_work();

  void cross_product(double* force, double* cross, double* rad);

  const std::string& outputFileName_;
  const int& frequency_;
  const std::vector<double>& parameters_;
  const bool useShifted_;
  const double includeDivU_;

  VectorFieldType* coordinates_;
  ScalarFieldType* pressure_;
  VectorFieldType* pressureForce_;
  VectorFieldType* viscousForce_;
  VectorFieldType* tauWallVector_;
  ScalarFieldType* tauWall_;
  ScalarFieldType* yplus_;
  ScalarFieldType* density_;
  ScalarFieldType* viscosity_;
  TensorFieldType* dudx_;
  GenericFieldType* exposedAreaVec_;
  ScalarFieldType* assembledArea_;

  const int w_;
};

} // namespace nalu
} // namespace sierra

#endif
