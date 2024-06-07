// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NALUPARSEDTYPES_H
#define NALUPARSEDTYPES_H

#include <string>
#include <vector>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

// our data types
struct Velocity
{
  double ux_, uy_, uz_;
  Velocity() : ux_(0.0), uy_(0.0), uz_(0.0) {}
};

struct Coordinates
{
  double x_, y_, z_;
  Coordinates() : x_(0.0), y_(0.0), z_(0.0) {}
};

struct Pressure
{
  double pressure_;
  Pressure() : pressure_(0.0) {}
};

struct TurbKinEnergy
{
  double turbKinEnergy_;
  TurbKinEnergy() : turbKinEnergy_(0.0) {}
};

struct SpecDissRate
{
  double specDissRate_;
  SpecDissRate() : specDissRate_(0.0) {}
};

struct TotDissRate
{
  double totDissRate_;
  TotDissRate() : totDissRate_(0.0) {}
};

struct GammaInf
{
  double gamma_;
  GammaInf() : gamma_(1.0) {}
};

struct Temperature
{
  double temperature_;
  Temperature() : temperature_(0.0) {}
};

struct MixtureFraction
{
  double mixFrac_;
  MixtureFraction() : mixFrac_(0.0) {}
};

struct VolumeOfFluid
{
  double volumeOfFluid_;
  VolumeOfFluid() : volumeOfFluid_(0.0) {}
};

struct MassFraction
{
  std::vector<double> massFraction_;
  MassFraction() {}
};

struct ReferenceTemperature
{
  double referenceTemperature_;
  ReferenceTemperature() : referenceTemperature_(298.0) {}
};

struct NormalHeatFlux
{
  double qn_;
  NormalHeatFlux() : qn_(0.0) {}
};

struct NormalTemperatureGradient
{
  double tempGradN_;
  NormalTemperatureGradient() : tempGradN_(0.0) {}
};

struct RoughnessHeight
{
  double z0_;
  RoughnessHeight() : z0_(0.1) {}
};

struct MasterSlave
{
  std::string master_;
  std::string slave_;
  MasterSlave() {}
};

struct UserData;
struct WallBoundaryConditionData;
struct InflowBoundaryConditionData;
struct OpenBoundaryConditionData;
struct SymmetryBoundaryConditionData;
struct ABLTopBoundaryConditionData;
struct PeriodicBoundaryConditionData;
struct OversetBoundaryConditionData;
struct NonConformalBoundaryConditionData;
class PostProcessingData;
struct UserFunctionInitialConditionData;

} // namespace nalu
} // namespace sierra

#endif /* NALUPARSEDTYPES_H */
