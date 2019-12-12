// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MDOTALGDRIVER_H
#define MDOTALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "ngp_algorithms/MdotOpenCorrectorAlg.h"
#include "SimdInterface.h"

namespace sierra {
namespace nalu {

class Realm;

class MdotAlgDriver : public NgpAlgDriver
{
public:
  MdotAlgDriver(Realm&, const bool);

  virtual ~MdotAlgDriver() = default;

  //! Reset data before calling algorithms
  virtual void pre_work() override;

  //! Perform mdot correction logic after algorithms have done their work
  virtual void post_work() override;

  //! Add up density accumulation from different topo element algorithms
  void add_density_accumulation(const DoubleType&);

  //! Update contribution from inflow sidesets
  void add_inflow_mdot(const DoubleType&);

  //! Update contributions from outflow sidesets
  void add_open_mdot(const DoubleType&);

  void add_density_accumulation(const double&);

  void add_inflow_mdot(const double&);

  void add_open_mdot(const double&);

  void add_open_mdot_post(const double&);

  double mdot_inflow() const { return mdotInflow_; }

  double mdot_rho_accum() const { return rhoAccum_; }

  double mdot_open() const { return mdotOpen_; }

  double mdot_open_post() const { return mdotOpenPost_; }

  const double& mdot_open_correction() const { return mdotOpenCorrection_; }

  void provide_output();

  void register_open_mdot_corrector_alg(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const std::string& algSuffix);

  template<template <typename> class NgpAlg,
           typename LegacyAlg,
           class ... Args>
  void register_open_mdot_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const stk::topology elemTopo,
    const std::string& algSuffix,
    const bool needCorrection,
    Args&& ... args)
  {
    register_face_elem_algorithm<NgpAlg, LegacyAlg>(
      algType, part, elemTopo, algSuffix, std::forward<Args>(args)...);

    if (needCorrection) {
      register_open_mdot_corrector_alg(algType, part, algSuffix);
    }
  }

  template<template <typename> class NgpAlg, class ... Args>
  void register_open_mdot_algorithm(
    AlgorithmType algType,
    stk::mesh::Part* part,
    const stk::topology elemTopo,
    const std::string& algSuffix,
    const bool needCorrection,
    Args&& ... args)
  {
    register_face_elem_algorithm<NgpAlg>(
      algType, part, elemTopo, algSuffix, std::forward<Args>(args)...);

    if (needCorrection) {
      register_open_mdot_corrector_alg(algType, part, algSuffix);
    }
  }

private:
  std::map<std::string, std::unique_ptr<Algorithm>> correctOpenMdotAlgs_;

  double rhoAccum_{0.0};
  double mdotInflow_{0.0};
  double mdotOpen_{0.0};
  double mdotOpenPost_{0.0};
  double mdotOpenCorrection_{0.0};
  unsigned mdotOpenIpCount_{0};

  bool elemContinuityEqs_{true};
  bool hasOpenBC_{true};
  bool isInit_{true};
};

}  // nalu
}  // sierra


#endif /* MDOTALGDRIVER_H */
