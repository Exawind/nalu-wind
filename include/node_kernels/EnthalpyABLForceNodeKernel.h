// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ENTHLAPYABLFORCENODEKERNEL_H
#define ENTHLAPYABLFORCENODEKERNEL_H

#include "node_kernels/NodeKernel.h"
#include "wind_energy/ABLSrcInterp.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class EnthalpyABLForceNodeKernel : public NGPNodeKernel<EnthalpyABLForceNodeKernel>
{
public:
  EnthalpyABLForceNodeKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&);

  KOKKOS_FUNCTION
  EnthalpyABLForceNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~EnthalpyABLForceNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> coordinates_;
  ngp::Field<double> dualNodalVolume_;

  ABLScalarInterpolator ablSrc_;

  unsigned coordinatesID_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolumeID_ {stk::mesh::InvalidOrdinal};

  const int nDim_;
};

}  // nalu
}  // sierra


#endif /* ENTHLAPYABLFORCENODEKERNEL_H */
