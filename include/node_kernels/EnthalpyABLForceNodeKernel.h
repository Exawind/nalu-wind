/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
