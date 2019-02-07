/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ScalarDiffFemKernel.h"
#include "AlgTraits.h"
#include "master_element/Hex8FEM.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
ScalarDiffFemKernel<AlgTraits>::ScalarDiffFemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* scalarQ,
  ScalarFieldType* diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    bulkData_(&bulkData),
    scalarQ_(scalarQ->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    meFEM_(new Hex8FEM()),
    ipWeight_(&meFEM_->weights_[0]),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(scalarQ->name()))
{
  ThrowRequireMsg(AlgTraits::topo_ == stk::topology::HEX_8,
                  "FEM_DIFF only available for hexes currently");

  // Save of required fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = get_field_ordinal(metaData, "coordinates");

  // master element, shape function is shifted consistently
  if ( shiftedGradOp_ )
    get_fem_shape_fn_data<AlgTraits>([&](double* ptr){meFEM_->shifted_shape_fcn(ptr);}, v_shape_function_);
  else
    get_fem_shape_fn_data<AlgTraits>([&](double* ptr){meFEM_->shape_fcn(ptr);}, v_shape_function_);

  dataPreReqs.add_fem_volume_me(meFEM_);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  dataPreReqs.add_gathered_nodal_field(diffFluxCoeff_, 1);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(FEM_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(FEM_GRAD_OP, CURRENT_COORDINATES);
}

template<typename AlgTraits>
ScalarDiffFemKernel<AlgTraits>::~ScalarDiffFemKernel()
{
  delete meFEM_;
}

template<typename AlgTraits>
void
ScalarDiffFemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_scalarQ = scratchViews.get_scratch_view_1D(scalarQ_);
  SharedMemView<DoubleType*>& v_diffFluxCoeff = scratchViews.get_scratch_view_1D(diffFluxCoeff_);
  
  SharedMemView<DoubleType***>& v_dndx = scratchViews.get_me_views(CURRENT_COORDINATES).dndx_fem;
  SharedMemView<DoubleType*>& v_det_j = scratchViews.get_me_views(CURRENT_COORDINATES).det_j_fem;

  for ( int ip = 0; ip < AlgTraits::numGp_; ++ip ) {

    // compute ip property
    DoubleType diffFluxCoeffIp = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);
      diffFluxCoeffIp += r*v_diffFluxCoeff(ic);
    }

    // start the assembly
    const DoubleType ipFactor = v_det_j(ip)*ipWeight_[ip];

    // row ir
    for ( int ir = 0; ir < AlgTraits::nodesPerElement_; ++ir) {

      // column ic
      DoubleType rhsSum = 0.0;
      for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

        DoubleType lhsSum = 0.0;
        DoubleType scalarQ = v_scalarQ(ic);
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
          const DoubleType fac = v_dndx(ip,ir,j)*diffFluxCoeffIp*v_dndx(ip,ic,j);
          lhsSum += fac;
          rhsSum += fac*scalarQ;
        }
        lhs(ir,ic) += lhsSum*ipFactor;
      }
      rhs(ir) -= rhsSum*ipFactor;
    }
  }
}

INSTANTIATE_KERNEL(ScalarDiffFemKernel)

}  // nalu
}  // sierra
