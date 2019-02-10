/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef UNITTESTKOKKOSME_H
#define UNITTESTKOKKOSME_H

#include "gtest/gtest.h"
#include "UnitTestUtils.h"

#include "ScratchViews.h"
#include "CopyAndInterleave.h"
#include "ElemDataRequests.h"
#include "AlgTraits.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"

#include "UnitTestHelperObjects.h"

namespace unit_test_utils {

template<typename AlgTraits>
class KokkosMEViews
{
public:
  KokkosMEViews(bool doInit=true, bool doPerturb=false)
    : comm_(MPI_COMM_WORLD),
      meta_(AlgTraits::nDim_),
      bulk_(meta_, comm_)
  {
    if (doInit)
      fill_mesh_and_init_data(doPerturb);
  }

  virtual ~KokkosMEViews() {}

  /** Create a 1-element STK mesh and initialize MasterElement data structures
   */
  void fill_mesh_and_init_data(bool doPerturb=false)
  {
    fill_mesh(doPerturb);
    init_me_data();
  }

  void fill_mesh(bool doPerturb=false)
  {
    if (doPerturb)
      unit_test_utils::create_one_perturbed_element(bulk_, AlgTraits::topo_);
    else
      unit_test_utils::create_one_reference_element(bulk_, AlgTraits::topo_);

    partVec_ = {meta_.get_part("block_1")};
    coordinates_ = static_cast<const VectorFieldType*>(
      meta_.coordinate_field());

    EXPECT_TRUE(coordinates_ != nullptr);

    const int numDof = 1;
    helperObjs_.reset(new HelperObjects(bulk_, AlgTraits::topo_, numDof, partVec_[0]));
    dataNeeded().add_coordinates_field(
      *coordinates_, AlgTraits::nDim_, sierra::nalu::CURRENT_COORDINATES);
  }

  void init_me_data()
  {
    // Initialize both surface and volume elements
    meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
    meSCV_ = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

    // Register them to ElemDataRequests
    dataNeeded().add_cvfem_surface_me(meSCS_);
    dataNeeded().add_cvfem_volume_me(meSCV_);

    // Initialize shape function views
    double scs_data[AlgTraits::numScsIp_*AlgTraits::nodesPerElement_];
    meSCS_->shape_fcn(scs_data);
    DoubleType* v_scs_data = &scs_shape_fcn_(0,0);
    for (int i=0; i < (AlgTraits::numScsIp_*AlgTraits::nodesPerElement_); ++i) {
      v_scs_data[i] = scs_data[i];
    }

    double scv_data[AlgTraits::numScvIp_*AlgTraits::nodesPerElement_];
    meSCV_->shape_fcn(scv_data);
    DoubleType* v_scv_data = &scv_shape_fcn_(0,0);
    for (int i=0; i < (AlgTraits::numScvIp_*AlgTraits::nodesPerElement_); ++i) {
      v_scv_data[i] = scv_data[i];
    }
  }

  template<typename LambdaFunction>
  void execute(LambdaFunction func)
  {
    ThrowAssertMsg(partVec_.size()==1, "KokkosMEViews unit-test assumes partVec_.size==1");

    helperObjs_->assembleElemSolverAlg->run_algorithm(bulk_, func);
  }

  inline sierra::nalu::ElemDataRequests& dataNeeded()
  { return helperObjs_->assembleElemSolverAlg->dataNeededByKernels_; }

  stk::ParallelMachine comm_;
  stk::mesh::MetaData meta_;
  stk::mesh::BulkData bulk_;
  stk::mesh::PartVector partVec_;
  const VectorFieldType* coordinates_{nullptr};

  std::unique_ptr<HelperObjects> helperObjs_{nullptr};

  sierra::nalu::MasterElement* meFC_{nullptr};
  sierra::nalu::MasterElement* meSCV_{nullptr};
  sierra::nalu::MasterElement* meSCS_{nullptr};


  sierra::nalu::AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> scv_shape_fcn_ {"scv_shape_function"};
  sierra::nalu::AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> scs_shape_fcn_ {"scs_shape_function"};
};

} // namespace unit_test_utils

#endif /* UNITTESTKOKKOSME_H */
