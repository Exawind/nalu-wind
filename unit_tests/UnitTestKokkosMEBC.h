/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef UNITTESTKOKKOSMEBC_H
#define UNITTESTKOKKOSMEBC_H

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

template <typename BcAlgTraits>
class KokkosMEBC
{
public:
  KokkosMEBC(int faceOrdinal,bool doInit=true, bool doPerturb=false)
     : comm_(MPI_COMM_WORLD),
       meta_(BcAlgTraits::nDim_),
       bulk_(meta_, comm_),
       faceOrdinal_(faceOrdinal)
   {
     if (doInit)
       fill_mesh_and_init_data(doPerturb);
   }

   virtual ~KokkosMEBC() {}

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
       unit_test_utils::create_one_perturbed_element(bulk_, BcAlgTraits::elemTopo_);
     else
       unit_test_utils::create_one_reference_element(bulk_,  BcAlgTraits::elemTopo_);

     partVec_ = {meta_.get_part("surface_" + std::to_string(faceOrdinal_))};
     coordinates_ = static_cast<const VectorFieldType*>( meta_.coordinate_field());

     EXPECT_TRUE(coordinates_ != nullptr);

     const int numDof = 1;
     helperObjs_.reset(new FaceElemHelperObjects(
       bulk_, BcAlgTraits::faceTopo_, BcAlgTraits::elemTopo_, numDof,
       partVec_[0]));

     elemDataNeeded().add_coordinates_field(
       *coordinates_, BcAlgTraits::nDim_, sierra::nalu::CURRENT_COORDINATES);
   }

   void init_me_data()
   {
     meFC_ = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::faceTopo_);
     meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element(BcAlgTraits::elemTopo_);
     // Register them to ElemDataRequests
    faceDataNeeded().add_cvfem_face_me(meFC_);
    elemDataNeeded().add_cvfem_surface_me(meSCS_);
   }

   template<typename LambdaFunction>
   void execute(LambdaFunction func)
   {
     ThrowRequireMsg(partVec_.size()==1, "KokkosMEViews unit-test assumes partVec_.size==1");
     ThrowRequireMsg(!bulk_.get_buckets(meta_.side_rank(), *partVec_[0]).empty(), "part does not contain side-ranked elements");

#ifndef KOKKOS_ENABLE_CUDA
     sierra::nalu::AssembleFaceElemSolverAlgorithm& alg = *(helperObjs_->assembleFaceElemSolverAlg);
     alg.run_face_elem_algorithm(bulk_, func);
#endif
   }

  sierra::nalu::ElemDataRequests& faceDataNeeded()
  {
    return helperObjs_->assembleFaceElemSolverAlg->faceDataNeeded_;
  }

  sierra::nalu::ElemDataRequests& elemDataNeeded()
  {
    return helperObjs_->assembleFaceElemSolverAlg->elemDataNeeded_;
  }

   stk::ParallelMachine comm_;
   stk::mesh::MetaData meta_;
   stk::mesh::BulkData bulk_;
   int faceOrdinal_;
   stk::mesh::PartVector partVec_;
   const VectorFieldType* coordinates_{nullptr};

   std::unique_ptr<FaceElemHelperObjects> helperObjs_;

   sierra::nalu::MasterElement* meFC_{nullptr};
   sierra::nalu::MasterElement* meSCV_{nullptr};
   sierra::nalu::MasterElement* meSCS_{nullptr};
};

} // namespace unit_test_utils

#endif /* UNITTESTKOKKOSME_H */
