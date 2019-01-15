#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "UnitTestUtils.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

#include <stk_util/parallel/Parallel.hpp>
#include <Kokkos_Core.hpp>

#include <ElemDataRequests.h>
#include <ScratchViews.h>

#include "UnitTestKokkosUtils.h"

namespace {

#ifndef KOKKOS_HAVE_CUDA

void element_discrete_laplacian_kernel_3d(
                       sierra::nalu::MasterElement& meSCS,
                       const ScalarFieldType* discreteLaplacianOfPressure,
                       const ScalarFieldType* nodalPressureField,
                       sierra::nalu::ScratchViews<double>& elemData)
{
    const int nDim = 3;
    const int nodesPerElem = meSCS.nodesPerElement_;
    const int numScsIp = meSCS.numIntPoints_;

    const int* lrscv = meSCS.adjacentNodes();

    sierra::nalu::SharedMemView<double*>& elemNodePressures = elemData.get_scratch_view_1D(*nodalPressureField);
    sierra::nalu::SharedMemView<double**>& scs_areav =
      elemData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scs_areav;
    sierra::nalu::SharedMemView<double***>& dndx =
      elemData.get_me_views(sierra::nalu::CURRENT_COORDINATES).dndx;
    const stk::mesh::Entity* elemNodes = elemData.elemNodes;

    for (int ip = 0; ip < numScsIp; ++ip ) {

      double dpdxIp = 0.0;
      for ( int ic = 0; ic < nodesPerElem; ++ic) {
        for ( int j = 0; j < nDim; ++j ) {
          dpdxIp += dndx(ip, ic, j)*elemNodePressures(ic)*scs_areav(ip,j);
        }
      }
      EXPECT_TRUE(std::abs(dpdxIp) > tol);

      const stk::mesh::Entity lNode = elemNodes[lrscv[2*ip+0]];
      const stk::mesh::Entity rNode = elemNodes[lrscv[2*ip+1]];

      Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, lNode), dpdxIp);
      Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, rNode), -dpdxIp);
    }
}

class SuppAlg
{
public:
  virtual ~SuppAlg(){}

  virtual void elem_execute(stk::topology topo,
                    sierra::nalu::MasterElement& meSCS,
                    sierra::nalu::ScratchViews<double>& elemData) = 0;
};

class DiscreteLaplacianSuppAlg : public SuppAlg
{
public:
  DiscreteLaplacianSuppAlg(sierra::nalu::ElemDataRequests& dataNeeded,
                           const VectorFieldType* coordField,
                           const ScalarFieldType* discreteLaplacianOfPressure,
                           const ScalarFieldType* nodalPressureField,
                           const stk::topology &topo)
   : discreteLaplacianOfPressure_(discreteLaplacianOfPressure),
     nodalPressureField_(nodalPressureField)
  {
    //here are the element-data pre-requisites we want computed before
    //our elem_execute method is called.
    dataNeeded.add_coordinates_field(*coordField, 3,
                                     sierra::nalu::CURRENT_COORDINATES);
    dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV,
                                       sierra::nalu::CURRENT_COORDINATES);
    dataNeeded.add_master_element_call(sierra::nalu::SCS_GRAD_OP,
                                       sierra::nalu::CURRENT_COORDINATES);
    dataNeeded.add_gathered_nodal_field(*nodalPressureField, 1);

    // add the master element
    sierra::nalu::MasterElement* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);
    dataNeeded.add_cvfem_surface_me(meSCS);
  }

  virtual ~DiscreteLaplacianSuppAlg() {}

  virtual void elem_execute(stk::topology /* topo */,
                    sierra::nalu::MasterElement& meSCS,
                    sierra::nalu::ScratchViews<double>& elemData)
  {
      element_discrete_laplacian_kernel_3d(meSCS,
            discreteLaplacianOfPressure_, nodalPressureField_, elemData);
  }

private:
  const ScalarFieldType* discreteLaplacianOfPressure_;
  const ScalarFieldType* nodalPressureField_;
};

//=========== Test class that mimics an element alg with supplemental alg and views ========
//
class TestElemAlgorithmWithSuppAlgViews
{
public:
  TestElemAlgorithmWithSuppAlgViews(stk::mesh::BulkData& bulk)
  : suppAlgs_(), bulkData_(bulk)
  {}

  void execute()
  {
      const stk::mesh::MetaData& meta = bulkData_.mesh_meta_data();
  
      const stk::mesh::BucketVector& elemBuckets = bulkData_.get_buckets(stk::topology::ELEM_RANK, meta.locally_owned_part());
  
      sierra::nalu::ElemDataRequestsNGP dataNeededNGP(dataNeededByKernels_, meta.get_fields().size());
      const int bytes_per_team = 0;
      const int bytes_per_thread = get_num_bytes_pre_req_data(dataNeededNGP, meta.spatial_dimension());
      auto team_exec = sierra::nalu::get_host_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);
      Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
      {
          const stk::mesh::Bucket& bkt = *elemBuckets[team.league_rank()];
          stk::topology topo = bkt.topology();
          sierra::nalu::MasterElement* meSCS = dataNeededNGP.get_cvfem_surface_me();

          sierra::nalu::ScratchViews<double> prereqData(team, bulkData_, topo.num_nodes(), dataNeededNGP);

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bkt.size()), [&](const size_t& jj)
          {
             fill_pre_req_data(dataNeededNGP, bulkData_, bkt[jj], prereqData);
            
             for(SuppAlg* alg : suppAlgs_) {
               alg->elem_execute(topo, *meSCS, prereqData);
             }
          });
      });
  }

  std::vector<SuppAlg*> suppAlgs_;
  sierra::nalu::ElemDataRequests dataNeededByKernels_;

private:
  stk::mesh::BulkData& bulkData_;
};


TEST_F(Hex8Mesh, elem_supp_alg_views)
{
    fill_mesh_and_initialize_test_fields("generated:20x20x20");

    TestElemAlgorithmWithSuppAlgViews testAlgorithm(bulk);

    //DiscreteLapacianSuppAlg constructor says which data it needs, by inserting
    //things into the 'dataNeededByKernels_' container.
    
    // find a topo, assume this is a homogeneous hex8 mesh
    for (size_t k = 0; k < partVec.size(); ++k )
      if ( partVec[k]->topology() != stk::topology::HEX_8 ) { 
        throw std::runtime_error("UnitTestElemSuppAlg only supports a homogeneous HEX8 mesh: " + partVec[k]->topology().name());
      }

    stk::topology partTopo = partVec[0]->topology();
    SuppAlg* suppAlg = new DiscreteLaplacianSuppAlg(testAlgorithm.dataNeededByKernels_,
                                                    coordField,
                                                    discreteLaplacianOfPressure, nodalPressureField, partTopo);

    testAlgorithm.suppAlgs_.push_back(suppAlg);

    testAlgorithm.execute();

    check_discrete_laplacian(exactLaplacian);

    delete suppAlg;
}

//end of stuff that's ifndef'd for KOKKOS_HAVE_CUDA
#endif

}

