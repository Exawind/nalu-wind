#include <gtest/gtest.h>
#include <limits>
#include <random>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_ngp/NgpMesh.hpp>
#include <stk_ngp/NgpFieldManager.hpp>

#include <stk_util/parallel/Parallel.hpp>
#include <Kokkos_Core.hpp>

#include <master_element/MasterElementFactory.h>
#include <ElemDataRequestsGPU.h>
#include <ScratchViews.h>

#include "UnitTestKokkosUtils.h"
#include "UnitTestUtils.h"

namespace {

#ifndef KOKKOS_ENABLE_CUDA

using sierra::nalu::SharedMemView;

class SuppAlg
{
public:
  virtual ~SuppAlg(){}

  virtual void elem_execute(stk::topology topo,
                    sierra::nalu::ScratchViews<double>& elemData) = 0;
};

class TestSuppAlg : public SuppAlg
{
public:
  TestSuppAlg(sierra::nalu::ElemDataRequests& dataNeeded,
              const ScalarFieldType* ndScalarField,
              const VectorFieldType* ndVectorField,
              const TensorFieldType* ndTensorField,
              const ScalarFieldType* elScalarField,
              const VectorFieldType* elVectorField,
              const TensorFieldType* elTensorField)
   :
     nodalScalarField(ndScalarField),
     nodalVectorField(ndVectorField),
     nodalTensorField(ndTensorField),
     elemScalarField(elScalarField),
     elemVectorField(elVectorField),
     elemTensorField(elTensorField)
  {
    //here are the element-data pre-requisites we want computed before
    //our elem_execute method is called.
    dataNeeded.add_gathered_nodal_field(*nodalScalarField, 1);
    dataNeeded.add_gathered_nodal_field(*nodalVectorField, 4);
    dataNeeded.add_gathered_nodal_field(*nodalTensorField, 3, 3);
    dataNeeded.add_element_field(*elemScalarField, 1);
    dataNeeded.add_element_field(*elemVectorField, 8);
    dataNeeded.add_element_field(*elemTensorField, 2, 2);
  }

  virtual ~TestSuppAlg() {}

  virtual void elem_execute(stk::topology topo,
                    sierra::nalu::ScratchViews<double>& elemData)
  {
    unsigned nodesPerElem = topo.num_nodes();

    SharedMemView<double*>& nodalScalarView = elemData.get_scratch_view_1D(*nodalScalarField);
    SharedMemView<double**>& nodalVectorView = elemData.get_scratch_view_2D(*nodalVectorField);
    SharedMemView<double***>& nodalTensorView = elemData.get_scratch_view_3D(*nodalTensorField);

    SharedMemView<double*>& elemScalarView = elemData.get_scratch_view_1D(*elemScalarField);
    SharedMemView<double*>& elemVectorView = elemData.get_scratch_view_1D(*elemVectorField);
    SharedMemView<double**>& elemTensorView = elemData.get_scratch_view_2D(*elemTensorField);

    EXPECT_EQ(nodesPerElem, nodalScalarView.extent(0));
    EXPECT_EQ(nodesPerElem, nodalVectorView.extent(0));
    EXPECT_EQ(4u,           nodalVectorView.extent(1));
    EXPECT_EQ(nodesPerElem, nodalTensorView.extent(0));
    EXPECT_EQ(3u,           nodalTensorView.extent(1));
    EXPECT_EQ(3u,           nodalTensorView.extent(2));

    EXPECT_EQ(1u, elemScalarView.extent(0));
    EXPECT_EQ(8u, elemVectorView.extent(0));
    EXPECT_EQ(2u, elemTensorView.extent(0));
    EXPECT_EQ(2u, elemTensorView.extent(1));
  }

private:
  const ScalarFieldType* nodalScalarField;
  const VectorFieldType* nodalVectorField;
  const TensorFieldType* nodalTensorField;
  const ScalarFieldType* elemScalarField;
  const VectorFieldType* elemVectorField;
  const TensorFieldType* elemTensorField;
};

//=========== Test class that mimics an alg with supplemental algs ========
//
class TestAlgorithm
{
public:
  TestAlgorithm(stk::mesh::BulkData& bulk)
  : suppAlgs_(),
    dataNeededByKernels_(bulk.mesh_meta_data()),
    bulkData_(bulk)
  {}

  void execute()
  {
      const stk::mesh::MetaData& meta = bulkData_.mesh_meta_data();
  
      const stk::mesh::BucketVector& elemBuckets = bulkData_.get_buckets(stk::topology::ELEM_RANK, meta.locally_owned_part());
  
      //In this unit-test we know we're working on a hex8 mesh. In real algorithms,
      //a topology would be available.
      dataNeededByKernels_.add_cvfem_surface_me(sierra::nalu::MasterElementRepo::get_surface_master_element(stk::topology::HEX_8));

      ngp::Mesh ngpMesh(bulkData_);
      ngp::FieldManager fieldMgr(bulkData_);

      sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeededByKernels_, meta.get_fields().size());
      const int bytes_per_team = 0;
      const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension());
      auto team_exec = sierra::nalu::get_host_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);
      Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
      {
          const stk::mesh::Bucket& bkt = *elemBuckets[team.league_rank()];
          stk::topology topo = bkt.topology();

          sierra::nalu::ScratchViews<double> prereqData(team, meta.spatial_dimension(), topo.num_nodes(), dataNeededNGP);

          // See get_num_bytes_pre_req_data for padding
          EXPECT_EQ(static_cast<unsigned>(bytes_per_thread), prereqData.total_bytes() + 8 * sizeof(double));

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bkt.size()), [&](const size_t& jj)
          {
            fill_pre_req_data(dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, bkt[jj], prereqData);
            
             for(SuppAlg* alg : suppAlgs_) {
               alg->elem_execute(topo, prereqData);
             }
          });
      });
  }

  std::vector<SuppAlg*> suppAlgs_;
  sierra::nalu::ElemDataRequests dataNeededByKernels_;

private:
  stk::mesh::BulkData& bulkData_;
};


TEST_F(Hex8Mesh, supp_alg_data_sharing)
{
    ScalarFieldType& nodalScalarField = meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "nodalScalarField");
    VectorFieldType& nodalVectorField = meta.declare_field<VectorFieldType>(stk::topology::NODE_RANK, "nodalVectorField");
    TensorFieldType& nodalTensorField = meta.declare_field<TensorFieldType>(stk::topology::NODE_RANK, "nodalTensorField");
    ScalarFieldType& elemScalarField = meta.declare_field<ScalarFieldType>(stk::topology::ELEM_RANK, "elemScalarField");
    VectorFieldType& elemVectorField = meta.declare_field<VectorFieldType>(stk::topology::ELEM_RANK, "elemVectorField");
    TensorFieldType& elemTensorField = meta.declare_field<TensorFieldType>(stk::topology::ELEM_RANK, "elemTensorField");

    const stk::mesh::Part& wholemesh = meta.universal_part();

    stk::mesh::put_field_on_mesh(nodalScalarField, wholemesh, nullptr);
    stk::mesh::put_field_on_mesh(nodalVectorField, wholemesh, 4, nullptr);
    stk::mesh::put_field_on_mesh(nodalTensorField, wholemesh, 3, 3, nullptr);

    stk::mesh::put_field_on_mesh(elemScalarField, wholemesh, nullptr);
    stk::mesh::put_field_on_mesh(elemVectorField, wholemesh, 8, nullptr);
    stk::mesh::put_field_on_mesh(elemTensorField, wholemesh, 2, 2, nullptr);

    fill_mesh("generated:10x10x10");

    TestAlgorithm testAlgorithm(bulk);

    //TestSuppAlg constructor says which data it needs, by inserting
    //things into the 'dataNeededByKernels_' container.
    
    SuppAlg* suppAlg = new TestSuppAlg(testAlgorithm.dataNeededByKernels_,
                                     &nodalScalarField, &nodalVectorField, &nodalTensorField,
                                     &elemScalarField, &elemVectorField, &elemTensorField);

    testAlgorithm.suppAlgs_.push_back(suppAlg);

    testAlgorithm.execute();

    delete suppAlg;
}

TEST_F(Hex8Mesh, inconsistent_field_requests)
{
    ScalarFieldType& nodalScalarField = meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "nodalScalarField");
    TensorFieldType& nodalTensorField = meta.declare_field<TensorFieldType>(stk::topology::NODE_RANK, "nodalTensorField");
    ScalarFieldType& elemScalarField = meta.declare_field<ScalarFieldType>(stk::topology::ELEM_RANK, "elemScalarField");
    TensorFieldType& elemTensorField = meta.declare_field<TensorFieldType>(stk::topology::ELEM_RANK, "elemTensorField");

    const stk::mesh::Part& wholemesh = meta.universal_part();

    stk::mesh::put_field_on_mesh(nodalScalarField, wholemesh, nullptr);
    stk::mesh::put_field_on_mesh(nodalTensorField, wholemesh, 3, 3, nullptr);

    stk::mesh::put_field_on_mesh(elemScalarField, wholemesh, nullptr);
    stk::mesh::put_field_on_mesh(elemTensorField, wholemesh, 2, 2, nullptr);

    fill_mesh("generated:10x10x10");

    sierra::nalu::ElemDataRequests prereqData(meta);

    prereqData.add_gathered_nodal_field(nodalScalarField, 1);
    EXPECT_THROW(prereqData.add_gathered_nodal_field(nodalScalarField, 2), std::logic_error);
    prereqData.add_element_field(elemScalarField, 1);
    EXPECT_THROW(prereqData.add_element_field(elemScalarField, 2), std::logic_error);

    prereqData.add_gathered_nodal_field(nodalTensorField, 3, 3);
    EXPECT_THROW(prereqData.add_gathered_nodal_field(nodalTensorField, 5, 5), std::logic_error);
    EXPECT_THROW(prereqData.add_gathered_nodal_field(nodalTensorField, 5), std::logic_error);

    prereqData.add_element_field(elemTensorField, 2, 2);
    EXPECT_THROW(prereqData.add_element_field(elemTensorField, 5, 5), std::logic_error);
    EXPECT_THROW(prereqData.add_element_field(elemTensorField, 5), std::logic_error);
}

//end of stuff that's ifndef'd for KOKKOS_ENABLE_CUDA
#endif

}

