#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"
#include <stk_util/parallel/Parallel.hpp>

#include "ElemDataRequests.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "CopyAndInterleave.h"

#include "AssembleFaceElemSolverAlgorithm.h"
#include "kernel/MomentumOpenAdvDiffElemKernel.h"
#include "kernel/MomentumSymmetryElemKernel.h"
#include "kernel/ScalarOpenAdvElemKernel.h"

#include <gtest/gtest.h>

namespace {
void fill_with_node_ids(stk::mesh::BulkData& bulk, IdFieldType* idField)
{
    const stk::mesh::BucketVector& nodeBuckets = bulk.buckets(stk::topology::NODE_RANK);
    for(const stk::mesh::Bucket* bptr : nodeBuckets) {
        for(stk::mesh::Entity node : *bptr) {
            double* idptr = stk::mesh::field_data(*idField, node);
            *idptr = bulk.identifier(node);
        }
    }
}

void verify_faces_exist(const stk::mesh::BulkData& bulk)
{
    EXPECT_TRUE(bulk.buckets(stk::topology::FACE_RANK).size() > 0);
}

class TestFaceElemKernel {
public:
    TestFaceElemKernel(stk::topology faceTopo, stk::topology elemTopo,
                       IdFieldType* idField,
                       sierra::nalu::ElemDataRequests& faceDataNeeded,
                       sierra::nalu::ElemDataRequests& elemDataNeeded)
    : numTimesExecuted_(0), faceTopo_(faceTopo), elemTopo_(elemTopo), idField_(idField)
    {
        faceDataNeeded.add_gathered_nodal_field(*idField, 1);
        elemDataNeeded.add_gathered_nodal_field(*idField, 1);
    }

    void execute( sierra::nalu::ScratchViews<DoubleType>& faceViews,
                 sierra::nalu::ScratchViews<DoubleType>& elemViews,
                 int numSimdFaces,
                 const int elemFaceOrdinal)
    {
        sierra::nalu::SharedMemView<DoubleType*>& faceNodeIds = faceViews.get_scratch_view_1D(*idField_);
        sierra::nalu::SharedMemView<DoubleType*>& elemNodeIds = elemViews.get_scratch_view_1D(*idField_);
        EXPECT_EQ(faceTopo_.num_nodes(), faceNodeIds.size());
        EXPECT_EQ(elemTopo_.num_nodes(), elemNodeIds.size());

        std::vector<int> faceNodeOrdinals(faceTopo_.num_nodes());

        for(int simdIndex=0; simdIndex<numSimdFaces; ++simdIndex) {
           elemTopo_.face_node_ordinals(elemFaceOrdinal, faceNodeOrdinals.begin());
           for(unsigned i=0; i<faceTopo_.num_nodes(); ++i) {
               DoubleType faceNodeId = faceNodeIds(i);
               DoubleType elemNodeId = elemNodeIds(faceNodeOrdinals[i]);
               EXPECT_NEAR(stk::simd::get_data(faceNodeId,simdIndex), stk::simd::get_data(elemNodeId,simdIndex), 1.e-9);
           }
        }

        ++numTimesExecuted_;
    }

    unsigned numTimesExecuted_;
    stk::topology faceTopo_;
    stk::topology elemTopo_;
    IdFieldType* idField_;
};

} //anonymous namespace

TEST_F(Hex8Mesh, faceElemBasic)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) {
    return;
  }
  fill_mesh("generated:1x1x1|sideset:xXyYzZ");
  verify_faces_exist(bulk);
  fill_with_node_ids(bulk, idField);

  stk::topology faceTopo = stk::topology::QUAD_4;
  stk::topology elemTopo = stk::topology::HEX_8;
  sierra::nalu::MasterElement* meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(faceTopo);
  sierra::nalu::MasterElement* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
  sierra::nalu::MasterElement* meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo);

  stk::mesh::Part* surface1 = meta.get_part("surface_1");
  unsigned numDof = 3;

  unit_test_utils::HelperObjects helperObjs(bulk, elemTopo, numDof, surface1);

  sierra::nalu::AssembleFaceElemSolverAlgorithm faceElemAlg(helperObjs.realm, surface1, &helperObjs.eqSystem,
                                                          faceTopo.num_nodes(), elemTopo.num_nodes());
  faceElemAlg.faceDataNeeded_.add_cvfem_face_me(meFC);
  faceElemAlg.elemDataNeeded_.add_cvfem_surface_me(meSCS);
  faceElemAlg.elemDataNeeded_.add_cvfem_volume_me(meSCV);

  TestFaceElemKernel faceElemKernel(faceTopo, elemTopo, idField,
                                    faceElemAlg.faceDataNeeded_, faceElemAlg.elemDataNeeded_);

  faceElemAlg.run_face_elem_algorithm(bulk,
          [&](sierra::nalu::SharedMemData_FaceElem<sierra::nalu::TeamHandleType,sierra::nalu::HostShmem> &smdata)
      {
        faceElemKernel.execute(smdata.simdFaceViews, smdata.simdElemViews, smdata.numSimdFaces, smdata.elemFaceOrdinal);
      });

  unsigned expectedNumFaces = 6;
  EXPECT_EQ(expectedNumFaces, faceElemKernel.numTimesExecuted_);
}

void move_face_from_surface2_to_surface3(stk::mesh::BulkData& bulk)
{
  stk::mesh::Part& surface2 = *bulk.mesh_meta_data().get_part("surface_2");
  stk::mesh::Part& surface3 = *bulk.mesh_meta_data().get_part("surface_3");
  stk::mesh::Part& surface3_q4 = *bulk.mesh_meta_data().get_part("surface_3_quad4");
  const stk::mesh::BucketVector& s2buckets = bulk.get_buckets(stk::topology::FACE_RANK, surface2);
  EXPECT_EQ(1u, s2buckets.size());
  EXPECT_EQ(2u, s2buckets[0]->size());
  stk::mesh::Entity s2face = (*s2buckets[0])[0];

  bulk.modification_begin();
  bulk.change_entity_parts(s2face, stk::mesh::ConstPartVector{&surface3, &surface3_q4}, stk::mesh::ConstPartVector{&surface2});
  bulk.modification_end();

  const stk::mesh::BucketVector& s3buckets = bulk.get_buckets(stk::topology::FACE_RANK, surface3);
  EXPECT_EQ(1u, s3buckets.size());
  EXPECT_EQ(3u, s3buckets[0]->size());

  const stk::mesh::Bucket& faceBucket = *s3buckets[0];
  stk::mesh::Entity firstFace = faceBucket[0];
  stk::mesh::Entity secondFace = faceBucket[1];
  //Each face should have just 1 attached element:
  EXPECT_EQ(1u, bulk.num_elements(firstFace));
  EXPECT_EQ(1u, bulk.num_elements(secondFace));

  //now make sure the first two faces in the bucket have different face-elem ordinals:
  const stk::mesh::ConnectivityOrdinal* firstFaceElemOrds = bulk.begin_element_ordinals(firstFace);
  const stk::mesh::ConnectivityOrdinal* secondFaceElemOrds = bulk.begin_element_ordinals(secondFace);
  EXPECT_NE(firstFaceElemOrds[0], secondFaceElemOrds[0]);
}

TEST_F(Hex8ElementWithBCFields, faceElemMomentumSymmetry)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) {
    return;
  }
  verify_faces_exist(bulk);

  sierra::nalu::SolutionOptions solnOptions;

  stk::topology faceTopo = stk::topology::QUAD_4;
  stk::topology elemTopo = stk::topology::HEX_8;
  stk::mesh::Part* surface1 = meta.get_part("all_surfaces");
  unit_test_utils::HelperObjects helperObjs(bulk, elemTopo, sierra::nalu::AlgTraitsQuad4Hex8::nDim_, surface1);

  sierra::nalu::AssembleFaceElemSolverAlgorithm faceElemAlg(helperObjs.realm, surface1, &helperObjs.eqSystem,
                                                          faceTopo.num_nodes(), elemTopo.num_nodes());

  auto  momentumSymmetryElemKernel =
      new sierra::nalu::MomentumSymmetryElemKernel<sierra::nalu::AlgTraitsQuad4Hex8>(meta, solnOptions, &velocity, &viscosity,
                                    faceElemAlg.faceDataNeeded_, faceElemAlg.elemDataNeeded_);

  faceElemAlg.activeKernels_.push_back(momentumSymmetryElemKernel);

  faceElemAlg.execute();
}

TEST_F(Hex8ElementWithBCFields, faceElemMomentumOpen)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) {
    return;
  }
  verify_faces_exist(bulk);

  sierra::nalu::SolutionOptions solnOptions;

  stk::topology faceTopo = stk::topology::QUAD_4;
  stk::topology elemTopo = stk::topology::HEX_8;
  stk::mesh::Part* surface1 = meta.get_part("all_surfaces");
  unit_test_utils::HelperObjects helperObjs(bulk, elemTopo, sierra::nalu::AlgTraitsQuad4Hex8::nDim_, surface1);

  sierra::nalu::AssembleFaceElemSolverAlgorithm faceElemAlg(helperObjs.realm, surface1, &helperObjs.eqSystem,
                                                          faceTopo.num_nodes(), elemTopo.num_nodes());

  auto  momentumOpenAdvDiffElemKernel =
    new sierra::nalu::MomentumOpenAdvDiffElemKernel<sierra::nalu::AlgTraitsQuad4Hex8>(meta, solnOptions, &helperObjs.eqSystem, &velocity, &Gjui, &viscosity,
                                                                                        faceElemAlg.faceDataNeeded_, faceElemAlg.elemDataNeeded_);

  faceElemAlg.activeKernels_.push_back(momentumOpenAdvDiffElemKernel);

  faceElemAlg.execute();
}

TEST_F(Hex8ElementWithBCFields, faceElemScalarOpen)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) {
    return;
  }
  verify_faces_exist(bulk);

  sierra::nalu::SolutionOptions solnOptions;

  stk::topology faceTopo = stk::topology::QUAD_4;
  stk::topology elemTopo = stk::topology::HEX_8;
  stk::mesh::Part* surface1 = meta.get_part("all_surfaces");
  unit_test_utils::HelperObjects helperObjs(bulk, elemTopo, sierra::nalu::AlgTraitsQuad4Hex8::nDim_, surface1);

  sierra::nalu::AssembleFaceElemSolverAlgorithm faceElemAlg(helperObjs.realm, surface1, &helperObjs.eqSystem,
                                                          faceTopo.num_nodes(), elemTopo.num_nodes());

  auto  scalarOpenAdvElemKernel =
    new sierra::nalu::ScalarOpenAdvElemKernel<sierra::nalu::AlgTraitsQuad4Hex8>(meta, solnOptions, &helperObjs.eqSystem, 
                                                                                &scalarQ, &bcScalarQ, &Gjq, &viscosity,
                                                                                faceElemAlg.faceDataNeeded_, faceElemAlg.elemDataNeeded_);

  faceElemAlg.activeKernels_.push_back(scalarOpenAdvElemKernel);

  faceElemAlg.execute();
}
