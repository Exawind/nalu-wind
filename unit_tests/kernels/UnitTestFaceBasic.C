#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"
#include <stk_util/parallel/Parallel.hpp>

#include "ElemDataRequests.h"
#include "ScratchViews.h"
#include "CopyAndInterleave.h"
#include "kernel/Kernel.h"

#include <gtest/gtest.h>

namespace {
  void verify_faces_exist(const stk::mesh::BulkData& bulk)
  {
      EXPECT_TRUE(bulk.buckets(stk::topology::FACE_RANK).size() > 0);
  }
}

class TestFaceKernel : public sierra::nalu::Kernel {
public:
  TestFaceKernel(stk::topology topo, ScalarFieldType* scalarQ, sierra::nalu::ElemDataRequests& dataNeeded)
  : numTimesExecuted_(0), topo_(topo), scalarQ_(scalarQ)
  {
    dataNeeded.add_gathered_nodal_field(*scalarQ, 1);
  }

  using sierra::nalu::Kernel::execute;
  virtual void execute(
    sierra::nalu::SharedMemView<DoubleType**>&  /* lhs */,
    sierra::nalu::SharedMemView<DoubleType*>&  /* rhs */,
    sierra::nalu::ScratchViews<DoubleType>& faceViews)
  {
    sierra::nalu::SharedMemView<DoubleType*>& scalarQview = faceViews.get_scratch_view_1D(*scalarQ_);
    EXPECT_EQ(topo_.num_nodes(), scalarQview.size());
    ++numTimesExecuted_;
  }

  unsigned numTimesExecuted_;
private:
  stk::topology topo_;
  ScalarFieldType* scalarQ_;
};

TEST_F(Hex8Mesh, faceBasic)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) {
    return;
  }
  fill_mesh("generated:1x1x1|sideset:xXyYzZ");
  verify_faces_exist(bulk);

  stk::topology faceTopo = stk::topology::QUAD_4;
  sierra::nalu::MasterElement* meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(faceTopo);

  stk::mesh::Part* surface1 = meta.get_part("surface_1");
  int numDof = 1;
  unit_test_utils::HelperObjects helperObjs(bulk, faceTopo, numDof, surface1);
  helperObjs.assembleElemSolverAlg->dataNeededByKernels_.add_cvfem_face_me(meFC);

  TestFaceKernel faceKernel(faceTopo, scalarQ, helperObjs.assembleElemSolverAlg->dataNeededByKernels_);
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(&faceKernel);

  helperObjs.assembleElemSolverAlg->execute();

  unsigned expectedNumFaces = 6;
  EXPECT_EQ(expectedNumFaces, faceKernel.numTimesExecuted_);
}
