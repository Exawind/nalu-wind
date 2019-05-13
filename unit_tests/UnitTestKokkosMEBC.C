/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "gtest/gtest.h"
#include "UnitTestKokkosMEBC.h"


namespace {
  void check_that_values_match(const sierra::nalu::SharedMemView<DoubleType***>& values,
    const double* oldValues)
  {
    int counter = 0;
    for(size_t i=0; i<values.extent(0); ++i) {
      for(size_t j=0; j<values.extent(1); ++j) {
        for(size_t k=0; k<values.extent(2); ++k) {
          EXPECT_NEAR(stk::simd::get_data(values(i,j,k),0), oldValues[counter++], tol)<<"i:"<<i<<", j:"<<j<<", k:"<<k;
        }
      }
    }
  }

  void copy_DoubleType0_to_double(const sierra::nalu::SharedMemView<DoubleType**>& view,
    std::vector<double>& vec)
  {
    const DoubleType* viewValues = view.data();
    int len = view.size();
    vec.resize(len);
    for(int i=0; i<len; ++i) {
      vec[i] = stk::simd::get_data(viewValues[i], 0);
    }
  }
}

void compare_old_face_grad_op(
  const int faceOrdinal,
  const bool shifted,
  const sierra::nalu::SharedMemView<DoubleType**>& v_coords,
  const sierra::nalu::SharedMemView<DoubleType***>& scs_fc_dndx,
  sierra::nalu::MasterElement* meSCS)
{
  int len = scs_fc_dndx.extent(0)*scs_fc_dndx.extent(1)*scs_fc_dndx.extent(2);
  std::vector<double> coords;
  copy_DoubleType0_to_double(v_coords, coords);
  std::vector<double> grad_op(len, 0.0);
  std::vector<double> deriv(len, 0.0);
  std::vector<double> det_j(len, 0.0);
  double error = 0;

  if (shifted) meSCS->shifted_face_grad_op(1, faceOrdinal, coords.data(), grad_op.data(), det_j.data(), &error);
  else         meSCS->        face_grad_op(1, faceOrdinal, coords.data(), grad_op.data(), det_j.data(), &error);

  EXPECT_NEAR(error, 0.0, tol);
  check_that_values_match(scs_fc_dndx, &grad_op[0]);
}

template<typename BcAlgTraits>
void test_MEBC_views(int faceOrdinal, const std::vector<sierra::nalu::ELEM_DATA_NEEDED>& elem_requests)
{
  unit_test_utils::KokkosMEBC<BcAlgTraits> driver(faceOrdinal, true, true);
  ASSERT_TRUE((BcAlgTraits::nDim_ == 3 && driver.bulk_.buckets(stk::topology::FACE_RANK).size() > 0) 
           || (BcAlgTraits::nDim_ == 2 && driver.bulk_.buckets(stk::topology::EDGE_RANK).size() > 0));

  // Register ME data requests
  for(sierra::nalu::ELEM_DATA_NEEDED request : elem_requests) {
    driver.elemDataNeeded().add_master_element_call(
      request, sierra::nalu::CURRENT_COORDINATES);
  }

  // Execute the loop and perform all tests
  driver.execute([&](sierra::nalu::SharedMemData_FaceElem<sierra::nalu::TeamHandleType,sierra::nalu::HostShmem>& smdata) {

    sierra::nalu::SharedMemView<DoubleType**>& v_coords = smdata.simdElemViews.get_scratch_view_2D(*driver.coordinates_);
    auto& meViews = smdata.simdElemViews.get_me_views(sierra::nalu::CURRENT_COORDINATES);

    for(sierra::nalu::ELEM_DATA_NEEDED request : elem_requests) {
      if (request == sierra::nalu::SCS_FACE_GRAD_OP) {
        compare_old_face_grad_op(faceOrdinal, false, v_coords, meViews.dndx_fc_scs, driver.meSCS_);
      }
      if (request == sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP) {
        compare_old_face_grad_op(faceOrdinal, true, v_coords, meViews.dndx_shifted_fc_scs, driver.meSCS_);
      }
    }
  });
}

TEST(KokkosMEBC, test_quad42D_views)
{
  for (int k = 0; k < 3; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsEdge2DQuad42D>(k, 
      {sierra::nalu::SCS_FACE_GRAD_OP, sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_quad92D_views)
{
  for (int k = 0; k < 3; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsEdge32DQuad92D>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_tri32D_views)
{
  for (int k = 0; k < 3; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsEdge2DTri32D>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP, sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_hex8_views)
{
  for (int k = 0; k < 6; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsQuad4Hex8>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP, sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_hex27_views)
{
  for (int k = 0; k < 6; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsQuad9Hex27>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_tet4_views)
{
  for (int k = 0; k < 4; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsTri3Tet4>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP, sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP});
  }
}


TEST(KokkosMEBC, test_wedge4_views)
{
  for (int k = 0; k < 3; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsQuad4Wed6>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP, sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP});
  }

  for (int k = 3; k < 5; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsTri3Wed6>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP, sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_pyr5_views)
{
  for (int k = 0; k < 4; ++k) {
    test_MEBC_views<sierra::nalu::AlgTraitsTri3Pyr5>(k,
      {sierra::nalu::SCS_FACE_GRAD_OP});
  }
  test_MEBC_views<sierra::nalu::AlgTraitsQuad4Pyr5>(4,
    {sierra::nalu::SCS_FACE_GRAD_OP, sierra::nalu::SCS_SHIFTED_FACE_GRAD_OP});
}
