// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "UnitTestKokkosMEBC.h"

namespace {
template <typename SHMEM>
void
check_that_values_match(
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& values,
  const DoubleType* oldValues)
{
  int counter = 0;
  for (size_t i = 0; i < values.extent(0); ++i) {
    for (size_t j = 0; j < values.extent(1); ++j) {
      for (size_t k = 0; k < values.extent(2); ++k) {
        EXPECT_NEAR(
          stk::simd::get_data(values(i, j, k), 0),
          stk::simd::get_data(oldValues[counter++], 0), tol)
          << "i:" << i << ", j:" << j << ", k:" << k;
      }
    }
  }
}

template <typename SHMEM>
void
check_that_values_match(
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& values,
  const DoubleType* oldValues)
{
  int counter = 0;
  for (size_t i = 0; i < values.extent(0); ++i) {
    for (size_t j = 0; j < values.extent(1); ++j) {
      EXPECT_NEAR(
        stk::simd::get_data(values(i, j), 0),
        stk::simd::get_data(oldValues[counter++], 0), tol)
        << "i:" << i << ", j:" << j;
    }
  }
}
} // namespace

template <typename SHMEM>
void
compare_old_face_grad_op(
  const int faceOrdinal,
  const bool shifted,
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& scs_dndx,
  sierra::kynema_ugf::MasterElement* meSCS)
{
  int len = scs_dndx.extent(0) * scs_dndx.extent(1) * scs_dndx.extent(2);
  std::vector<DoubleType> grad_op(len, 0.0);
  std::vector<DoubleType> det_j(len, 0.0);

  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    grad(
      grad_op.data(), scs_dndx.extent(0), scs_dndx.extent(1),
      scs_dndx.extent(2));
  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    det(
      det_j.data(), scs_dndx.extent(0), scs_dndx.extent(1), scs_dndx.extent(2));
  sierra::kynema_ugf::SharedMemView<
    DoubleType**, sierra::kynema_ugf::DeviceShmem>
    coords(v_coords.data(), v_coords.extent(0), v_coords.extent(1));

  if (shifted)
    meSCS->shifted_face_grad_op(faceOrdinal, coords, grad, det);
  else
    meSCS->face_grad_op(faceOrdinal, coords, grad, det);

  check_that_values_match(scs_dndx, grad_op.data());
}

template <typename BcAlgTraits>
void
test_MEBC_views(
  int faceOrdinal,
  const std::vector<sierra::kynema_ugf::ELEM_DATA_NEEDED>& elem_requests,
  const std::vector<sierra::kynema_ugf::ELEM_DATA_NEEDED>& face_requests =
    std::vector<sierra::kynema_ugf::ELEM_DATA_NEEDED>())
{
  unit_test_utils::KokkosMEBC<BcAlgTraits> driver(faceOrdinal, true, true);
  ASSERT_TRUE(
    (BcAlgTraits::nDim_ == 3 &&
     driver.bulk_->buckets(stk::topology::FACE_RANK).size() > 0) ||
    (BcAlgTraits::nDim_ == 2 &&
     driver.bulk_->buckets(stk::topology::EDGE_RANK).size() > 0));

  // Register ME data requests
  for (sierra::kynema_ugf::ELEM_DATA_NEEDED request : elem_requests) {
    driver.elemDataNeeded().add_master_element_call(
      request, sierra::kynema_ugf::CURRENT_COORDINATES);
  }
  for (sierra::kynema_ugf::ELEM_DATA_NEEDED request : face_requests) {
    driver.faceDataNeeded().add_master_element_call(
      request, sierra::kynema_ugf::CURRENT_COORDINATES);
  }

  // Execute the loop and perform all tests
  driver.execute([&](
                   sierra::kynema_ugf::SharedMemData_FaceElem<
                     sierra::kynema_ugf::TeamHandleType,
                     sierra::kynema_ugf::DeviceShmem>& smdata) {
    sierra::kynema_ugf::SharedMemView<
      DoubleType**, sierra::kynema_ugf::DeviceShmem>& v_coords =
      smdata.simdElemViews.get_scratch_view_2D(*driver.coordinates_);
    auto& meViews = smdata.simdElemViews.get_me_views(
      sierra::kynema_ugf::CURRENT_COORDINATES);
    auto& fcViews = smdata.simdFaceViews.get_me_views(
      sierra::kynema_ugf::CURRENT_COORDINATES);

    for (sierra::kynema_ugf::ELEM_DATA_NEEDED request : elem_requests) {
      if (request == sierra::kynema_ugf::SCS_FACE_GRAD_OP) {
        compare_old_face_grad_op(
          faceOrdinal, false, v_coords, meViews.dndx_fc_scs, driver.meSCS_);
      }
      if (request == sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP) {
        compare_old_face_grad_op(
          faceOrdinal, true, v_coords, meViews.dndx_shifted_fc_scs,
          driver.meSCS_);
      }
    }
  });
}

TEST(KokkosMEBC, test_quad42D_views)
{
  for (int k = 0; k < 3; ++k) {
    test_MEBC_views<sierra::kynema_ugf::AlgTraitsEdge2DQuad42D>(
      k, {sierra::kynema_ugf::SCS_FACE_GRAD_OP,
          sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_tri32D_views)
{
  for (int k = 0; k < 3; ++k) {
    test_MEBC_views<sierra::kynema_ugf::AlgTraitsEdge2DTri32D>(
      k, {sierra::kynema_ugf::SCS_FACE_GRAD_OP,
          sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_hex8_views)
{
  for (int k = 0; k < 6; ++k) {
    test_MEBC_views<sierra::kynema_ugf::AlgTraitsQuad4Hex8>(
      k, {sierra::kynema_ugf::SCS_FACE_GRAD_OP,
          sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_tet4_views)
{
  for (int k = 0; k < 4; ++k) {
    test_MEBC_views<sierra::kynema_ugf::AlgTraitsTri3Tet4>(
      k, {sierra::kynema_ugf::SCS_FACE_GRAD_OP,
          sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_wedge4_views)
{
  for (int k = 0; k < 3; ++k) {
    test_MEBC_views<sierra::kynema_ugf::AlgTraitsQuad4Wed6>(
      k, {sierra::kynema_ugf::SCS_FACE_GRAD_OP,
          sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP});
  }

  for (int k = 3; k < 5; ++k) {
    test_MEBC_views<sierra::kynema_ugf::AlgTraitsTri3Wed6>(
      k, {sierra::kynema_ugf::SCS_FACE_GRAD_OP,
          sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP});
  }
}

TEST(KokkosMEBC, test_pyr5_views)
{
  for (int k = 0; k < 4; ++k) {
    test_MEBC_views<sierra::kynema_ugf::AlgTraitsTri3Pyr5>(
      k, {sierra::kynema_ugf::SCS_FACE_GRAD_OP});
  }
  test_MEBC_views<sierra::kynema_ugf::AlgTraitsQuad4Pyr5>(
    4, {sierra::kynema_ugf::SCS_FACE_GRAD_OP,
        sierra::kynema_ugf::SCS_SHIFTED_FACE_GRAD_OP});
}
