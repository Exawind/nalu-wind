// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "UnitTestKokkosME.h"
#include "UnitTestKokkosMEGold.h"

#include <master_element/MasterElementRepo.h>
template <typename DBLTYPE, typename SHMEM>
void
check_that_values_match(
  const sierra::nalu::SharedMemView<DoubleType*, SHMEM>& values,
  const DBLTYPE* oldValues)
{
  for (size_t i = 0; i < values.extent(0); ++i) {
    EXPECT_NEAR(
      stk::simd::get_data(values(i), 0), stk::simd::get_data(oldValues[i], 0),
      tol)
      << "i:" << i;
  }
}

template <typename DBLTYPE, typename SHMEM>
void
check_that_values_match(
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& values,
  const DBLTYPE* oldValues)
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

template <typename DBLTYPE, typename SHMEM>
void
check_that_values_match(
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& values,
  const DBLTYPE* oldValues)
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
copy_DoubleType0_to_double(
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& view,
  std::vector<double>& vec)
{
  const DoubleType* viewValues = view.data();
  int len = view.size();
  vec.resize(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = stk::simd::get_data(viewValues[i], 0);
  }
}

template <typename SHMEM>
void
copy_DoubleType0_to_double(
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& view,
  std::vector<double>& vec)
{
  const DoubleType* viewValues = view.data();
  int len = view.size();
  vec.resize(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = stk::simd::get_data(viewValues[i], 0);
  }
}

template <typename SHMEM>
void
compare_old_scv_volume(
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::nalu::SharedMemView<DoubleType*, SHMEM>& scv_volume,
  sierra::nalu::MasterElement* meSCV)
{
  int len = scv_volume.extent(0);
  std::vector<DoubleType> volume(len, 0.0);
  sierra::nalu::SharedMemView<DoubleType*, sierra::nalu::DeviceShmem> vol(
    volume.data(), volume.size());
  sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem> coords(
    v_coords.data(), v_coords.extent(0), v_coords.extent(1));
  meSCV->determinant(coords, vol);
  check_that_values_match(scv_volume, volume.data());
}

template <typename SHMEM>
void
compare_old_scs_areav(
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& scs_areav,
  sierra::nalu::MasterElement* meSCS)
{
  int len = scs_areav.extent(0) * scs_areav.extent(1);
  std::vector<DoubleType> areav(len, 0.0);
  sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem> area(
    areav.data(), scs_areav.extent(0), scs_areav.extent(1));
  sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem> coords(
    v_coords.data(), v_coords.extent(0), v_coords.extent(1));
  meSCS->determinant(coords, area);
  check_that_values_match(scs_areav, areav.data());
}

template <typename SHMEM>
void
compare_old_scs_grad_op(
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& scs_dndx,
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& scs_deriv,
  sierra::nalu::MasterElement* meSCS)
{
  int len = scs_dndx.extent(0) * scs_dndx.extent(1) * scs_dndx.extent(2);
  std::vector<DoubleType> grad_op(len, 0.0);
  std::vector<DoubleType> deriv(len, 0.0);
  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> gradop(
    grad_op.data(), scs_dndx.extent(0), scs_dndx.extent(1), scs_dndx.extent(2));
  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> der(
    deriv.data(), scs_deriv.extent(0), scs_deriv.extent(1),
    scs_deriv.extent(2));
  sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem> coords(
    v_coords.data(), v_coords.extent(0), v_coords.extent(1));
  meSCS->grad_op(coords, gradop, der);
  check_that_values_match(scs_dndx, grad_op.data());
  check_that_values_match(scs_deriv, deriv.data());
}

template <typename SHMEM>
void
compare_old_scs_shifted_grad_op(
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& scs_dndx,
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& scs_deriv,
  sierra::nalu::MasterElement* meSCS)
{
  int len = scs_dndx.extent(0) * scs_dndx.extent(1) * scs_dndx.extent(2);
  std::vector<DoubleType> grad_op(len, 0.0);
  std::vector<DoubleType> deriv(len, 0.0);

  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> gradop(
    grad_op.data(), scs_dndx.extent(0), scs_dndx.extent(1), scs_dndx.extent(2));

  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> der(
    deriv.data(), scs_deriv.extent(0), scs_deriv.extent(1),
    scs_deriv.extent(2));
  sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem> coords(
    v_coords.data(), v_coords.extent(0), v_coords.extent(1));

  meSCS->shifted_grad_op(coords, gradop, der);
  check_that_values_match(scs_deriv, deriv.data());
}

template <typename SHMEM>
void
compare_old_scs_gij(
  const sierra::nalu::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& v_gijUpper,
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& v_gijLower,
  const sierra::nalu::SharedMemView<DoubleType***, SHMEM>& /* v_deriv */,
  sierra::nalu::MasterElement* meSCS)
{
  int gradOpLen =
    meSCS->nodesPerElement_ * meSCS->num_integration_points() * meSCS->nDim_;
  std::vector<DoubleType> grad_op(gradOpLen, 0.0);
  std::vector<DoubleType> v_deriv(gradOpLen, 0.0);

  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> gradop(
    grad_op.data(), meSCS->num_integration_points(), meSCS->nodesPerElement_,
    meSCS->nDim_);

  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> deriv(
    v_deriv.data(), meSCS->num_integration_points(), meSCS->nodesPerElement_,
    meSCS->nDim_);

  sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem> coords(
    v_coords.data(), v_coords.extent(0), v_coords.extent(1));

  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem>
    gijUpper(
      v_gijUpper.data(), v_gijUpper.extent(0), v_gijUpper.extent(1),
      v_gijUpper.extent(2));

  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem>
    gijLower(
      v_gijLower.data(), v_gijLower.extent(0), v_gijLower.extent(1),
      v_gijLower.extent(2));

  meSCS->grad_op(coords, gradop, deriv);
  meSCS->gij(coords, gijUpper, gijLower, deriv);
  check_that_values_match(v_gijUpper, gijUpper.data());
  check_that_values_match(v_gijLower, gijLower.data());
}

template <typename AlgTraits>
void
test_ME_views(const std::vector<sierra::nalu::ELEM_DATA_NEEDED>& requests)
{
  unit_test_utils::KokkosMEViews<AlgTraits> driver(true, true);

  // Passing `true` to constructor has already initialized everything
  // driver.fill_mesh_and_init_data(/* doPerturb = */ false);

  // Register ME data requests
  for (sierra::nalu::ELEM_DATA_NEEDED request : requests) {
    driver.dataNeeded().add_master_element_call(
      request, sierra::nalu::CURRENT_COORDINATES);
  }

  sierra::nalu::MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
      AlgTraits::topo_);
  sierra::nalu::MasterElement* meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element_on_host(
      AlgTraits::topo_);

  // Execute the loop and perform all tests
  driver.execute([&](
                   sierra::nalu::SharedMemData<
                     sierra::nalu::DeviceTeamHandleType,
                     sierra::nalu::DeviceShmem>& smdata) {
    // Extract data from scratchViews
    sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem>&
      v_coords =
        smdata.simdPrereqData.get_scratch_view_2D(*driver.coordinates_);
    auto& meViews =
      smdata.simdPrereqData.get_me_views(sierra::nalu::CURRENT_COORDINATES);

    if (meSCS != nullptr) {
      for (sierra::nalu::ELEM_DATA_NEEDED request : requests) {
        if (request == sierra::nalu::SCS_AREAV) {
          compare_old_scs_areav(v_coords, meViews.scs_areav, meSCS);
        }
        if (request == sierra::nalu::SCS_GRAD_OP) {
          compare_old_scs_grad_op(v_coords, meViews.dndx, meViews.deriv, meSCS);
        }
        if (request == sierra::nalu::SCS_SHIFTED_GRAD_OP) {
          compare_old_scs_shifted_grad_op(
            v_coords, meViews.dndx_shifted, meViews.deriv, meSCS);
        }
        if (request == sierra::nalu::SCS_GIJ) {
          compare_old_scs_gij(
            v_coords, meViews.gijUpper, meViews.gijLower, meViews.deriv, meSCS);
        }
      }
    }
    if (meSCV != nullptr) {
      for (sierra::nalu::ELEM_DATA_NEEDED request : requests) {
        if (request == sierra::nalu::SCV_VOLUME) {
          compare_old_scv_volume(v_coords, meViews.scv_volume, meSCV);
        }
        if (request == sierra::nalu::SCV_GRAD_OP) {
          if (AlgTraits::topo_ == stk::topology::HEX_8) {
            check_that_values_match(
              meViews.dndx_scv, &kokkos_me_gold::hex8_scv_grad_op[0]);
          } else if (AlgTraits::topo_ == stk::topology::TET_4) {
            check_that_values_match(
              meViews.dndx_scv, &kokkos_me_gold::tet4_scv_grad_op[0]);
          }
        }
        if (request == sierra::nalu::SCV_SHIFTED_GRAD_OP) {
          if (AlgTraits::topo_ == stk::topology::HEX_8) {
            check_that_values_match(
              meViews.dndx_scv_shifted,
              &kokkos_me_gold::hex8_scv_shifted_grad_op[0]);
          } else if (AlgTraits::topo_ == stk::topology::TET_4) {
            check_that_values_match(
              meViews.dndx_scv_shifted, &kokkos_me_gold::tet4_scv_grad_op[0]);
          }
        }
      }
    }
  });
}

#ifndef KOKKOS_ENABLE_GPU
TEST(KokkosME, test_hex8_views)
{
  test_ME_views<sierra::nalu::AlgTraitsHex8>(
    {sierra::nalu::SCS_AREAV, sierra::nalu::SCS_GRAD_OP,
     //   sierra::nalu::SCS_SHIFTED_GRAD_OP,
     sierra::nalu::SCS_GIJ, sierra::nalu::SCV_VOLUME, sierra::nalu::SCV_GRAD_OP,
     sierra::nalu::SCV_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_tet4_views)
{
  test_ME_views<sierra::nalu::AlgTraitsTet4>(
    {sierra::nalu::SCS_AREAV, sierra::nalu::SCS_GRAD_OP,
     sierra::nalu::SCS_SHIFTED_GRAD_OP, sierra::nalu::SCS_GIJ,
     sierra::nalu::SCV_VOLUME, sierra::nalu::SCV_GRAD_OP,
     sierra::nalu::SCV_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_tri32D_views)
{
  test_ME_views<sierra::nalu::AlgTraitsTri3_2D>(
    {sierra::nalu::SCS_AREAV, sierra::nalu::SCS_GRAD_OP, sierra::nalu::SCS_GIJ,
     sierra::nalu::SCV_VOLUME});
}

TEST(KokkosME, test_tri32D_shifted_grad_op)
{
  test_ME_views<sierra::nalu::AlgTraitsTri3_2D>(
    {sierra::nalu::SCS_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_quad42D_views)
{
  test_ME_views<sierra::nalu::AlgTraitsQuad4_2D>(
    {sierra::nalu::SCS_AREAV, sierra::nalu::SCS_GRAD_OP, sierra::nalu::SCS_GIJ,
     sierra::nalu::SCV_VOLUME});
}

TEST(KokkosME, test_quad42D_shifted_grad_op)
{
  test_ME_views<sierra::nalu::AlgTraitsQuad4_2D>(
    {sierra::nalu::SCS_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_wed6_views)
{
  test_ME_views<sierra::nalu::AlgTraitsWed6>(
    {sierra::nalu::SCV_VOLUME, sierra::nalu::SCS_AREAV,
     sierra::nalu::SCS_GRAD_OP, sierra::nalu::SCS_GIJ});
}

TEST(KokkosME, test_wed6_shifted_grad_op)
{
  test_ME_views<sierra::nalu::AlgTraitsWed6>(
    {sierra::nalu::SCS_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_pyr5_views)
{
  test_ME_views<sierra::nalu::AlgTraitsPyr5>(
    {sierra::nalu::SCS_AREAV, sierra::nalu::SCS_GRAD_OP,
     sierra::nalu::SCV_VOLUME});
}

TEST(KokkosME, test_pyr5_views_shifted_grad_op)
{
  test_ME_views<sierra::nalu::AlgTraitsPyr5>({
    sierra::nalu::SCS_SHIFTED_GRAD_OP,
  });
}

TEST(KokkosME, test_pyr5_views_gij)
{
  test_ME_views<sierra::nalu::AlgTraitsPyr5>({
    sierra::nalu::SCS_GRAD_OP,
    sierra::nalu::SCS_GIJ,
  });
}

#endif // KOKKOS_ENABLE_GPU
