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
  const sierra::kynema_ugf::SharedMemView<DoubleType*, SHMEM>& values,
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
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& values,
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
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& values,
  const DBLTYPE* oldValues)
{
  int counter = 0;
  for (size_t i = 0; i < values.extent(0); ++i) {
    for (size_t j = 0; j < values.extent(1); ++j) {
      for (size_t k = 0; k < values.extent(2); ++k) {
        ASSERT_NEAR(
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
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& view,
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
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& view,
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
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::kynema_ugf::SharedMemView<DoubleType*, SHMEM>& scv_volume,
  sierra::kynema_ugf::MasterElement* meSCV)
{
  int len = scv_volume.extent(0);
  std::vector<DoubleType> volume(len, 0.0);
  sierra::kynema_ugf::SharedMemView<
    DoubleType*, sierra::kynema_ugf::DeviceShmem>
    vol(volume.data(), volume.size());
  sierra::kynema_ugf::SharedMemView<
    DoubleType**, sierra::kynema_ugf::DeviceShmem>
    coords(v_coords.data(), v_coords.extent(0), v_coords.extent(1));
  meSCV->determinant(coords, vol);
  check_that_values_match(scv_volume, volume.data());
}

template <typename SHMEM>
void
compare_old_scs_areav(
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& scs_areav,
  sierra::kynema_ugf::MasterElement* meSCS)
{
  int len = scs_areav.extent(0) * scs_areav.extent(1);
  std::vector<DoubleType> areav(len, 0.0);
  sierra::kynema_ugf::SharedMemView<
    DoubleType**, sierra::kynema_ugf::DeviceShmem>
    area(areav.data(), scs_areav.extent(0), scs_areav.extent(1));
  sierra::kynema_ugf::SharedMemView<
    DoubleType**, sierra::kynema_ugf::DeviceShmem>
    coords(v_coords.data(), v_coords.extent(0), v_coords.extent(1));
  meSCS->determinant(coords, area);
  check_that_values_match(scs_areav, areav.data());
}

template <typename SHMEM>
void
compare_old_scs_grad_op(
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& scs_dndx,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& scs_deriv,
  sierra::kynema_ugf::MasterElement* meSCS)
{
  int len = scs_dndx.extent(0) * scs_dndx.extent(1) * scs_dndx.extent(2);
  std::vector<DoubleType> grad_op(len, 0.0);
  std::vector<DoubleType> deriv(len, 0.0);
  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    gradop(
      grad_op.data(), scs_dndx.extent(0), scs_dndx.extent(1),
      scs_dndx.extent(2));
  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    der(
      deriv.data(), scs_deriv.extent(0), scs_deriv.extent(1),
      scs_deriv.extent(2));
  sierra::kynema_ugf::SharedMemView<
    DoubleType**, sierra::kynema_ugf::DeviceShmem>
    coords(v_coords.data(), v_coords.extent(0), v_coords.extent(1));
  meSCS->grad_op(coords, gradop, der);
  check_that_values_match(scs_dndx, grad_op.data());
}

template <typename SHMEM>
void
compare_old_scs_shifted_grad_op(
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& scs_dndx,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& scs_deriv,
  sierra::kynema_ugf::MasterElement* meSCS)
{
  int len = scs_dndx.extent(0) * scs_dndx.extent(1) * scs_dndx.extent(2);
  std::vector<DoubleType> grad_op(len, 0.0);
  std::vector<DoubleType> deriv(len, 0.0);

  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    gradop(
      grad_op.data(), scs_dndx.extent(0), scs_dndx.extent(1),
      scs_dndx.extent(2));

  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    der(
      deriv.data(), scs_deriv.extent(0), scs_deriv.extent(1),
      scs_deriv.extent(2));
  sierra::kynema_ugf::SharedMemView<
    DoubleType**, sierra::kynema_ugf::DeviceShmem>
    coords(v_coords.data(), v_coords.extent(0), v_coords.extent(1));

  meSCS->shifted_grad_op(coords, gradop, der);
}

template <typename SHMEM>
void
compare_old_scs_gij(
  const sierra::kynema_ugf::SharedMemView<DoubleType**, SHMEM>& v_coords,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& v_gijUpper,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& v_gijLower,
  const sierra::kynema_ugf::SharedMemView<DoubleType***, SHMEM>& /* v_deriv */,
  sierra::kynema_ugf::MasterElement* meSCS)
{
  int gradOpLen =
    meSCS->nodesPerElement_ * meSCS->num_integration_points() * meSCS->nDim_;
  std::vector<DoubleType> grad_op(gradOpLen, 0.0);
  std::vector<DoubleType> v_deriv(gradOpLen, 0.0);

  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    gradop(
      grad_op.data(), meSCS->num_integration_points(), meSCS->nodesPerElement_,
      meSCS->nDim_);

  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    deriv(
      v_deriv.data(), meSCS->num_integration_points(), meSCS->nodesPerElement_,
      meSCS->nDim_);

  sierra::kynema_ugf::SharedMemView<
    DoubleType**, sierra::kynema_ugf::DeviceShmem>
    coords(v_coords.data(), v_coords.extent(0), v_coords.extent(1));

  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    gijUpper(
      v_gijUpper.data(), v_gijUpper.extent(0), v_gijUpper.extent(1),
      v_gijUpper.extent(2));

  sierra::kynema_ugf::SharedMemView<
    DoubleType***, sierra::kynema_ugf::DeviceShmem>
    gijLower(
      v_gijLower.data(), v_gijLower.extent(0), v_gijLower.extent(1),
      v_gijLower.extent(2));

  meSCS->grad_op(coords, gradop, deriv);
  meSCS->gij(coords, gijUpper, gijLower, deriv);
  // check_that_values_match(v_gijUpper, gijUpper.data());
  // check_that_values_match(v_gijLower, gijLower.data());
}

template <typename AlgTraits>
void
test_ME_views(const std::vector<sierra::kynema_ugf::ELEM_DATA_NEEDED>& requests)
{
  unit_test_utils::KokkosMEViews<AlgTraits> driver(true, true);

  // Passing `true` to constructor has already initialized everything
  // driver.fill_mesh_and_init_data(/* doPerturb = */ false);

  // Register ME data requests
  for (sierra::kynema_ugf::ELEM_DATA_NEEDED request : requests) {
    driver.dataNeeded().add_master_element_call(
      request, sierra::kynema_ugf::CURRENT_COORDINATES);
  }

  sierra::kynema_ugf::MasterElement* meSCS =
    sierra::kynema_ugf::MasterElementRepo::get_surface_master_element_on_host(
      AlgTraits::topo_);
  sierra::kynema_ugf::MasterElement* meSCV =
    sierra::kynema_ugf::MasterElementRepo::get_volume_master_element_on_host(
      AlgTraits::topo_);

  // Execute the loop and perform all tests
  driver.execute([&](
                   sierra::kynema_ugf::SharedMemData<
                     sierra::kynema_ugf::DeviceTeamHandleType,
                     sierra::kynema_ugf::DeviceShmem>& smdata) {
    // Extract data from scratchViews
    sierra::kynema_ugf::SharedMemView<
      DoubleType**, sierra::kynema_ugf::DeviceShmem>& v_coords =
      smdata.simdPrereqData.get_scratch_view_2D(*driver.coordinates_);
    auto& meViews = smdata.simdPrereqData.get_me_views(
      sierra::kynema_ugf::CURRENT_COORDINATES);

    if (meSCS != nullptr) {
      for (sierra::kynema_ugf::ELEM_DATA_NEEDED request : requests) {
        if (request == sierra::kynema_ugf::SCS_AREAV) {
          compare_old_scs_areav(v_coords, meViews.scs_areav, meSCS);
        }
        if (request == sierra::kynema_ugf::SCS_GRAD_OP) {
          compare_old_scs_grad_op(v_coords, meViews.dndx, meViews.deriv, meSCS);
        }
        if (request == sierra::kynema_ugf::SCS_SHIFTED_GRAD_OP) {
          compare_old_scs_shifted_grad_op(
            v_coords, meViews.dndx_shifted, meViews.deriv, meSCS);
        }
        if (request == sierra::kynema_ugf::SCS_GIJ) {
          compare_old_scs_gij(
            v_coords, meViews.gijUpper, meViews.gijLower, meViews.deriv, meSCS);
        }
      }
    }
    if (meSCV != nullptr) {
      for (sierra::kynema_ugf::ELEM_DATA_NEEDED request : requests) {
        if (request == sierra::kynema_ugf::SCV_VOLUME) {
          compare_old_scv_volume(v_coords, meViews.scv_volume, meSCV);
        }
        if (request == sierra::kynema_ugf::SCV_GRAD_OP) {
          if (AlgTraits::topo_ == stk::topology::HEX_8) {
            check_that_values_match(
              meViews.dndx_scv, &kokkos_me_gold::hex8_scv_grad_op[0]);
          } else if (AlgTraits::topo_ == stk::topology::TET_4) {
            check_that_values_match(
              meViews.dndx_scv, &kokkos_me_gold::tet4_scv_grad_op[0]);
          }
        }
        if (request == sierra::kynema_ugf::SCV_SHIFTED_GRAD_OP) {
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
  test_ME_views<sierra::kynema_ugf::AlgTraitsHex8>(
    {sierra::kynema_ugf::SCS_AREAV, sierra::kynema_ugf::SCS_GRAD_OP,
     sierra::kynema_ugf::SCV_VOLUME, sierra::kynema_ugf::SCV_GRAD_OP,
     sierra::kynema_ugf::SCV_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_tet4_views)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsTet4>(
    {sierra::kynema_ugf::SCS_AREAV, sierra::kynema_ugf::SCS_GRAD_OP,
     sierra::kynema_ugf::SCS_SHIFTED_GRAD_OP, sierra::kynema_ugf::SCV_VOLUME,
     sierra::kynema_ugf::SCV_GRAD_OP, sierra::kynema_ugf::SCV_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_tri32D_views)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsTri3_2D>(
    {sierra::kynema_ugf::SCS_AREAV, sierra::kynema_ugf::SCS_GRAD_OP,
     sierra::kynema_ugf::SCV_VOLUME});
}

TEST(KokkosME, test_tri32D_shifted_grad_op)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsTri3_2D>(
    {sierra::kynema_ugf::SCS_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_quad42D_views)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsQuad4_2D>(
    {sierra::kynema_ugf::SCS_AREAV, sierra::kynema_ugf::SCS_GRAD_OP,
     sierra::kynema_ugf::SCV_VOLUME});
}

TEST(KokkosME, test_quad42D_shifted_grad_op)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsQuad4_2D>(
    {sierra::kynema_ugf::SCS_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_wed6_views)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsWed6>(
    {sierra::kynema_ugf::SCV_VOLUME, sierra::kynema_ugf::SCS_AREAV,
     sierra::kynema_ugf::SCS_GRAD_OP});
}

TEST(KokkosME, test_wed6_shifted_grad_op)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsWed6>(
    {sierra::kynema_ugf::SCS_SHIFTED_GRAD_OP});
}

TEST(KokkosME, test_pyr5_views)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsPyr5>(
    {sierra::kynema_ugf::SCS_AREAV, sierra::kynema_ugf::SCS_GRAD_OP,
     sierra::kynema_ugf::SCV_VOLUME});
}

TEST(KokkosME, test_pyr5_views_shifted_grad_op)
{
  test_ME_views<sierra::kynema_ugf::AlgTraitsPyr5>({
    sierra::kynema_ugf::SCS_SHIFTED_GRAD_OP,
  });
}

#endif // KOKKOS_ENABLE_GPU
