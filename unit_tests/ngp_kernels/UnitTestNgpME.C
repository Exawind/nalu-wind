// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "UnitTestUtils.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "master_element/Hex8CVFEM.h"
#include "master_element/Wed6CVFEM.h"
#include "master_element/Pyr5CVFEM.h"
#include "master_element/Tet4CVFEM.h"
#include "master_element/Quad43DCVFEM.h"
#include "master_element/Tri33DCVFEM.h"
#include "AlgTraits.h"

namespace unit_test_me_ngp {

using namespace sierra::nalu;

template <typename T>
void
test_mescs_create_impl()
{
#if 0
  using METype = typename T::masterElementScs_;
  METype* mescs = kokkos_malloc_on_device<METype>("MEalloc");
  std::cerr << mescs << std::endl;
  Kokkos::parallel_for("MEplacement", 1, KOKKOS_LAMBDA (const int) {
      new (mescs) METype();
      printf("%d\n", mescs->nodesPerElement_);
    });
  kokkos_free_on_device(mescs);
#else
  const auto* mescs =
    MasterElementRepo::get_surface_master_element_on_dev(T::topo_);
  EXPECT_TRUE(mescs != nullptr);
#endif
}

template <typename T>
void
test_mescv_create_impl()
{
  const auto* mescv =
    MasterElementRepo::get_volume_master_element_on_dev(T::topo_);
  EXPECT_TRUE(mescv != nullptr);
}

#define MESCS_TEST(METype)                                                     \
  TEST(MECreate, NGP_scs_##METype)                                             \
  {                                                                            \
    test_mescs_create_impl<AlgTraits##METype>();                               \
  }

#define MESCV_TEST(METype)                                                     \
  TEST(MECreate, NGP_scv_##METype)                                             \
  {                                                                            \
    test_mescv_create_impl<AlgTraits##METype>();                               \
  }

MESCS_TEST(Hex8);
MESCS_TEST(Tet4);
MESCS_TEST(Pyr5);
MESCS_TEST(Wed6);
MESCS_TEST(Quad4);
MESCS_TEST(Tri3);

MESCV_TEST(Hex8);
MESCV_TEST(Tet4);
MESCV_TEST(Pyr5);
MESCV_TEST(Wed6);

} // namespace unit_test_me_ngp
