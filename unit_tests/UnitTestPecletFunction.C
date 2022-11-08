// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "PecletFunction.h"
#include "SimdInterface.h"
#include "NGPInstance.h"

#include <vector>
#include <cmath>

namespace {

constexpr double tolerance = 1.0e-6;

template <typename PecFuncType, typename ValueType>
ValueType
exec_on_device(PecFuncType* devptr, ValueType pecNum)
{
  ValueType pecFac = 0.0;
  Kokkos::parallel_reduce(
    1, KOKKOS_LAMBDA(int, ValueType& pf) { pf = devptr->execute(pecNum); },
    pecFac);
  return pecFac;
}

} // namespace

TEST(PecletFunction, NGP_classic_double)
{
  const double A = 5.0;
  const double hybridFactor = 1.0;
  std::vector<double> pecletNumbers = {0.0, 1.0, std::sqrt(5.0), 1e5};
  std::vector<double> pecletFactors = {0.0, 1.0 / 6.0, 0.5, 1.0};

  auto* pecFunc =
    sierra::nalu::nalu_ngp::create<sierra::nalu::ClassicPecletFunction<double>>(
      A, hybridFactor);

  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(
      exec_on_device(pecFunc, pecletNumbers[i]), pecletFactors[i], tolerance);
  }

  sierra::nalu::nalu_ngp::destroy(pecFunc);
}

TEST(PecletFunction, NGP_classic_simd)
{
  const DoubleType A = 5.0;
  const DoubleType hybridFactor = 1.0;
  NALU_ALIGNED DoubleType pecletNumbers[] = {0.0, 1.0, std::sqrt(5.0), 1e5};
  std::vector<double> pecletFactors = {0.0, 1.0 / 6.0, 0.5, 1.0};

  auto* pecFunc = sierra::nalu::nalu_ngp::create<
    sierra::nalu::ClassicPecletFunction<DoubleType>>(A, hybridFactor);

#if !defined(KOKKOS_ENABLE_GPU)
  for (int i = 0; i < 4; i++) {
    const DoubleType pecFac = exec_on_device(pecFunc, pecletNumbers[i]);
    for (int is = 0; is < stk::simd::ndoubles; is++) {
      EXPECT_NEAR(stk::simd::get_data(pecFac, is), pecletFactors[i], tolerance);
    }
  }
#endif

  sierra::nalu::nalu_ngp::destroy(pecFunc);
}

TEST(PecletFunction, NGP_tanh_double)
{
  const double c1 = 5000.0;
  const double c2 = 200.0;
  std::vector<double> pecletNumbers = {-c1 - 10.0 * c2, c1, c1 + 10.0 * c2};
  std::vector<double> pecletFactors = {0.0, 0.5, 1.0};

  auto* pecFunc =
    sierra::nalu::nalu_ngp::create<sierra::nalu::TanhFunction<double>>(c1, c2);

  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(
      exec_on_device(pecFunc, pecletNumbers[i]), pecletFactors[i], tolerance);
  }

  sierra::nalu::nalu_ngp::destroy(pecFunc);
}

TEST(PecletFunction, NGP_tanh_simd)
{
  const DoubleType c1 = 5000.0;
  const DoubleType c2 = 200.0;
  NALU_ALIGNED DoubleType pecletNumbers[] = {-10.0 * c2, c1, c1 + 10.0 * c2};
  std::vector<double> pecletFactors = {0.0, 0.5, 1.0};

  auto* pecFunc =
    sierra::nalu::nalu_ngp::create<sierra::nalu::TanhFunction<DoubleType>>(
      c1, c2);

#if !defined(KOKKOS_ENABLE_GPU)
  for (int i = 0; i < 3; i++) {
    const DoubleType pecFac = exec_on_device(pecFunc, pecletNumbers[i]);
    for (int is = 0; is < stk::simd::ndoubles; is++) {
      EXPECT_NEAR(stk::simd::get_data(pecFac, is), pecletFactors[i], tolerance);
    }
  }
#endif

  sierra::nalu::nalu_ngp::destroy(pecFunc);
}
