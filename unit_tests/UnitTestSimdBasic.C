#include <gtest/gtest.h>

#include <stk_util/environment/WallTime.hpp>
#include <stk_simd/Simd.hpp>
#include <KokkosInterface.h>

#include "UnitTestUtils.h"

#include <limits>
#include <vector>

TEST(Simd, basic)
{
#if defined(STK_SIMD_AVX512) || defined(STK_SIMD_AVX) || defined(STK_SIMD_SSE)
  EXPECT_EQ(stk::simd::nfloats, 2 * stk::simd::ndoubles);
#endif
}

TEST(Simd, whichInstructions)
{
  std::cout << "STK simd native type: " << typeid(kokkos_simd::simd_abi::native).name() << std::endl;
  std::cout << "stk::simd::ndoubles = " << stk::simd::ndoubles << std::endl;
}

typedef std::vector<double, non_std::AlignedAllocator<double, 64>>
  aligned_vector;
template <typename T>
using AlignedVector = std::vector<T, non_std::AlignedAllocator<T, 64>>;

void
initialize(int N, aligned_vector& x, aligned_vector& y)
{
  for (int n = 0; n < N; ++n) {
    x[n] = (rand() - 0.5) / RAND_MAX;
    y[n] = (rand() - 0.5) / RAND_MAX;
  }
}

TEST(Simd, stkMath)
{
  const int N = 512; // this is a multiple of the simd width
                     // if this is not true, the remainder
                     // must be handled appropriately
  aligned_vector x(N);
  aligned_vector y(N);
  aligned_vector solution(N);

  initialize(N, x, y);

  for (int n = 0; n < N; n += stk::simd::ndoubles) {
    const stk::simd::Double xl = stk::simd::load(&x[n]);
    const stk::simd::Double yl = stk::simd::load(&y[n]);
    stk::simd::Double zl = stk::math::abs(xl) * stk::math::exp(yl);
    stk::simd::store(&solution[n], zl);
  }

  for (int n = 0; n < N; ++n) {
    EXPECT_NEAR(std::abs(x[n]) * std::exp(y[n]), solution[n], 1.e-6);
  }
}

TEST(Simd, Views)
{
  const int N = 3;
  auto DoubleView =
    Kokkos::View<stk::simd::Double**, Kokkos::HostSpace>("DoubleView", N, N);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      DoubleView(i, j) = 1.0 * (i + j + 1);
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      stk::simd::Double& d = DoubleView(i, j);
      std::cout << i << "," << j << ": ";
      for (int k = 0; k < stk::simd::ndoubles; ++k) {
        std::cout << stk::simd::get_data(d, k) << ",";
      }
      std::cout << std::endl;
    }
  }

  stk::simd::Double& d = DoubleView(0, 0);
  double* all = &d[0];
  for (int i = 0; i < N * N * stk::simd::ndoubles; ++i) {
    std::cout << i << ": " << all[i] << std::endl;
  }
}
