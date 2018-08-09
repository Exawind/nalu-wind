#ifndef _UnitTestViewUtils_h_
#define _UnitTestViewUtils_h_

#include <gtest/gtest.h>
#include <string>
#include <ostream>
#include <Kokkos_Core.hpp>

namespace unit_test_utils {

template <typename T>
void dump_2d_view(const T& v, bool clipzero = true)
{
  for (unsigned j = 0; j < v.extent(1); ++j) {
    for (unsigned i = 0; i < v.extent(0); ++i) {
      double vout = (std::abs(v(j,i)) < 1.0e-15  && clipzero) ? 0.0 : v(j,i);
      std::cout << "(" << i << ", " << j << ")" << ", " << std::setw(5) << v.label() << ": " << std::setw(12) << vout;
      if (i != v.extent(0)-1) { std::cout << ", "; }
    }
    std::cout << std::endl;
  }
  std::cout << "--------------" << std::endl;
}

inline double clipv(double x) {
  return (std::abs(x) < 1.0e-15) ? 0.0 : x;
}

inline std::string vector_string(double x, double y) {
  std::stringstream ss;
  ss << "(" << clipv(x) << ", " << clipv(y) << ")";
  return ss.str();
}

template <typename T>
void dump_2d_vector_view(const T& v)
{
    for (unsigned j = 0; j < v.extent(1); ++j) {
      for (unsigned i = 0; i < v.extent(0); ++i) {
        std::cout << std::setw(5) << v.label() << "(" << i << ", " << j << "): "
        << std::setw(5) << vector_string(v(0,j,i), v(1,j,i));
        if (i != v.extent(0)-1) { std::cout << ", "; }
      }
      std::cout << std::endl;
    }
  std::cout << "--------------" << std::endl;
}

}

#define EXPECT_VIEW_NEAR_1D(x,y,tol) \
{ \
  EXPECT_EQ(x.extent(0), y.extent(0)); \
  for (unsigned i = 0; i < x.extent(0); ++i) { \
    EXPECT_NEAR(x(i), y(i), tol) << "i = " << i; \
  } \
}

#define EXPECT_VIEW_NEAR_2D(x,y,tol) \
{ \
  EXPECT_EQ(x.extent(0), y.extent(0)); \
  for (unsigned j = 0; j < x.extent(0);++j) { \
    for (unsigned i = 0; i < x.extent(1);++i) { \
      EXPECT_NEAR(x(j,i), y(j,i), tol) << "(j,i) = (" << j << ", " << i << ")";  \
    } \
  } \
}

#define EXPECT_VIEW_NEAR_3D(x,y,tol) \
{ \
  EXPECT_EQ(x.extent(0), y.extent(0)); \
  EXPECT_EQ(x.extent(1), y.extent(1)); \
  EXPECT_EQ(x.extent(2), y.extent(2)); \
  for (unsigned k = 0; k < x.extent(0);++k) { \
    for (unsigned j = 0; j < x.extent(1);++j) { \
      for (unsigned i = 0; i < x.extent(2);++i) { \
        ASSERT_NEAR(x(k,j,i), y(k,j,i), tol) << "(k,j,i) = (" << k << ", " << j << ", " << i << ")"; \
      } \
    } \
  } \
}

#define EXPECT_VIEW_NEAR_4D(x,y,tol) \
{ \
  EXPECT_EQ(x.extent(0), y.extent(0)); \
  EXPECT_EQ(x.extent(1), y.extent(1)); \
  EXPECT_EQ(x.extent(2), y.extent(2)); \
  EXPECT_EQ(x.extent(3), y.extent(3)); \
  for (unsigned l = 0; l < x.extent(0); ++l) { \
    for (unsigned k = 0; k < x.extent(1); ++k) { \
      for (unsigned j = 0; j < x.extent(2); ++j) { \
        for (unsigned i = 0; i < x.extent(3); ++i) { \
          EXPECT_NEAR(x(l,k,j,i), y(l,k,j,i), tol) << "(l,k,j,i) = (" << l<< ", " <<  k << ", " << j << ", " << i << ")"; \
        } \
      } \
    } \
  } \
}
#define TEST_POLY(x,y,z)  TEST(x, y##_##order_##z) { y<z>(); }

#define TEST_POLY_to5(x,y)  TEST(x, y##_##order_##1to7) \
{ \
   y<1>();  \
   y<2>();  \
   y<3>();  \
   y<4>();  \
   y<5>();  \
}


#endif

