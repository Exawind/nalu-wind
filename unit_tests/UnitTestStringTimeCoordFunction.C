#include "gtest/gtest.h"

#include "user_functions/StringTimeCoordFunction.h"

namespace sierra::nalu {
TEST(StringTimeCoordFunction, same_on_host_as_string_function)
{
  std::string func = "1 + 3 * t + x + y + z*y";
  const double val_at_1234 = 21;
  ASSERT_DOUBLE_EQ(StringTimeCoordFunction(func)(1, 2, 3, 4), val_at_1234);
}

double
val_executed_in_kernel(
  StringTimeCoordFunction func, double t, double x, double y, double z)
{
  double val = 0;
  Kokkos::parallel_reduce(
    1, KOKKOS_LAMBDA(int, double& v) { v += func(t, x, y, z); },
    Kokkos::Sum<double>(val));
  return val;
}

TEST(StringTimeCoordFunction, runs_on_device)
{
  std::string func =
    "1 + 3 * sin(t) + exp(x) + exp(y) + ((z>0 && z < 1) ? z*y : x*z)";

  StringTimeCoordFunction f(func);
  auto val_1 = f(1, 2, 3, 4);

  auto val_2 = val_executed_in_kernel(f, 1, 2, 3, 4);
  ASSERT_DOUBLE_EQ(val_1, val_2);
}

TEST(StringTimeCoordFunction, empty)
{
  EXPECT_ANY_THROW(StringTimeCoordFunction f(""));
}

TEST(StringTimeCoordFunction, unparsable)
{
  EXPECT_ANY_THROW(StringTimeCoordFunction f("c++/45g"));
}

TEST(StringTimeCoordFunction, undefined)
{
  EXPECT_ANY_THROW(StringTimeCoordFunction f("h(t) + 34"));
}

TEST(StringTimeCoordFunction, copy)
{
  std::string s = "3*t + x*y - z";
  StringTimeCoordFunction f(s);

  StringTimeCoordFunction f2(s);
  StringTimeCoordFunction f3(f2);
  StringTimeCoordFunction f4 = f2;

  const double t = 2.5;
  const double x = 1.1;
  const double y = -1.5;
  const double z = 12.5;
  const double ans = 7.5 - 1.1 * 1.5 - 12.5;
  EXPECT_DOUBLE_EQ(val_executed_in_kernel(f2, t, x, y, z), ans);
  EXPECT_DOUBLE_EQ(val_executed_in_kernel(f3, t, x, y, z), ans);
  EXPECT_DOUBLE_EQ(val_executed_in_kernel(f2, t, x, y, z), ans);
  EXPECT_DOUBLE_EQ(val_executed_in_kernel(f4, t, x, y, z), ans);

  EXPECT_TRUE(f2.is_spatial());
}

TEST(StringTimeCoordFunction, time_only)
{
  StringTimeCoordFunction f = {"3*t"};

  const double t = 2.5;
  const double x = 1.1;
  const double y = -1.5;
  const double z = 12.5;
  const double ans = 7.5;

  EXPECT_DOUBLE_EQ(val_executed_in_kernel(f, t, x, y, z), ans);
  EXPECT_FALSE(f.is_spatial());
}

TEST(StringTimeCoordFunction, spatial_dim)
{
  ASSERT_EQ(StringTimeCoordFunction{"3*t + x"}.spatial_dim(), 1);
  ASSERT_EQ(StringTimeCoordFunction{"y"}.spatial_dim(), 2);
  ASSERT_EQ(StringTimeCoordFunction{"x + z"}.spatial_dim(), 3);
  ASSERT_EQ(StringTimeCoordFunction{"y+z"}.spatial_dim(), 3);
}

} // namespace sierra::nalu
