#include "gtest/gtest.h"

#include "user_functions/TabulatedTemperatureAuxFunction.h"

namespace sierra::nalu {
TEST(TabulatedTemperatureAuxFunction, end_of_range)
{
  TabulatedTemperatureAuxFunction func({0, 1, 2}, {2, 3, 4});

  double coords[3] = {1, 2, 3};
  double temperature;
  func.do_evaluate(coords, 0., 3, 1, &temperature, 1, 0, 1);
  ASSERT_DOUBLE_EQ(temperature, 4.);
}

TEST(TabulatedTemperatureAuxFunction, matches_at_table_entry)
{
  TabulatedTemperatureAuxFunction func({0, 1, 2}, {2, 3, 4});

  double coords[3] = {1, 2, 1};
  double temperature;
  func.do_evaluate(coords, 0., 3, 1, &temperature, 1, 0, 1);
  ASSERT_DOUBLE_EQ(temperature, 3.);
}

TEST(TabulatedTemperatureAuxFunction, averages_halfway_between_points)
{
  TabulatedTemperatureAuxFunction func({0, 1, 2}, {2, 3, 4});

  double coords[3] = {1, 2, 1.5};
  double temperature;
  func.do_evaluate(coords, 0., 3, 1, &temperature, 1, 0, 1);
  ASSERT_DOUBLE_EQ(temperature, (3 + 4) / 2.);
}

TEST(TabulatedTemperatureAuxFunction, before_ranges)
{
  TabulatedTemperatureAuxFunction func({0, 1, 2}, {2, 3, 4});

  double coords[3] = {1, 2, -1};
  double temperature;
  func.do_evaluate(coords, 0., 3, 1, &temperature, 1, 0, 1);
  ASSERT_DOUBLE_EQ(temperature, 2);
}

} // namespace sierra::nalu