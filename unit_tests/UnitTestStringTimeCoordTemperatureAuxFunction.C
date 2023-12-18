#include "gtest/gtest.h"

#include "user_functions/StringTimeCoordTemperatureAuxFunction.h"

namespace sierra::nalu {
TEST(StringTimeCoordTemperatureAuxFunction, only_z)
{
  StringTimeCoordTemperatureAuxFunction func("z");

  double coords[3] = {1, 2, 3};
  double temperature;
  func.do_evaluate(coords, 0., 3, 1, &temperature, 1, 0, 1);
  ASSERT_DOUBLE_EQ(temperature, 3.);
}

TEST(StringTimeCoordTemperatureAuxFunction, only_y)
{
  StringTimeCoordTemperatureAuxFunction func("y");

  double coords[3] = {1, 2, 1};
  double temperature;
  func.do_evaluate(coords, 0., 3, 1, &temperature, 1, 0, 1);
  ASSERT_DOUBLE_EQ(temperature, 2.);
}

TEST(StringTimeCoordTemperatureAuxFunction, full)
{
  StringTimeCoordTemperatureAuxFunction func("log(1+t) + exp(x)*log(y) + z");

  double coords[3] = {1, 2, 1};
  double temperature;
  func.do_evaluate(coords, 0., 3, 1, &temperature, 1, 0, 1);
  ASSERT_DOUBLE_EQ(temperature, 1 + exp(1) * log(2));
}

} // namespace sierra::nalu