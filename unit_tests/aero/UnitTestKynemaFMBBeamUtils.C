#include <gtest/gtest.h>

#include <aero/fmb/KynemaFMBBeamUtils.h>

#include <array>
#include <cmath>

namespace {

using namespace sierra::kynema_ugf;

constexpr double kTolerance = 1.e-12;

TEST(KynemaFMBBeamUtils, RotatesVectorsAndComputesCrossProduct)
{
  const std::array<double, 4> rotation = {
    std::sqrt(0.5), 0.0, 0.0, std::sqrt(0.5)};
  const std::array<double, 3> vector = {1.0, 0.0, 0.0};
  std::array<double, 3> rotated{};
  std::array<double, 3> recovered{};
  std::array<double, 3> crossProduct{};

  RotateVectorByQuaternion(rotation, vector, rotated);
  RotateVectorByQuaternionInv(rotation, rotated, recovered);
  CrossProduct3(vector, rotated, crossProduct);

  EXPECT_NEAR(0.0, rotated[0], kTolerance);
  EXPECT_NEAR(1.0, rotated[1], kTolerance);
  EXPECT_NEAR(0.0, rotated[2], kTolerance);
  EXPECT_NEAR(vector[0], recovered[0], kTolerance);
  EXPECT_NEAR(vector[1], recovered[1], kTolerance);
  EXPECT_NEAR(vector[2], recovered[2], kTolerance);
  EXPECT_NEAR(0.0, crossProduct[0], kTolerance);
  EXPECT_NEAR(0.0, crossProduct[1], kTolerance);
  EXPECT_NEAR(1.0, crossProduct[2], kTolerance);
}

TEST(KynemaFMBBeamUtils, ComputesBarycentricWeightsAndBasisValues)
{
  const std::array<double, 3> nodes = {-1.0, 0.0, 1.0};
  std::array<double, 3> barycentricWeights{};
  std::array<double, 3> weights{};

  ComputeBarycentricWeights(nodes.data(), 3, barycentricWeights.data());
  LagrangePolynomialInterpWeights(
    0.5, nodes.data(), barycentricWeights.data(), 3, weights.data());

  EXPECT_NEAR(0.5, barycentricWeights[0], kTolerance);
  EXPECT_NEAR(-1.0, barycentricWeights[1], kTolerance);
  EXPECT_NEAR(0.5, barycentricWeights[2], kTolerance);
  EXPECT_NEAR(-0.125, weights[0], kTolerance);
  EXPECT_NEAR(0.75, weights[1], kTolerance);
  EXPECT_NEAR(0.375, weights[2], kTolerance);

  LagrangePolynomialInterpWeights(
    0.0, nodes.data(), barycentricWeights.data(), 3, weights.data());
  EXPECT_DOUBLE_EQ(0.0, weights[0]);
  EXPECT_DOUBLE_EQ(1.0, weights[1]);
  EXPECT_DOUBLE_EQ(0.0, weights[2]);
}

TEST(KynemaFMBBeamUtils, ComputesBasisDerivativesAtInteriorAndNodalPoints)
{
  const std::array<double, 3> nodes = {-1.0, 0.0, 1.0};
  std::array<double, 3> barycentricWeights{};
  std::array<double, 3> weights{};
  std::array<double, 3> derivatives{};

  ComputeBarycentricWeights(nodes.data(), 3, barycentricWeights.data());
  LagrangePolynomialInterpWeightsAndDerivatives(
    0.5, nodes.data(), barycentricWeights.data(), 3, weights.data(),
    derivatives.data());

  EXPECT_NEAR(-0.125, weights[0], kTolerance);
  EXPECT_NEAR(0.75, weights[1], kTolerance);
  EXPECT_NEAR(0.375, weights[2], kTolerance);
  EXPECT_NEAR(0.0, derivatives[0], kTolerance);
  EXPECT_NEAR(-1.0, derivatives[1], kTolerance);
  EXPECT_NEAR(1.0, derivatives[2], kTolerance);

  LagrangePolynomialInterpWeightsAndDerivatives(
    0.0, nodes.data(), barycentricWeights.data(), 3, weights.data(),
    derivatives.data());
  EXPECT_DOUBLE_EQ(0.0, weights[0]);
  EXPECT_DOUBLE_EQ(1.0, weights[1]);
  EXPECT_DOUBLE_EQ(0.0, weights[2]);
  EXPECT_NEAR(-0.5, derivatives[0], kTolerance);
  EXPECT_NEAR(0.0, derivatives[1], kTolerance);
  EXPECT_NEAR(0.5, derivatives[2], kTolerance);
}

TEST(KynemaFMBBeamUtils, ComputesShapeFunctionsAndInterpolatesFields)
{
  const std::array<double, 3> nodes = {-1.0, 0.0, 1.0};
  std::array<double, 3> barycentricWeights{};
  std::array<double, 3> scratchWeights{};
  std::array<double, 9> shapeFunctions{};
  const std::array<double, 4> field = {0.0, 10.0, 2.0, 20.0};
  const std::array<double, 2> linearNodes = {-1.0, 1.0};
  std::array<double, 2> linearBarycentricWeights{};
  std::array<double, 2> interpolated{};

  ComputeBarycentricWeights(nodes.data(), 3, barycentricWeights.data());
  ComputeShapeFunctionValues(
    nodes.data(), 3, nodes.data(), barycentricWeights.data(), 3,
    scratchWeights.data(), shapeFunctions.data());
  ComputeBarycentricWeights(
    linearNodes.data(), 2, linearBarycentricWeights.data());
  InterpolateFieldAtPoint(
    0.25, linearNodes.data(), field.data(), 2, 2,
    linearBarycentricWeights.data(), scratchWeights.data(),
    interpolated.data());

  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      EXPECT_DOUBLE_EQ(
        row == column ? 1.0 : 0.0, shapeFunctions[3 * row + column]);
    }
  }
  EXPECT_NEAR(1.25, interpolated[0], kTolerance);
  EXPECT_NEAR(16.25, interpolated[1], kTolerance);
}

TEST(KynemaFMBBeamUtils, EvaluatesBladeDistanceAndDerivative)
{
  const std::array<double, 2> nodes = {-1.0, 1.0};
  const std::array<double, 14> positions = {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                            1.0,  0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  const std::array<double, 3> query = {0.25, 2.0, -1.0};
  std::array<double, 2> barycentricWeights{};
  std::array<double, 2> scratchWeights{};
  std::array<double, 2> scratchDerivatives{};
  std::array<double, 3> interpolatedPosition{};

  ComputeBarycentricWeights(nodes.data(), 2, barycentricWeights.data());
  const double distanceSquared = BladePointDistanceSquaredAtXi(
    0.25, query.data(), nodes.data(), positions.data(), 2,
    barycentricWeights.data(), scratchWeights.data(),
    interpolatedPosition.data());
  const double derivative = BladePointFPrimeAtXi(
    0.0, query.data(), nodes.data(), positions.data(), 2,
    barycentricWeights.data(), scratchWeights.data(),
    scratchDerivatives.data());

  EXPECT_NEAR(5.0, distanceSquared, kTolerance);
  EXPECT_NEAR(0.25, interpolatedPosition[0], kTolerance);
  EXPECT_NEAR(0.0, interpolatedPosition[1], kTolerance);
  EXPECT_NEAR(0.0, interpolatedPosition[2], kTolerance);
  EXPECT_NEAR(-0.5, derivative, kTolerance);
  EXPECT_NEAR(
    5.0, SquaredDistance3(interpolatedPosition.data(), query.data()),
    kTolerance);
}

TEST(KynemaFMBBeamUtils, FindsInteriorAndEndpointClosestPoints)
{
  const std::array<double, 2> nodes = {-1.0, 1.0};
  const std::array<double, 14> positions = {-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                            1.0,  0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  std::array<double, 2> barycentricWeights{};
  std::array<double, 2> scratchWeights{};
  std::array<double, 2> scratchDerivatives{};
  std::array<double, 3> closestPosition{};
  double closestXi = 0.0;
  double distanceSquared = 0.0;

  ComputeBarycentricWeights(nodes.data(), 2, barycentricWeights.data());
  const std::array<double, 3> interiorQuery = {0.25, 2.0, -1.0};
  FindClosestPointOnBlade(
    interiorQuery.data(), nodes.data(), positions.data(), 2,
    barycentricWeights.data(), scratchWeights.data(), scratchDerivatives.data(),
    closestXi, closestPosition.data(), distanceSquared);
  EXPECT_NEAR(0.25, closestXi, kTolerance);
  EXPECT_NEAR(0.25, closestPosition[0], kTolerance);
  EXPECT_NEAR(5.0, distanceSquared, kTolerance);

  const std::array<double, 3> endpointQuery = {3.0, 2.0, 0.0};
  FindClosestPointOnBlade(
    endpointQuery.data(), nodes.data(), positions.data(), 2,
    barycentricWeights.data(), scratchWeights.data(), scratchDerivatives.data(),
    closestXi, closestPosition.data(), distanceSquared);
  EXPECT_DOUBLE_EQ(1.0, closestXi);
  EXPECT_NEAR(1.0, closestPosition[0], kTolerance);
  EXPECT_NEAR(8.0, distanceSquared, kTolerance);
}

} // namespace
