#include <gtest/gtest.h>

#include <aero/fmb/KynemaFMBBeamUtils.h>

#include <array>
#include <cmath>
#include <cstddef>

namespace {

using namespace sierra::kynema_ugf;

constexpr double kTolerance = 1.e-12;
constexpr std::size_t kInterpolationNodes = 11;

std::array<double, kInterpolationNodes>
make_lagrange_nodes()
{
  std::array<double, kInterpolationNodes> nodes{};
  for (std::size_t node = 0; node < nodes.size(); ++node) {
    nodes[node] = -1.0 + 2.0 * static_cast<double>(node) /
                            static_cast<double>(nodes.size() - 1);
  }
  return nodes;
}

std::array<double, 7 * kInterpolationNodes>
make_straight_blade_positions(
  const std::array<double, kInterpolationNodes>& nodes)
{
  std::array<double, 7 * kInterpolationNodes> positions{};
  for (std::size_t node = 0; node < nodes.size(); ++node) {
    positions[7 * node] = nodes[node];
    positions[7 * node + 3] = 1.0;
  }
  return positions;
}

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
  const auto nodes = make_lagrange_nodes();
  const std::array<double, kInterpolationNodes> expectedWeightRatios = {
    1.0, -10.0, 45.0, -120.0, 210.0, -252.0,
    210.0, -120.0, 45.0, -10.0, 1.0};
  std::array<double, kInterpolationNodes> barycentricWeights{};
  std::array<double, kInterpolationNodes> weights{};

  ComputeBarycentricWeights(
    nodes.data(), kInterpolationNodes, barycentricWeights.data());
  LagrangePolynomialInterpWeights(
    0.5, nodes.data(), barycentricWeights.data(), kInterpolationNodes,
    weights.data());

  for (std::size_t node = 0; node < kInterpolationNodes; ++node) {
    EXPECT_NEAR(
      expectedWeightRatios[node],
      barycentricWeights[node] / barycentricWeights[0], kTolerance);
  }

  double weightSum = 0.0;
  double tenthOrderValue = 0.0;
  for (std::size_t node = 0; node < kInterpolationNodes; ++node) {
    weightSum += weights[node];
    tenthOrderValue += weights[node] * std::pow(nodes[node], 10);
  }
  EXPECT_NEAR(1.0, weightSum, kTolerance);
  EXPECT_NEAR(std::pow(0.5, 10), tenthOrderValue, kTolerance);

  LagrangePolynomialInterpWeights(
    0.0, nodes.data(), barycentricWeights.data(), kInterpolationNodes,
    weights.data());
  for (std::size_t node = 0; node < kInterpolationNodes; ++node) {
    EXPECT_DOUBLE_EQ(node == 5 ? 1.0 : 0.0, weights[node]);
  }
}

TEST(KynemaFMBBeamUtils, ComputesBasisDerivativesAtInteriorAndNodalPoints)
{
  const auto nodes = make_lagrange_nodes();
  std::array<double, kInterpolationNodes> barycentricWeights{};
  std::array<double, kInterpolationNodes> weights{};
  std::array<double, kInterpolationNodes> derivatives{};

  ComputeBarycentricWeights(
    nodes.data(), kInterpolationNodes, barycentricWeights.data());
  LagrangePolynomialInterpWeightsAndDerivatives(
    0.5, nodes.data(), barycentricWeights.data(), kInterpolationNodes,
    weights.data(), derivatives.data());

  double weightSum = 0.0;
  double derivativeSum = 0.0;
  double tenthOrderDerivative = 0.0;
  for (std::size_t node = 0; node < kInterpolationNodes; ++node) {
    weightSum += weights[node];
    derivativeSum += derivatives[node];
    tenthOrderDerivative += derivatives[node] * std::pow(nodes[node], 10);
  }
  EXPECT_NEAR(1.0, weightSum, kTolerance);
  EXPECT_NEAR(0.0, derivativeSum, kTolerance);
  EXPECT_NEAR(10.0 * std::pow(0.5, 9), tenthOrderDerivative, kTolerance);

  LagrangePolynomialInterpWeightsAndDerivatives(
    0.0, nodes.data(), barycentricWeights.data(), kInterpolationNodes,
    weights.data(), derivatives.data());
  double nodalDerivativeSum = 0.0;
  double linearDerivative = 0.0;
  for (std::size_t node = 0; node < kInterpolationNodes; ++node) {
    EXPECT_DOUBLE_EQ(node == 5 ? 1.0 : 0.0, weights[node]);
    nodalDerivativeSum += derivatives[node];
    linearDerivative += derivatives[node] * nodes[node];
  }
  EXPECT_NEAR(0.0, nodalDerivativeSum, kTolerance);
  EXPECT_NEAR(1.0, linearDerivative, kTolerance);
}

TEST(KynemaFMBBeamUtils, EvaluatesBladeDistanceAndDerivative)
{
  const auto nodes = make_lagrange_nodes();
  const auto positions = make_straight_blade_positions(nodes);
  const std::array<double, 3> query = {0.25, 2.0, -1.0};
  std::array<double, kInterpolationNodes> barycentricWeights{};
  std::array<double, kInterpolationNodes> scratchWeights{};
  std::array<double, kInterpolationNodes> scratchDerivatives{};
  std::array<double, 3> interpolatedPosition{};

  ComputeBarycentricWeights(
    nodes.data(), kInterpolationNodes, barycentricWeights.data());
  const double distanceSquared = BladePointDistanceSquaredAtXi(
    0.25, query.data(), nodes.data(), positions.data(), kInterpolationNodes,
    barycentricWeights.data(), scratchWeights.data(),
    interpolatedPosition.data());
  const double derivative = BladePointFPrimeAtXi(
    0.0, query.data(), nodes.data(), positions.data(), kInterpolationNodes,
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
  const auto nodes = make_lagrange_nodes();
  const auto positions = make_straight_blade_positions(nodes);
  std::array<double, kInterpolationNodes> barycentricWeights{};
  std::array<double, kInterpolationNodes> scratchWeights{};
  std::array<double, kInterpolationNodes> scratchDerivatives{};
  std::array<double, 3> closestPosition{};
  double closestXi = 0.0;
  double distanceSquared = 0.0;

  ComputeBarycentricWeights(
    nodes.data(), kInterpolationNodes, barycentricWeights.data());
  const std::array<double, 3> interiorQuery = {0.25, 2.0, -1.0};
  FindClosestPointOnBlade(
    interiorQuery.data(), nodes.data(), positions.data(), kInterpolationNodes,
    barycentricWeights.data(), scratchWeights.data(), scratchDerivatives.data(),
    closestXi, closestPosition.data(), distanceSquared);
  EXPECT_NEAR(0.25, closestXi, kTolerance);
  EXPECT_NEAR(0.25, closestPosition[0], kTolerance);
  EXPECT_NEAR(5.0, distanceSquared, kTolerance);

  const std::array<double, 3> endpointQuery = {3.0, 2.0, 0.0};
  FindClosestPointOnBlade(
    endpointQuery.data(), nodes.data(), positions.data(), kInterpolationNodes,
    barycentricWeights.data(), scratchWeights.data(), scratchDerivatives.data(),
    closestXi, closestPosition.data(), distanceSquared);
  EXPECT_DOUBLE_EQ(1.0, closestXi);
  EXPECT_NEAR(1.0, closestPosition[0], kTolerance);
  EXPECT_NEAR(8.0, distanceSquared, kTolerance);
}

} // namespace
