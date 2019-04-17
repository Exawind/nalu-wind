/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/QuadratureRule.h>

#include <stk_util/util/ReportHandler.hpp>

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace sierra{
namespace nalu{

int gauss_points_required_for_order(int order) { return (order % 2 == 0) ? order / 2 + 1 : (order + 1) / 2; }

int integration_points_required(int poly, QuadratureRuleType quadType)
{
  switch (quadType)
  {
    case QuadratureRuleType::MINIMUM_GAUSS: return gauss_points_required_for_order(poly);
    case QuadratureRuleType::DOUBLE_GAUSS: return gauss_points_required_for_order(2*poly);
  }
  ThrowAssert(false);
  return -1;
}

TensorProductQuadratureRule::TensorProductQuadratureRule(int polyOrder, QuadratureRuleType quadType)
: numQuad_(integration_points_required(polyOrder, quadType)),
  scsLoc_("scs_locations", polyOrder),
  scsEndLoc_("scs_locations_with_end_points", polyOrder + 2),
  abscissae_("gauss_abscissae", numQuad_),
  weights_("gauss_point_weights", numQuad_)
{
  ThrowAssert(numQuad_ > 0);

  auto scsLoc = gauss_legendre_rule(polyOrder).first;
  for (int k = 0; k < polyOrder; ++k) {
    scsLoc_[k] = scsLoc[k];
  }

  auto scsEndLoc = pad_end_points(scsLoc, -1, +1);
  for (int k = 0; k < polyOrder + 2; ++k) {
    scsEndLoc_[k] = scsEndLoc[k];
  }

  std::vector<double> abscissae;
  std::vector<double> weights;
  std::tie(abscissae, weights) = gauss_legendre_rule(numQuad_);
  const double isoParametricFactor = 0.5;

  for (int k = 0; k < numQuad_; ++k) {
    abscissae_[k] = abscissae[k];
    weights_[k] = isoParametricFactor * weights[k];
  }

}

double isoparametric_map(double xl, double xr, double x) { return 0.5 * (x * (xr - xl) +  (xl + xr)); }

double TensorProductQuadratureRule::integration_point_location(int nodeOrd, int ipOrd) const
{
  return isoparametric_map(scsEndLoc_[nodeOrd], scsEndLoc_[nodeOrd + 1], abscissae_[ipOrd]);
}

double TensorProductQuadratureRule::integration_point_weight(int s1Node, int s1Ip) const
{
  const double Ls1 = (scsEndLoc_[s1Node + 1] - scsEndLoc_[s1Node]) * weights_[s1Ip];
  const double weightedIsoparametricLength = Ls1;
  return weightedIsoparametricLength;
}

double TensorProductQuadratureRule::integration_point_weight(int s1Node, int s2Node, int s1Ip, int s2Ip) const
{
  const double Ls1 = (scsEndLoc_[s1Node + 1] - scsEndLoc_[s1Node]) * weights_[s1Ip];
  const double Ls2 = (scsEndLoc_[s2Node + 1] - scsEndLoc_[s2Node]) * weights_[s2Ip];
  const double weightedIsoparametricArea = Ls1 * Ls2;
  return weightedIsoparametricArea;
}

double TensorProductQuadratureRule::integration_point_weight(
  int s1Node, int s2Node, int s3Node,
  int s1Ip, int s2Ip, int s3Ip) const
{
  const double Ls1 = (scsEndLoc_[s1Node + 1] - scsEndLoc_[s1Node]) * weights_[s1Ip];
  const double Ls2 = (scsEndLoc_[s2Node + 1] - scsEndLoc_[s2Node]) * weights_[s2Ip];
  const double Ls3 = (scsEndLoc_[s3Node + 1] - scsEndLoc_[s3Node]) * weights_[s3Ip];
  const double weightedIsoparametricVolume = Ls1 * Ls2 * Ls3;
  return weightedIsoparametricVolume;
}

}  // namespace nalu
} // namespace sierra
