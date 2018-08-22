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

namespace sierra{
namespace nalu{

TensorProductQuadratureRule::TensorProductQuadratureRule(std::string type, int polyOrder)
{
  scsLoc_ = gauss_legendre_rule(polyOrder).first;

  // pad the scs surfaces with the end-points of the element (-1 and +1)
  scsEndLoc_ = pad_end_points(scsLoc_, -1, +1);

  if (type == "GaussLegendre") {
    numQuad_ =  (polyOrder % 2 == 0) ? polyOrder/2 + 1 : (polyOrder+1)/2;
    std::tie(abscissae_, weights_) = gauss_legendre_rule(numQuad_);
    double isoparametricFactor = 0.5;
    for (auto& weight : weights_) {
      weight *= isoparametricFactor;
    }
    useSGL_ = false;
  }
  else if (type == "SGL") {
    int numNodes = polyOrder+1;
    numQuad_ = 1; // only 1 quadrature point per scv
    std::tie(abscissae_, weights_) = SGL_quadrature_rule(numNodes, scsEndLoc_.data());
    useSGL_ = true;
  }
  else {
    ThrowRequireMsg(false, "Invalid quadrature type");
  }
}
//--------------------------------------------------------------------------
TensorProductQuadratureRule::TensorProductQuadratureRule(
  std::string type,
  int numQuad,
  std::vector<double>& scsLocs)
: numQuad_(numQuad)
{
  // pad the scs surfaces with the end-points of the element (-1 and +1)
  scsEndLoc_.resize(scsLocs.size()+2);
  scsEndLoc_[0] = -1.0;

  for (unsigned j = 0; j < scsLocs.size();++j) {
    scsEndLoc_[j+1] = scsLocs[j];
  }
  scsEndLoc_[scsLocs.size()+1] = +1.0;

  if (type == "GaussLegendre") {
    std::tie(abscissae_, weights_) = gauss_legendre_rule(numQuad);
    double isoparametricFactor = 0.5;
    for (auto& weight : weights_) {
      weight *= isoparametricFactor;
    }
    useSGL_ = false;
  }
  else if (type == "SGL") {
    int numNodes = scsLocs.size()+1;
    numQuad_ = 1; // only 1 quadrature point per scv
    std::tie(abscissae_, weights_) = SGL_quadrature_rule(numNodes, scsEndLoc_.data());
    useSGL_ = true;
  }
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::isoparametric_mapping(
  const double b,
  const double a,
  const double xi) const
{
  return (0.5*(xi*(b-a) + (a+b)));
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::integration_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{
  double location1D;
  if (!useSGL_) {
    location1D = isoparametric_mapping(
      scsEndLoc_[nodeOrdinal+1],
      scsEndLoc_[nodeOrdinal],
      abscissae_[gaussPointOrdinal] );
  }
  else {
    location1D = abscissae_[nodeOrdinal];
  }
  return location1D;
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::integration_point_weight(
  int s1Node, int s2Node, int s3Node,
  int s1Ip, int s2Ip, int s3Ip) const
{
  double weight;
  if (!useSGL_) {
    const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
    const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
    const double Ls3 = scsEndLoc_[s3Node+1]-scsEndLoc_[s3Node];
    const double isoparametricArea = Ls1 * Ls2 * Ls3;

    weight = isoparametricArea
           * weights_[s1Ip]
           * weights_[s2Ip]
           * weights_[s3Ip];
   }
   else {
     weight = 1.0; // weights will be applied in the assembly
   }
   return weight;
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::integration_point_weight(
  int s1Node, int s2Node,
  int s1Ip, int s2Ip) const
{
  //surface integration
  double weight;
  if (!useSGL_) {
    const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
    const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
    const double isoparametricArea = Ls1 * Ls2;
    weight = isoparametricArea * weights_[s1Ip] * weights_[s2Ip];
  }
  else {
    weight = 1.0; // weights will be applied in the assembly
  }
  return weight;
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::integration_point_weight(int s1Node, int s1Ip) const
{
  double weight;
  if (!useSGL_) {
    const double isoparametricLength = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
    weight = isoparametricLength * weights_[s1Ip];
  }
  else {
    weight = 1.0; // weights will be applied in the assembly
  }
  return weight;
}

}  // namespace nalu
} // namespace sierra
