/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductQuadratureRule_h
#define TensorProductQuadratureRule_h

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

namespace sierra{
namespace nalu{

enum class QuadratureRuleType {MINIMUM_GAUSS = 0, DOUBLE_GAUSS = 1};

class TensorProductQuadratureRule
{
public:
  TensorProductQuadratureRule(int polyOrder, QuadratureRuleType = QuadratureRuleType::MINIMUM_GAUSS);
  ~TensorProductQuadratureRule() = default;

  int num_quad() const { return numQuad_; }
  double abscissa(int j) const { return abscissae_[j]; };
  double weights(int j) const { return weights_[j]; };
  double scs_loc(int j) const { return scsLoc_[j]; };
  double scs_end_loc(int j) const { return scsEndLoc_[j]; };

  Kokkos::View<double*> abscissa() const { return abscissae_; };
  Kokkos::View<double*> weights() const { return weights_; };
  Kokkos::View<double*> scs_loc() const { return scsLoc_; };
  Kokkos::View<double*> scs_end_loc() const { return scsEndLoc_; };


  double integration_point_location(int nodeOrdinal, int gaussPointOrdinal) const;
  double integration_point_weight( int s1Node, int s2Node, int s3Node, int s1Ip, int s2Ip, int s3Ip) const;
  double integration_point_weight(int s1Node, int s2Node, int s1Ip, int s2Ip) const;
  double integration_point_weight(int s1Node, int s1Ip) const;
private:
  const int numQuad_;
  Kokkos::View<double*> scsLoc_;
  Kokkos::View<double*> scsEndLoc_;
  Kokkos::View<double*> abscissae_;
  Kokkos::View<double*> weights_;

};

} // namespace nalu
} // namespace Sierra

#endif
