// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COMPILE_TIME_ELEMENTS_H
#define COMPILE_TIME_ELEMENTS_H

#include "AlgTraits.h"
#include "master_element/IntegrationRules.h"
#include "master_element/MasterElementFunctions.h"
#include "ArrayND.h"
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include "ElementBasis.h"

namespace sierra::nalu::impl {

template <typename BasisT, typename IntgT>
struct CVFEMData
{
  using basis_t = BasisT;
  using intg_t = IntgT;
  static constexpr auto scs_interp = utils::interpolants<basis_t>(intg_t::scs);
  static constexpr auto scv_interp = utils::interpolants<basis_t>(intg_t::scv);
  static constexpr auto scs_deriv = utils::deriv_coeffs<basis_t>(intg_t::scs);
  static constexpr auto scv_deriv = utils::deriv_coeffs<basis_t>(intg_t::scv);
};

template <typename AlgTraits, QuadType q>
struct ElemDataSelector
{
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsTri3_2D, q>
{
  using elem_data_t = CVFEMData<Tri3Basis, TriIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsQuad4_2D, q>
{
  using elem_data_t = CVFEMData<Quad42DBasis, QuadIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsHex8, q>
{
  using elem_data_t = CVFEMData<Hex8Basis, HexIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsTet4, q>
{
  using elem_data_t = CVFEMData<Tet4Basis, TetIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsWed6, q>
{
  using elem_data_t = CVFEMData<Wed6Basis, WedIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsQuad4, q>
{
  using elem_data_t = CVFEMData<Quad42DBasis, QuadIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsTri3, q>
{
  using elem_data_t = CVFEMData<Tri3Basis, TriIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsEdge_2D, q>
{
  using elem_data_t = CVFEMData<LineBasis, EdgeIntegrationRule<q>>;
};

template <QuadType q>
struct ElemDataSelector<AlgTraitsPyr5, q>
{
  using elem_data_t = CVFEMData<
    std::conditional_t<q == QuadType::MID, Pyr5Basis, Pyr5DegenHexBasis>,
    PyrIntegrationRule<q>>;
};

} // namespace sierra::nalu::impl

namespace sierra::nalu {

template <typename AlgTraits, QuadType q>
using elem_data_t = typename impl::ElemDataSelector<AlgTraits, q>::elem_data_t;

enum class QuadRank { SCS, SCV };
} // namespace sierra::nalu

namespace sierra::nalu {
template <typename AlgTraits, QuadRank rank>
std::enable_if_t<rank == QuadRank::SCS, double>
shape_fcn(QuadType quad, int ip, int n)
{
  if (quad == QuadType::SHIFTED) {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::SHIFTED>::scs_interp;
    return shp(ip, n);
  } else {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::MID>::scs_interp;
    return shp(ip, n);
  }
}

template <typename AlgTraits, QuadRank rank>
std::enable_if_t<rank == QuadRank::SCV, double>
shape_fcn(QuadType quad, int ip, int n)
{
  if (quad == QuadType::SHIFTED) {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::SHIFTED>::scv_interp;
    return shp(ip, n);
  } else {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::MID>::scv_interp;
    return shp(ip, n);
  }
}

template <typename AlgTraits, QuadRank rank>
KOKKOS_FUNCTION std::enable_if_t<
  rank == QuadRank::SCS,
  decltype(elem_data_t<AlgTraits, QuadType::MID>::scs_interp)>
shape_fcn(QuadType quad)
{
  if (quad == QuadType::SHIFTED) {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::SHIFTED>::scs_interp;
    return shp;
  } else {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::MID>::scs_interp;
    return shp;
  }
}

template <typename AlgTraits, QuadRank rank>
KOKKOS_FUNCTION std::enable_if_t<
  rank == QuadRank::SCV,
  decltype(elem_data_t<AlgTraits, QuadType::MID>::scv_interp)>
shape_fcn(QuadType quad)
{
  if (quad == QuadType::SHIFTED) {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::SHIFTED>::scv_interp;
    return shp;
  } else {
    static constexpr auto shp =
      elem_data_t<AlgTraits, QuadType::MID>::scv_interp;
    return shp;
  }
}

namespace impl {
template <
  typename AlgTraits,
  QuadRank rank,
  QuadType q,
  typename CoordViewT,
  typename GradViewT>
KOKKOS_FUNCTION std::enable_if_t<rank == QuadRank::SCS, void>
grad_op(const CoordViewT& x, const GradViewT& grad_coeff)
{
  static constexpr auto deriv = elem_data_t<AlgTraits, q>::scs_deriv;
  generic_grad_op<AlgTraits>(deriv, x, grad_coeff);
}

template <
  typename AlgTraits,
  QuadRank rank,
  QuadType q,
  typename CoordViewT,
  typename GradViewT>
KOKKOS_FUNCTION std::enable_if_t<rank == QuadRank::SCV, void>
grad_op(const CoordViewT& x, const GradViewT& grad_coeff)
{
  static constexpr auto deriv = elem_data_t<AlgTraits, q>::scv_deriv;
  generic_grad_op<AlgTraits>(deriv, x, grad_coeff);
}

} // namespace impl

KOKKOS_INLINE_FUNCTION QuadType
use_shifted_quad(bool shifted)
{
  return shifted ? QuadType::SHIFTED : QuadType::MID;
}

// çç

} // namespace sierra::nalu

#endif