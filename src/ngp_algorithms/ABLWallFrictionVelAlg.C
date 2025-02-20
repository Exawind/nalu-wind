// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/ABLWallFrictionVelAlg.h"

#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "wind_energy/MoninObukhov.h"

#include "stk_mesh/base/Field.hpp"
#include <stk_mesh/base/NgpMesh.hpp>

namespace sierra {
namespace nalu {

namespace {

template <typename PsiFunc>
KOKKOS_FUNCTION double
calc_utau(
  const double uh,
  const double zh,
  const double term1,
  const double Tflux,
  const double Lfac,
  const double kappa,
  PsiFunc psi_func,
  const double psi_fac)
{
  const double eps = 1.0e-8;
  const double convTol = 1.0e-7;
  const double perturb = 1.0e-3;
  const int maxIters = 40;

  // Return utau as epsilon if the velocity at surface is zero
  if (stk::math::abs(uh) < eps)
    return eps;

  double utau0, utau1;
  // Proceed with normal computations
  utau0 = kappa * uh / term1;
  if (Tflux > 0.0)
    utau0 *= 3.0;

  utau1 = (1.0 + perturb) * utau0;

  bool converged = false;
  double utau = utau0;
  for (int k = 0; k < maxIters; ++k) {
    double L0 = utau0 * utau0 * utau0 * Lfac;
    double L1 = utau1 * utau1 * utau1 * Lfac;

    double sgnq = (Tflux > 0.0) ? 1.0 : -1.0;
    L0 = -sgnq * stk::math::max(1.0e-10, stk::math::abs(L0));
    L1 = -sgnq * stk::math::max(1.1e-10, stk::math::abs(L1));

    const double znorm0 = zh / L0;
    const double znorm1 = zh / L1;

    const double denom0 = term1 - psi_func(znorm0, psi_fac);
    const double denom1 = term1 - psi_func(znorm1, psi_fac);

    const double f0 = utau0 - uh * kappa / denom0;
    const double f1 = utau1 - uh * kappa / denom1;

    double dutau = utau1 - utau0;
    dutau = (dutau > 0.0) ? (stk::math::max(1.0e-15, dutau))
                          : (stk::math::min(-1.0e-15, dutau));

    double fprime = (f1 - f0) / dutau;
    fprime = (fprime > 0.0) ? (stk::math::max(1.0e-15, fprime))
                            : (stk::math::min(-1.0e-15, fprime));

    const double utau_tmp = utau1;
    utau1 = utau0 - f0 / fprime;
    utau0 = utau_tmp;

    if (stk::math::abs(f1) < convTol) {
      converged = true;
      utau = stk::math::max(0.0, utau1);
      break;
    }
  }

  if (!converged)
    printf("Issue with utau");
  return utau;
}

} // namespace

template <typename BcAlgTraits>
ABLWallFrictionVelAlg<BcAlgTraits>::ABLWallFrictionVelAlg(
  Realm& realm,
  stk::mesh::Part* part,
  WallFricVelAlgDriver& algDriver,
  const bool useShifted,
  const double gravity,
  const double z0,
  const double Tref,
  const double kappa)
  : Algorithm(realm, part),
    algDriver_(algDriver),
    faceData_(realm.meta_data()),
    velocityNp1_(
      get_field_ordinal(realm.meta_data(), "velocity", stk::mesh::StateNP1)),
    bcVelocity_(get_field_ordinal(realm.meta_data(), "wall_velocity_bc")),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    bcHeatFlux_(get_field_ordinal(realm.meta_data(), "heat_flux_bc")),
    specificHeat_(get_field_ordinal(realm.meta_data(), "specific_heat")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallFricVel_(get_field_ordinal(
      realm.meta_data(),
      "wall_friction_velocity_bip",
      realm.meta_data().side_rank())),
    wallNormDist_(get_field_ordinal(
      realm.meta_data(),
      "wall_normal_distance_bip",
      realm.meta_data().side_rank())),
    gravity_(gravity),
    z0_(z0),
    Tref_(Tref),
    kappa_(kappa),
    useShifted_(useShifted),
    meFC_(
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
        BcAlgTraits::topo_))
{
  faceData_.add_cvfem_face_me(meFC_);

  faceData_.add_coordinates_field(
    get_field_ordinal(realm_.meta_data(), realm_.get_coordinates_name()),
    BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  faceData_.add_gathered_nodal_field(bcVelocity_, BcAlgTraits::nDim_);
  faceData_.add_gathered_nodal_field(density_, 1);
  faceData_.add_gathered_nodal_field(bcHeatFlux_, 1);
  faceData_.add_gathered_nodal_field(specificHeat_, 1);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(wallNormDist_, BcAlgTraits::numFaceIp_);
}

template <typename BcAlgTraits>
void
ABLWallFrictionVelAlg<BcAlgTraits>::execute()
{
  namespace mo = abl_monin_obukhov;
  using ElemSimdData = sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpUtau = fieldMgr.template get_field<double>(wallFricVel_);

  // Bring class members into local scope for device capture
  const unsigned velID = velocityNp1_;
  const unsigned bcVelID = bcVelocity_;
  const unsigned rhoID = density_;
  const unsigned bcHeatFluxID = bcHeatFlux_;
  const unsigned specHeatID = specificHeat_;
  const unsigned areaVecID = exposedAreaVec_;
  const unsigned wDistID = wallNormDist_;

  const DoubleType gravity = gravity_;
  const DoubleType z0 = z0_;
  const DoubleType Tref = Tref_;
  const DoubleType kappa = kappa_;
  const DoubleType beta_m = beta_m_;
  const DoubleType gamma_m = gamma_m_;

  const bool useShifted = useShifted_;

  const double eps = 1.0e-8;
  const DoubleType Lmax = 1.0e8;

  const stk::mesh::Selector sel =
    realm_.meta_data().locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const auto utauOps = nalu_ngp::simd_elem_field_updater(ngpMesh, ngpUtau);

  // Reducer to accumulate the area-weighted utau sum as well as total area for
  // wall boundary of this specific topology.
  nalu_ngp::ArraySimdDouble2 utauSum(0.0);
  Kokkos::Sum<nalu_ngp::ArraySimdDouble2> utauReducer(utauSum);

  const auto shp =
    shape_fcn<BcAlgTraits, QuadRank::SCV>(use_shifted_quad(useShifted));

  const std::string algName =
    "ABLWallFrictionVelAlg_" + std::to_string(BcAlgTraits::topo_);
  nalu_ngp::run_elem_par_reduce(
    algName, meshInfo, realm_.meta_data().side_rank(), faceData_, sel,
    KOKKOS_LAMBDA(ElemSimdData & edata, nalu_ngp::ArraySimdDouble2 & uSum) {
      // Unit normal vector
      NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];
      NALU_ALIGNED DoubleType velIp[BcAlgTraits::nDim_];
      NALU_ALIGNED DoubleType bcVelIp[BcAlgTraits::nDim_];

      auto& scrViews = edata.simdScrView;
      const auto& v_vel = scrViews.get_scratch_view_2D(velID);
      const auto& v_bcvel = scrViews.get_scratch_view_2D(bcVelID);
      const auto& v_rho = scrViews.get_scratch_view_1D(rhoID);
      const auto& v_bcHeatFlux = scrViews.get_scratch_view_1D(bcHeatFluxID);
      const auto& v_specHeat = scrViews.get_scratch_view_1D(specHeatID);
      const auto& v_areavec = scrViews.get_scratch_view_2D(areaVecID);
      const auto& v_wallnormdist = scrViews.get_scratch_view_1D(wDistID);

      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        DoubleType aMag = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          aMag += v_areavec(ip, d) * v_areavec(ip, d);
        }
        aMag = stk::math::sqrt(aMag);

        // unit normal and reset velocities
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          nx[d] = v_areavec(ip, d) / aMag;
          velIp[d] = 0.0;
          bcVelIp[d] = 0.0;
        }

        const DoubleType zh = v_wallnormdist(ip);

        // Compute quantities at the boundary integration points
        DoubleType heatFluxIp = 0.0;
        DoubleType rhoIp = 0.0;
        DoubleType CpIp = 0.0;
        for (int ic = 0; ic < BcAlgTraits::nodesPerElement_; ++ic) {
          const DoubleType r = shp(ip, ic);
          heatFluxIp += r * v_bcHeatFlux(ic);
          rhoIp += r * v_rho(ic);
          CpIp += r * v_specHeat(ic);

          for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            velIp[d] += r * v_vel(ic, d);
            bcVelIp[d] += r * v_bcvel(ic, d);
          }
        }

        DoubleType uTangential = 0.0;
        for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
          DoubleType uiTan = 0.0;
          DoubleType uiBcTan = 0.0;

          for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
            DoubleType ninj = nx[i] * nx[j];
            if (i == j) {
              const DoubleType om_ninj = 1.0 - ninj;
              uiTan += om_ninj * velIp[j];
              uiBcTan += om_ninj * bcVelIp[j];
            } else {
              uiTan -= ninj * velIp[j];
              uiBcTan -= ninj * bcVelIp[j];
            }
          }
          uTangential += (uiTan - uiBcTan) * (uiTan - uiBcTan);
        }
        uTangential = stk::math::sqrt(uTangential);

        const DoubleType Tflux = heatFluxIp / (rhoIp * CpIp);
        const DoubleType Lfac = stk::math::if_then_else(
          (stk::math::abs(Tflux) < eps), Lmax,
          (-Tref / (kappa * gravity * Tflux)));
        const DoubleType term = stk::math::log(zh / z0);

        DoubleType utau_calc = eps;
        for (int si = 0; si < edata.numSimdElems; ++si) {
          const double Tflux1 = stk::simd::get_data(Tflux, si);

          double utau;
          if (Tflux1 < -eps) {
            utau = calc_utau(
              stk::simd::get_data(uTangential, si), stk::simd::get_data(zh, si),
              stk::simd::get_data(term, si), stk::simd::get_data(Tflux, si),
              stk::simd::get_data(Lfac, si), stk::simd::get_data(kappa, si),
              mo::psim_stable<double>, stk::simd::get_data(beta_m, si));
          } else if (Tflux1 > eps) {
            utau = calc_utau(
              stk::simd::get_data(uTangential, si), stk::simd::get_data(zh, si),
              stk::simd::get_data(term, si), stk::simd::get_data(Tflux, si),
              stk::simd::get_data(Lfac, si), stk::simd::get_data(kappa, si),
              mo::psim_unstable<double>, stk::simd::get_data(gamma_m, si));
          } else {
            utau = stk::simd::get_data(kappa, si) *
                   stk::simd::get_data(uTangential, si) /
                   stk::simd::get_data(term, si);
          }

          stk::simd::set_data(utau_calc, si, utau);
        }
        utauOps(edata, ip) = utau_calc;

        // Accumulate utau for statistics output
        uSum.array_[0] += utau_calc * aMag;
        uSum.array_[1] += aMag;
      }
    },
    utauReducer);

  algDriver_.accumulate_utau_area_sum(utauSum.array_[0], utauSum.array_[1]);
}

INSTANTIATE_KERNEL_FACE(ABLWallFrictionVelAlg)

} // namespace nalu
} // namespace sierra
