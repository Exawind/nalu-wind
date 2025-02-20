// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestKokkosUtils.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "master_element/Hex8CVFEM.h"
#include "master_element/Quad43DCVFEM.h"
#include "master_element/CompileTimeElements.h"

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

#include <cmath>

namespace {

/** Trigonometric field functions for unit testing
 *
 * This convenience class generates trigonometric test functions for primitive
 * variables based on Steady Taylor Vortex and Steady Thermal MMS.
 */
struct TrigFieldFunction
{
  TrigFieldFunction() : pi(std::acos(-1.0)) {}

  void velocity(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];

    qField[0] = -unot * std::cos(a * pi * x) * std::sin(a * pi * y);
    qField[1] = +vnot * std::sin(a * pi * x) * std::cos(a * pi * y);
  }

  void dudx(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];

    const double a_pi = a * pi;
    const double cosx = std::cos(a_pi * x);
    const double sinx = std::sin(a_pi * x);
    const double cosy = std::cos(a_pi * y);
    const double siny = std::sin(a_pi * y);

    // du_1 / dx_j
    qField[0] = unot * a_pi * sinx * siny;
    qField[1] = -unot * a_pi * cosx * cosy;
    qField[2] = 0.0;

    // du_2 / dx_j
    qField[3] = vnot * a_pi * cosx * cosy;
    qField[4] = -vnot * a_pi * sinx * siny;
    qField[5] = 0.0;

    // z components = 0.0
    qField[6] = 0.0;
    qField[7] = 0.0;
    qField[8] = 0.0;
  }

  void pressure(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];

    qField[0] =
      -pnot / 4.0 * (std::cos(2.0 * a * pi * x) + std::cos(2.0 * a * pi * y));
  }

  void dpdx(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];

    qField[0] = 0.5 * a * pi * std::sin(2.0 * a * pi * x);
    qField[1] = 0.5 * a * pi * std::sin(2.0 * a * pi * y);
  }

  void temperature(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      (k / 4.0) * (std::cos(2.0 * aT * pi * x) + std::cos(2.0 * aT * pi * y) +
                   std::cos(2.0 * aT * pi * z));
  }

  void density(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] = rhonot * (std::cos(a * pi * x) * std::cos(a * pi * y) *
                          std::cos(a * pi * z));
  }

  void tke(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      2 * tkenot + tkenot * (std::cos(a * pi * x) * std::sin(a * pi * y) *
                             std::cos(a * pi * z));
  }

  void alpha(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      alphanot + alphanot * (std::cos(a * pi * x) * std::sin(a * pi * y) *
                             std::cos(a * pi * z));
  }

  void dkdx(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];
    const double z = coords[2];

    const double a_pi = a * pi;
    const double cosx = std::cos(a_pi * x);
    const double sinx = std::sin(a_pi * x);
    const double cosy = std::cos(a_pi * y);
    const double siny = std::sin(a_pi * y);
    const double cosz = std::cos(a_pi * z);
    const double sinz = std::sin(a_pi * z);

    qField[0] = -tkenot * a_pi * sinx * siny * cosz;
    qField[1] = tkenot * a_pi * cosx * cosy * cosz;
    qField[2] = -tkenot * a_pi * cosx * siny * sinz;
  }

  void sdr(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      2 * sdrnot + sdrnot * (std::cos(a * pi * x) * std::sin(a * pi * y) *
                             std::sin(a * pi * z));
  }

  void gamma_intermittency(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    // Range should be from 0.02 to 1.0
    qField[0] =
      gamma_intermittencynot +
      abs(std::cos(a * pi * x) * std::sin(a * pi * y) * std::sin(a * pi * z)) /
        (1.0 - gamma_intermittencynot);
  }

  void dwalldistdx(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];
    const double z = coords[2];

    const double a_pi = a * pi;
    const double cosx = std::cos(a_pi * x);
    const double sinx = std::sin(a_pi * x);
    const double cosy = std::cos(a_pi * y);
    const double siny = std::sin(a_pi * y);
    const double cosz = std::cos(a_pi * z);
    const double sinz = std::sin(a_pi * z);

    qField[0] = -dwalldistdxnot * a_pi * sinx * siny * cosz;
    qField[1] = dwalldistdxnot * a_pi * cosx * cosy * cosz;
    qField[2] = -dwalldistdxnot * a_pi * cosx * siny * sinz;
  }

  void dnDotVdx(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];
    const double z = coords[2];

    const double a_pi = a * pi;
    const double cosx = std::cos(a_pi * x);
    const double sinx = std::sin(a_pi * x);
    const double cosy = std::cos(a_pi * y);
    const double siny = std::sin(a_pi * y);
    const double cosz = std::cos(a_pi * z);
    const double sinz = std::sin(a_pi * z);

    qField[0] = -dnDotVdxnot * a_pi * sinx * siny * cosz;
    qField[1] = dnDotVdxnot * a_pi * cosx * cosy * cosz;
    qField[2] = -dnDotVdxnot * a_pi * cosx * siny * sinz;
  }

  void tdr(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      2 * tdrnot + tdrnot * (std::cos(a * pi * x) * std::sin(a * pi * y) *
                             std::sin(a * pi * z));
  }

  void dwdx(const double* coords, double* qField) const
  {
    const double x = coords[0];
    const double y = coords[1];
    const double z = coords[2];

    const double a_pi = a * pi;
    const double cosx = std::cos(a_pi * x);
    const double sinx = std::sin(a_pi * x);
    const double cosy = std::cos(a_pi * y);
    const double siny = std::sin(a_pi * y);
    const double cosz = std::cos(a_pi * z);
    const double sinz = std::sin(a_pi * z);

    qField[0] = -tkenot * a_pi * sinx * siny * sinz;
    qField[1] = tkenot * a_pi * cosx * cosy * sinz;
    qField[2] = tkenot * a_pi * cosx * siny * cosz;
  }

  void turbulent_viscosity(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      2 * tviscnot + tviscnot * (std::cos(a * pi * x) * std::cos(a * pi * y) *
                                 std::cos(a * pi * z));
  }

  void
  tensor_turbulent_viscosity(const double* /* coords */, double* qField) const
  {
    // mu_1j
    qField[0] = 0.1;
    qField[1] = 0.0;
    qField[2] = 0.0;

    // mu_1j
    qField[3] = 0.0;
    qField[4] = 0.1;
    qField[5] = 0.0;

    // mu_2j
    qField[6] = 0.0;
    qField[7] = 0.0;
    qField[8] = 0.1;
  }

  void sst_f_one_blending(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      sst_f_one_blendingnot *
      (std::sin(a * pi * x) * std::cos(a * pi * y) * std::sin(a * pi * z));
  }

  void iddes_rans_indicator(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] =
      iddes_rans_indicatornot *
      (std::sin(a * pi * x) * std::cos(a * pi * y) * std::sin(a * pi * z));
  }

  void minimum_distance_to_wall(const double* coords, double* qField) const
  {
    double x = coords[0];
    qField[0] = 10 * x + 10;
  }

  void dplus_wall_function(const double* coords, double* qField) const
  {
    double x = coords[0];
    qField[0] = 2 * x + 2;
  }

  void dhdx(const double* /*coords*/, double* qField) const
  {
    qField[0] = 30.0;
    qField[1] = 10.0;
    qField[2] = -16.0;
  }

private:
  /// Factor for u-component of velocity
  static constexpr double unot{1.0};

  /// Factor for v-component of velocity
  static constexpr double vnot{1.0};

  /// Factor for  pressure field
  static constexpr double pnot{1.0};

  /// Factor for temperature field
  static constexpr double k{1.0};

  /// Frequency multiplier for velocity and pressure fields
  static constexpr double a{0.3};

  /// Frequency for temperature fields
  static constexpr double aT{1.0};

  /// Factor for density field
  static constexpr double rhonot{1.0};

  /// Factor for tke field
  static constexpr double tkenot{1.0};

  /// Factor for adaptivity parameter field
  static constexpr double alphanot{0.5};

  /// Factor for sdr field
  static constexpr double sdrnot{1.0};

  /// Factor for gamma_intermittency field
  static constexpr double gamma_intermittencynot{0.02};

  /// Factor for dwalldistdx field
  static constexpr double dwalldistdxnot{1.0};

  /// Factor for dnDotVdx field
  static constexpr double dnDotVdxnot{1.0};

  /// Factor for tdr field
  static constexpr double tdrnot{1.0};

  /// Factor for tvisc field
  static constexpr double tviscnot{1.0};

  /// Factor for fOneBlend field
  static constexpr double sst_f_one_blendingnot{1.0};

  /// Factor for rans_indicator field
  static constexpr double iddes_rans_indicatornot{1.0};

  const double pi;
};

/** Initialize the field array with the trigonometric test function
 *
 * @param bulk Reference to STK BulkData
 * @param qField Field to be initialized. Template specialized on this field
 */
template <typename T>
void
init_trigonometric_field(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  T& qField)
{
  using FieldInitFunction =
    void (TrigFieldFunction::*)(const double*, double*) const;
  const TrigFieldFunction stv;
  const auto fieldName = qField.name();
  FieldInitFunction funcPtr = nullptr;

  if ((fieldName == "velocity") || (fieldName == "velocity_bc"))
    funcPtr = &TrigFieldFunction::velocity;
  else if ((fieldName == "dudx") || (fieldName == "average_dudx"))
    funcPtr = &TrigFieldFunction::dudx;
  else if (fieldName == "pressure")
    funcPtr = &TrigFieldFunction::pressure;
  else if (fieldName == "dpdx")
    funcPtr = &TrigFieldFunction::dpdx;
  else if (fieldName == "temperature")
    funcPtr = &TrigFieldFunction::temperature;
  else if (fieldName == "density")
    funcPtr = &TrigFieldFunction::density;
  else if (fieldName == "turbulent_ke")
    funcPtr = &TrigFieldFunction::tke;
  else if (fieldName == "k_ratio")
    funcPtr = &TrigFieldFunction::alpha;
  else if (fieldName == "dkdx")
    funcPtr = &TrigFieldFunction::dkdx;
  else if (fieldName == "specific_dissipation_rate")
    funcPtr = &TrigFieldFunction::sdr;
  else if (fieldName == "gamma_transition")
    funcPtr = &TrigFieldFunction::gamma_intermittency;
  else if (fieldName == "dwalldistdx")
    funcPtr = &TrigFieldFunction::dwalldistdx;
  else if (fieldName == "dnDotVdx")
    funcPtr = &TrigFieldFunction::dnDotVdx;
  else if (fieldName == "total_dissipation_rate")
    funcPtr = &TrigFieldFunction::tdr;
  else if (fieldName == "dwdx")
    funcPtr = &TrigFieldFunction::dwdx;
  else if (fieldName == "dhdx")
    funcPtr = &TrigFieldFunction::dhdx;
  else if (fieldName == "turbulent_viscosity")
    funcPtr = &TrigFieldFunction::turbulent_viscosity;
  else if (fieldName == "tensor_turbulent_viscosity")
    funcPtr = &TrigFieldFunction::tensor_turbulent_viscosity;
  else if (fieldName == "sst_f_one_blending")
    funcPtr = &TrigFieldFunction::sst_f_one_blending;
  else if (fieldName == "iddes_rans_indicator")
    funcPtr = &TrigFieldFunction::iddes_rans_indicator;
  else if (fieldName == "minimum_distance_to_wall")
    funcPtr = &TrigFieldFunction::minimum_distance_to_wall;
  else if (fieldName == "dplus_wall_function")
    funcPtr = &TrigFieldFunction::dplus_wall_function;
  else
    funcPtr = nullptr;

  EXPECT_TRUE(funcPtr != nullptr);

  const auto& meta = bulk.mesh_meta_data();
  EXPECT_EQ(meta.spatial_dimension(), 3u);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);

  for (const stk::mesh::Bucket* bptr : buckets) {
    for (stk::mesh::Entity node : *bptr) {
      const double* coords = stk::mesh::field_data(coordinates, node);
      double* qNode = stk::mesh::field_data(qField, node);

      ((stv).*(funcPtr))(coords, qNode);
    }
  }
}

template <typename LOOP_BODY>
void
init_trigonometric_field(
  const stk::mesh::BulkData& bulk, const LOOP_BODY& inner_loop_body)
{
  const auto& meta = bulk.mesh_meta_data();
  EXPECT_EQ(meta.spatial_dimension(), 3u);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);

  for (const stk::mesh::Bucket* bptr : buckets) {
    for (stk::mesh::Entity node : *bptr) {
      inner_loop_body(node);
    }
  }
}

} // anonymous namespace

namespace unit_test_kernel_utils {

void
velocity_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& velocity)
{
  // Add additional test functions in future?
  init_trigonometric_field(bulk, coordinates, velocity);
}

void
dudx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::TensorFieldType& dudx)
{
  // Add additional test functions in future?
  init_trigonometric_field(bulk, coordinates, dudx);
}

void
pressure_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& pressure)
{
  init_trigonometric_field(bulk, coordinates, pressure);
}

void
dpdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dpdx)
{
  init_trigonometric_field(bulk, coordinates, dpdx);
}

void
temperature_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& temperature)
{
  init_trigonometric_field(bulk, coordinates, temperature);
}

void
density_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& density)
{
  init_trigonometric_field(bulk, coordinates, density);
}

void
tke_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& tke)
{
  init_trigonometric_field(bulk, coordinates, tke);
}

void
alpha_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& alpha)
{
  init_trigonometric_field(bulk, coordinates, alpha);
}

void
dkdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dkdx)
{
  init_trigonometric_field(bulk, coordinates, dkdx);
}

void
sdr_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& sdr)
{
  init_trigonometric_field(bulk, coordinates, sdr);
}

void
gamma_intermittency_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& gamma_intermittency)
{
  init_trigonometric_field(bulk, coordinates, gamma_intermittency);
}

void
dwalldistdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dwalldistdx)
{
  init_trigonometric_field(bulk, coordinates, dwalldistdx);
}

void
dnDotVdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dnDotVdx)
{
  init_trigonometric_field(bulk, coordinates, dnDotVdx);
}

void
tdr_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& tdr)
{
  init_trigonometric_field(bulk, coordinates, tdr);
}

void
dwdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dwdx)
{
  init_trigonometric_field(bulk, coordinates, dwdx);
}

void
turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& turbulent_viscosity)
{
  init_trigonometric_field(bulk, coordinates, turbulent_viscosity);
}

void
tensor_turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::GenericFieldType& mutij)
{
  init_trigonometric_field(bulk, coordinates, mutij);
}

void
sst_f_one_blending_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& sst_f_one_blending)
{
  init_trigonometric_field(bulk, coordinates, sst_f_one_blending);
}

void
iddes_rans_indicator_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& iddes_rans_indicator)
{
  init_trigonometric_field(bulk, coordinates, iddes_rans_indicator);
}

void
minimum_distance_to_wall_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& minimum_distance_to_wall)
{
  init_trigonometric_field(bulk, coordinates, minimum_distance_to_wall);
}

void
dplus_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& dplus)
{
  init_trigonometric_field(bulk, coordinates, dplus);
}

void
property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::ScalarFieldType& mixFraction,
  sierra::nalu::ScalarFieldType& property,
  const double primary,
  const double secondary)
{
  init_trigonometric_field(bulk, [&](stk::mesh::Entity node) {
    const double mixFrac = *stk::mesh::field_data(mixFraction, node);
    double* theProp = stk::mesh::field_data(property, node);
    *theProp = primary * mixFrac + secondary * (1.0 - mixFrac);
  });
}

void
inverse_property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::ScalarFieldType& mixFraction,
  sierra::nalu::ScalarFieldType& property,
  const double primary,
  const double secondary)
{
  init_trigonometric_field(bulk, [&](stk::mesh::Entity node) {
    const double z = *stk::mesh::field_data(mixFraction, node);
    double* theProp = stk::mesh::field_data(property, node);
    *theProp = 1.0 / (z / primary + (1.0 - z) / secondary);
  });
}

void
mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  const sierra::nalu::ScalarFieldType& mixtureFrac,
  const double znot,
  const double amf)
{
  const double pi = acos(-1.0);
  init_trigonometric_field(bulk, [&](stk::mesh::Entity node) {
    const double* coords = stk::mesh::field_data(coordinates, node);
    double* mixFrac = stk::mesh::field_data(mixtureFrac, node);
    const double x = coords[0];
    const double y = coords[1];
    const double z = coords[2];
    *mixFrac = znot * cos(amf * pi * x) * cos(amf * pi * y) * cos(amf * pi * z);
  });
}

void
dhdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dhdx)
{
  init_trigonometric_field(bulk, coordinates, dhdx);
}

void
calc_mass_flow_rate(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& velocity,
  const sierra::nalu::ScalarFieldType& density,
  const sierra::nalu::VectorFieldType& edgeAreaVec,
  sierra::nalu::ScalarFieldType& massFlowRate)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  const sierra::nalu::ScalarFieldType& densityNp1 =
    density.field_of_state(stk::mesh::StateNP1);
  const sierra::nalu::VectorFieldType& velocityNp1 =
    velocity.field_of_state(stk::mesh::StateNP1);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::EDGE_RANK, selector);

  for (auto b : buckets) {
    const auto bktlen = b->size();
    const double* av = stk::mesh::field_data(edgeAreaVec, *b);
    double* mdot = stk::mesh::field_data(massFlowRate, *b);

    for (size_t ie = 0; ie < bktlen; ++ie) {
      const auto* edge_nodes = b->begin_nodes(ie);
      const auto nodeL = edge_nodes[0];
      const auto nodeR = edge_nodes[1];

      const double* velL = stk::mesh::field_data(velocityNp1, nodeL);
      const double* velR = stk::mesh::field_data(velocityNp1, nodeR);

      const double rhoL = *stk::mesh::field_data(densityNp1, nodeL);
      const double rhoR = *stk::mesh::field_data(densityNp1, nodeR);

      double tmdot = 0.0;
      for (int d = 0; d < ndim; ++d)
        tmdot += 0.5 * (rhoL * velL[d] + rhoR * velR[d]) * av[ie * ndim + d];

      mdot[ie] = tmdot;
    }
  }
}

void
calc_mass_flow_rate_scs(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const sierra::nalu::VectorFieldType& coordinates,
  const sierra::nalu::ScalarFieldType& density,
  const sierra::nalu::VectorFieldType& velocity,
  const sierra::nalu::GenericFieldType& massFlowRate)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using Hex8Traits = sierra::nalu::AlgTraitsHex8;
  using ElemSimdData = sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  const unsigned npe = Hex8Traits::nodesPerElement_;
  EXPECT_EQ(ndim, 3);
  EXPECT_EQ(topo.num_nodes(), npe);

  // Register necessary data for element gathers
  sierra::nalu::ElemDataRequests dataReq(meta);
  auto meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      Hex8Traits::topo_);
  dataReq.add_cvfem_surface_me(meSCS);

  dataReq.add_coordinates_field(
    coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(velocity, ndim);
  dataReq.add_gathered_nodal_field(density, 1);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  const stk::mesh::Selector sel =
    meta.locally_owned_part() | meta.globally_shared_part();

  const auto velID = velocity.mesh_meta_data_ordinal();
  const auto rhoID = density.mesh_meta_data_ordinal();
  const auto mdotID = massFlowRate.mesh_meta_data_ordinal();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpMdot = fieldMgr.get_field<double>(mdotID);
  const auto mdotOps =
    sierra::nalu::nalu_ngp::simd_elem_field_updater(ngpMesh, ngpMdot);

  auto v_shape_fcn = sierra::nalu::shape_fcn<
    sierra::nalu::AlgTraitsHex8, sierra::nalu::QuadRank::SCS>(
    sierra::nalu::use_shifted_quad(false));

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_calc_mdot_scs", meshInfo, stk::topology::ELEM_RANK, dataReq, sel,
    KOKKOS_LAMBDA(ElemSimdData & edata) {
      NALU_ALIGNED Traits::DblType rhoU[Hex8Traits::nDim_];

      auto& scrViews = edata.simdScrView;
      auto& v_rho = scrViews.get_scratch_view_1D(rhoID);
      auto& v_vel = scrViews.get_scratch_view_2D(velID);
      auto& meViews = scrViews.get_me_views(sierra::nalu::CURRENT_COORDINATES);
      auto& v_area = meViews.scs_areav;

      for (int ip = 0; ip < Hex8Traits::numScsIp_; ++ip) {
        for (int d = 0; d < Hex8Traits::nDim_; ++d)
          rhoU[d] = 0.0;

        for (int ic = 0; ic < Hex8Traits::nodesPerElement_; ++ic) {
          const auto r = v_shape_fcn(ip, ic);
          for (int d = 0; d < Hex8Traits::nDim_; ++d)
            rhoU[d] += r * v_rho(ic) * v_vel(ic, d);
        }

        Traits::DblType tmdot = 0.0;
        for (int d = 0; d < Hex8Traits::nDim_; ++d)
          tmdot += rhoU[d] * v_area(ip, d);

        // Scatter to all elements in this SIMD group
        mdotOps(edata, ip) = tmdot;
      }
    });

  ngpMdot.modify_on_device();
  ngpMdot.sync_to_host();
}

void
calc_open_mass_flow_rate(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const sierra::nalu::VectorFieldType& coordinates,
  const sierra::nalu::ScalarFieldType& density,
  const sierra::nalu::VectorFieldType& velocity,
  const sierra::nalu::GenericFieldType& exposedAreaVec,
  const sierra::nalu::GenericFieldType& massFlowRate)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using Quad4Traits = sierra::nalu::AlgTraitsQuad4;
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  const unsigned npe = Quad4Traits::nodesPerFace_;
  EXPECT_EQ(ndim, 3);
  EXPECT_EQ(topo.num_nodes(), npe);

  // Register necessary element data gathers
  sierra::nalu::ElemDataRequests dataReq(meta);
  auto meFC =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      Quad4Traits::topo_);
  dataReq.add_cvfem_surface_me(meFC);
  dataReq.add_coordinates_field(
    coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(velocity, ndim);
  dataReq.add_gathered_nodal_field(density, 1);
  dataReq.add_face_field(
    exposedAreaVec, Quad4Traits::numFaceIp_, Quad4Traits::nDim_);
  dataReq.add_face_field(massFlowRate, Quad4Traits::numFaceIp_);

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  const stk::mesh::Selector sel =
    meta.locally_owned_part() | meta.globally_shared_part();

  const auto velID = velocity.mesh_meta_data_ordinal();
  const auto rhoID = density.mesh_meta_data_ordinal();
  const auto mdotID = massFlowRate.mesh_meta_data_ordinal();
  const auto areaID = exposedAreaVec.mesh_meta_data_ordinal();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpMdot = fieldMgr.get_field<double>(mdotID);
  const auto mdotOps =
    sierra::nalu::nalu_ngp::simd_elem_field_updater(ngpMesh, ngpMdot);

  auto shape_fcn = sierra::nalu::shape_fcn<
    sierra::nalu::AlgTraitsHex8, sierra::nalu::QuadRank::SCS>(
    sierra::nalu::use_shifted_quad(false));

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_calc_open_mdot", meshInfo, meta.side_rank(), dataReq, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      NALU_ALIGNED Traits::DblType rhoU[Quad4Traits::nDim_];

      auto& scrViews = edata.simdScrView;
      const auto& v_rho = scrViews.get_scratch_view_1D(rhoID);
      const auto& v_vel = scrViews.get_scratch_view_2D(velID);
      const auto& v_area = scrViews.get_scratch_view_2D(areaID);
      const auto& meViews =
        scrViews.get_me_views(sierra::nalu::CURRENT_COORDINATES);

      for (int ip = 0; ip < Quad4Traits::numFaceIp_; ++ip) {
        for (int d = 0; d < Quad4Traits::nDim_; ++d)
          rhoU[d] = 0.0;

        for (int ic = 0; ic < Quad4Traits::nodesPerFace_; ++ic) {
          const Traits::DblType r = shape_fcn(ip, ic);
          for (int d = 0; d < Quad4Traits::nDim_; ++d)
            rhoU[d] += r * v_rho(ic) * v_vel(ic, d);
        }

        Traits::DblType tmdot = 0.0;
        for (int d = 0; d < Quad4Traits::nDim_; ++d)
          tmdot += rhoU[d] * v_area(ip, d);

        mdotOps(edata, ip) = tmdot;
      }
    });

  ngpMdot.modify_on_device();
  ngpMdot.sync_to_host();
}

void
calc_edge_area_vec(
  const stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const sierra::nalu::VectorFieldType& coordinates,
  const sierra::nalu::VectorFieldType& edgeAreaVec)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  auto meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(topo);
  const auto npe = meSCS->nodesPerElement_;
  const auto numScsIp = meSCS->num_integration_points();
  const int* lrscv = meSCS->adjacentNodes();
  const int* scsIpEdge = meSCS->scsIpEdgeOrd();

  // Scratch arrays
  std::vector<double> w_coords(ndim * npe);
  std::vector<double> w_scs_areav(ndim * numScsIp);
  sierra::nalu::SharedMemView<double**> coords(w_coords.data(), npe, ndim);
  sierra::nalu::SharedMemView<double**> scs_areav(
    w_scs_areav.data(), numScsIp, ndim);

  // Reset edge area vector to zero
  stk::mesh::field_fill(0.0, edgeAreaVec);

  const stk::mesh::Selector sel =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEMENT_RANK, sel);

  for (auto b : buckets) {
    STK_ThrowRequire(b->topology() == topo);

    const auto bktlen = b->size();
    for (size_t ie = 0; ie < bktlen; ++ie) {
      const auto* elem_nodes = b->begin_nodes(ie);
      const auto num_nodes = b->num_nodes(ie);

      for (size_t in = 0; in < num_nodes; ++in) {
        const auto node = elem_nodes[in];
        const double* coord = stk::mesh::field_data(coordinates, node);
        for (int d = 0; d < ndim; ++d) {
          coords(in, d) = coord[d];
        }
      }

      meSCS->determinant(coords, scs_areav);

      const auto* elem_edges = b->begin_edges(ie);

      for (int ip = 0; ip < numScsIp; ++ip) {
        const int iedge = scsIpEdge[ip];
        const auto edge = elem_edges[iedge];

        double* av = stk::mesh::field_data(edgeAreaVec, edge);
        const auto* edge_nodes = bulk.begin_nodes(edge);
        // Index of "left" node in the element relations
        const int iLn = lrscv[2 * ip];

        // STK identifier for the left node according to the element and edge
        const auto lnElemId = bulk.identifier(elem_nodes[iLn]);
        const auto lnEdgeId = bulk.identifier(edge_nodes[0]);

        // If the left node on both edge and element are same, then they are
        // oriented the same way, i.e., sign multiplier is just 1.0, otherwise
        // reverse sign
        const double sgn = (lnElemId == lnEdgeId) ? 1.0 : -1.0;

        // accumulate contribution from this subcontrol surface to edge area
        // vector
        for (int d = 0; d < ndim; ++d) {
          av[d] += w_scs_areav[ip * ndim + d] * sgn;
        }
      }
    }
  }
}

void
calc_exposed_area_vec(
  const stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::GenericFieldType& exposedAreaVec)
{
  using Quad4Traits = sierra::nalu::AlgTraitsQuad4;
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  const unsigned npe = Quad4Traits::nodesPerFace_;
  EXPECT_EQ(ndim, 3);
  EXPECT_EQ(topo.num_nodes(), npe);

  // Register necessary element data gathers
  sierra::nalu::ElemDataRequests dataReq(meta);
  auto meFC =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      Quad4Traits::topo_);
  dataReq.add_cvfem_surface_me(meFC);
  dataReq.add_coordinates_field(
    coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_face_field(
    exposedAreaVec, Quad4Traits::numFaceIp_, Quad4Traits::nDim_);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  const stk::mesh::Selector sel =
    meta.locally_owned_part() | meta.globally_shared_part();

  const auto areaID = exposedAreaVec.mesh_meta_data_ordinal();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto areaVec = fieldMgr.get_field<double>(areaID);
  const auto areaVecOps =
    sierra::nalu::nalu_ngp::simd_elem_field_updater(ngpMesh, areaVec);

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_calc_exposed_area_vec", meshInfo, meta.side_rank(), dataReq, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      auto& scrViews = edata.simdScrView;
      const auto& meViews =
        scrViews.get_me_views(sierra::nalu::CURRENT_COORDINATES);
      const auto& v_area = meViews.scs_areav;

      for (int ip = 0; ip < Quad4Traits::numFaceIp_; ++ip)
        for (int d = 0; d < Quad4Traits::nDim_; ++d)
          areaVecOps(edata, ip * Quad4Traits::nDim_ + d) = v_area(ip, d);
    });

  areaVec.modify_on_device();
  areaVec.sync_to_host();
}

void
expect_all_near(
  const Kokkos::View<double*>& devCalcValue,
  const double* exactValue,
  const double tol)
{
  const int length = devCalcValue.extent(0);

  auto hostCalcValue =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devCalcValue);

  for (int i = 0; i < length; ++i) {
    EXPECT_NEAR(hostCalcValue[i], exactValue[i], tol);
  }
}

void
expect_all_near(
  const Kokkos::View<double*>& devCalcValue,
  const double exactValue,
  const double tol)
{
  const int length = devCalcValue.extent(0);

  auto hostCalcValue =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devCalcValue);

  for (int i = 0; i < length; ++i) {
    EXPECT_NEAR(hostCalcValue[i], exactValue, tol);
  }
}

void
expect_all_near_2d(
  const Kokkos::View<double**>& devCalcValue,
  const double* exactValue,
  const double tol)
{
  const int dim1 = devCalcValue.extent(0);
  const int dim2 = devCalcValue.extent(1);

  auto hostCalcValue =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devCalcValue);

  for (int i = 0; i < dim1; i++)
    for (int j = 0; j < dim2; j++)
      EXPECT_NEAR(hostCalcValue(i, j), exactValue[i * dim2 + j], tol);
}

} // namespace unit_test_kernel_utils
