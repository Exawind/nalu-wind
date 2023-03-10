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

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <ngp_utils/NgpFieldManager.h>

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

  /// Factor for tdr field
  static constexpr double tdrnot{1.0};

  /// Factor for tvisc field
  static constexpr double tviscnot{1.0};

  /// Factor for fOneBlend field
  static constexpr double sst_f_one_blendingnot{1.0};

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
  const VectorFieldType& coordinates,
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
  const VectorFieldType& coordinates,
  VectorFieldType& velocity)
{
  // Add additional test functions in future?
  init_trigonometric_field(bulk, coordinates, velocity);
}

void
dudx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  GenericFieldType& dudx)
{
  // Add additional test functions in future?
  init_trigonometric_field(bulk, coordinates, dudx);
}

void
pressure_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& pressure)
{
  init_trigonometric_field(bulk, coordinates, pressure);
}

void
dpdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dpdx)
{
  init_trigonometric_field(bulk, coordinates, dpdx);
}

void
temperature_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& temperature)
{
  init_trigonometric_field(bulk, coordinates, temperature);
}

void
density_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& density)
{
  init_trigonometric_field(bulk, coordinates, density);
}

void
tke_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& tke)
{
  init_trigonometric_field(bulk, coordinates, tke);
}

void
alpha_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& alpha)
{
  init_trigonometric_field(bulk, coordinates, alpha);
}

void
dkdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dkdx)
{
  init_trigonometric_field(bulk, coordinates, dkdx);
}

void
sdr_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& sdr)
{
  init_trigonometric_field(bulk, coordinates, sdr);
}

void
tdr_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& tdr)
{
  init_trigonometric_field(bulk, coordinates, tdr);
}

void
dwdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dwdx)
{
  init_trigonometric_field(bulk, coordinates, dwdx);
}

void
turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& turbulent_viscosity)
{
  init_trigonometric_field(bulk, coordinates, turbulent_viscosity);
}

void
tensor_turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  GenericFieldType& mutij)
{
  init_trigonometric_field(bulk, coordinates, mutij);
}
void
sst_f_one_blending_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& sst_f_one_blending)
{
  init_trigonometric_field(bulk, coordinates, sst_f_one_blending);
}

void
minimum_distance_to_wall_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& minimum_distance_to_wall)
{
  init_trigonometric_field(bulk, coordinates, minimum_distance_to_wall);
}

void
dplus_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& dplus)
{
  init_trigonometric_field(bulk, coordinates, dplus);
}

void
property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const ScalarFieldType& mixFraction,
  ScalarFieldType& property,
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
  const ScalarFieldType& mixFraction,
  ScalarFieldType& property,
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
  const VectorFieldType& coordinates,
  const ScalarFieldType& mixtureFrac,
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
  const VectorFieldType& coordinates,
  VectorFieldType& dhdx)
{
  init_trigonometric_field(bulk, coordinates, dhdx);
}

void
calc_mass_flow_rate(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& velocity,
  const ScalarFieldType& density,
  const VectorFieldType& edgeAreaVec,
  ScalarFieldType& massFlowRate)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  const ScalarFieldType& densityNp1 =
    density.field_of_state(stk::mesh::StateNP1);
  const VectorFieldType& velocityNp1 =
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
  const VectorFieldType& coordinates,
  const ScalarFieldType& density,
  const VectorFieldType& velocity,
  const GenericFieldType& massFlowRate)
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
  dataReq.add_master_element_call(
    sierra::nalu::SCS_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);

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

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_calc_mdot_scs", meshInfo, stk::topology::ELEM_RANK, dataReq, sel,
    KOKKOS_LAMBDA(ElemSimdData & edata) {
      NALU_ALIGNED Traits::DblType rhoU[Hex8Traits::nDim_];

      auto& scrViews = edata.simdScrView;
      auto& v_rho = scrViews.get_scratch_view_1D(rhoID);
      auto& v_vel = scrViews.get_scratch_view_2D(velID);
      auto& meViews = scrViews.get_me_views(sierra::nalu::CURRENT_COORDINATES);
      auto& v_area = meViews.scs_areav;
      auto& v_shape_fcn = meViews.scs_shape_fcn;

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
  const VectorFieldType& coordinates,
  const ScalarFieldType& density,
  const VectorFieldType& velocity,
  const GenericFieldType& exposedAreaVec,
  const GenericFieldType& massFlowRate)
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
  dataReq.add_master_element_call(
    sierra::nalu::SCS_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);

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
      const auto& shape_fcn = meViews.scs_shape_fcn;

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
  const VectorFieldType& coordinates,
  const VectorFieldType& edgeAreaVec)
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
    ThrowRequire(b->topology() == topo);

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
  const VectorFieldType& coordinates,
  GenericFieldType& exposedAreaVec)
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

#if !defined(KOKKOS_ENABLE_GPU)

#if 0
void calc_projected_nodal_gradient_interior(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnv,
  const ScalarFieldType& scalarField,
  const VectorFieldType& gradField)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using Hex8Traits = sierra::nalu::AlgTraitsHex8;
  using ElemSimdDataType = sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  const int npe = Hex8Traits::nodesPerElement_;
  EXPECT_EQ(ndim, 3);
  EXPECT_EQ(topo.num_nodes(), npe);

  sierra::nalu::ElemDataRequests dataReq(meta);

  auto meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(Hex8Traits::topo_);

  dataReq.add_cvfem_surface_me(meSCS);
  dataReq.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(scalarField, 1);
  dataReq.add_gathered_nodal_field(dnv, 1);
  dataReq.add_gathered_nodal_field(gradField, ndim);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector sel =
    meta.locally_owned_part() | meta.globally_shared_part();

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto gradFieldID = gradField.mesh_meta_data_ordinal();
  const auto dnvID = dnv.mesh_meta_data_ordinal();
  const auto scalarID = scalarField.mesh_meta_data_ordinal();
  auto ngpGradField = fieldMgr.get_field<double>(gradFieldID);
  const auto gradFieldOps = sierra::nalu::nalu_ngp::simd_elem_nodal_field_updater(
    ngpMesh, ngpGradField);


  sierra::nalu::nalu_ngp::run_elem_algorithm(
    meshInfo, stk::topology::ELEM_RANK, dataReq, sel,
    KOKKOS_LAMBDA(ElemSimdDataType& edata) {
      const int* lrscv = meSCS->adjacentNodes();
      auto& scrView = edata.simdScrView;
      const auto& v_dnv = scrView.get_scratch_view_1D(dnvID);
      const auto& v_scalar = scrView.get_scratch_view_1D(scalarID);
      const auto& meViews = scrView.get_me_views(sierra::nalu::CURRENT_COORDINATES);
      const auto& v_areav = meViews.scs_areav;
      const auto& v_shape_fcn = meViews.scs_shape_fcn;

      for (int ip=0; ip < Hex8Traits::numScsIp_; ++ip) {
        Traits::DblType qIp = 0.0;
        for (int n=0; n < Hex8Traits::nodesPerElement_; ++n)
          qIp += v_shape_fcn(ip, n) * v_scalar(n);

        int il = lrscv[2 * ip];
        int ir = lrscv[2 * ip + 1];

        for (int d=0; d < Hex8Traits::nDim_; ++d) {
          Traits::DblType fac = qIp * v_areav(ip, d);
          Traits::DblType valL = fac / v_dnv(il);
          Traits::DblType valR = fac / v_dnv(ir);

          gradFieldOps(edata, il, d) += valL;
          gradFieldOps(edata, ir, d) -= valR;
        }
      }
    });
}
#endif

template <typename PhiType, typename GradPhiType>
void
calc_projected_nodal_gradient_interior(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnv,
  const PhiType& phi,
  const GradPhiType& gradPhi)
{
  static_assert(
    ((std::is_same<PhiType, ScalarFieldType>::value &&
      std::is_same<GradPhiType, VectorFieldType>::value) ||
     (std::is_same<PhiType, VectorFieldType>::value &&
      std::is_same<GradPhiType, GenericFieldType>::value)),
    "Improper field types passed to nodal gradient calculator");
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;
  using Hex8Traits = sierra::nalu::AlgTraitsHex8;
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  constexpr int ncomp = std::is_same<
                          typename stk::mesh::FieldTraits<PhiType>::tag1,
                          stk::mesh::Cartesian>::value
                          ? Hex8Traits::nDim_
                          : 1;
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  const unsigned npe = Hex8Traits::nodesPerElement_;
  EXPECT_EQ(ndim, 3);
  EXPECT_EQ(topo.num_nodes(), npe);

  sierra::nalu::ElemDataRequests dataReq(meta);

  auto meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_dev(
      Hex8Traits::topo_);

  dataReq.add_cvfem_surface_me(meSCS);
  dataReq.add_coordinates_field(
    coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_gathered_nodal_field(phi, 1);
  dataReq.add_gathered_nodal_field(dnv, 1);
  dataReq.add_gathered_nodal_field(gradPhi, ndim);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);
  dataReq.add_master_element_call(
    sierra::nalu::SCS_SHAPE_FCN, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector sel =
    meta.locally_owned_part() | meta.globally_shared_part();

  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto gradPhiID = gradPhi.mesh_meta_data_ordinal();
  const auto dnvID = dnv.mesh_meta_data_ordinal();
  const auto scalarID = phi.mesh_meta_data_ordinal();
  auto ngpGradField = fieldMgr.get_field<double>(gradPhiID);
  const auto gradPhiOps = sierra::nalu::nalu_ngp::simd_elem_nodal_field_updater(
    ngpMesh, ngpGradField);

  sierra::nalu::nalu_ngp::run_elem_algorithm(
    "unittest_calc_png_interior", meshInfo, stk::topology::ELEM_RANK, dataReq,
    sel, KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      const int* lrscv = meSCS->adjacentNodes();
      auto& scrView = edata.simdScrView;
      const auto& v_dnv = scrView.get_scratch_view_1D(dnvID);
      const auto& v_scalar = scrView.get_scratch_view_1D(scalarID);
      const auto& meViews =
        scrView.get_me_views(sierra::nalu::CURRENT_COORDINATES);
      const auto& v_areav = meViews.scs_areav;
      const auto& v_shape_fcn = meViews.scs_shape_fcn;

      for (int di = 0; di < ncomp; ++di) {
        for (int ip = 0; ip < Hex8Traits::numScsIp_; ++ip) {
          Traits::DblType qIp = 0.0;
          for (int n = 0; n < Hex8Traits::nodesPerElement_; ++n)
            qIp += v_shape_fcn(ip, n) * v_scalar(n);

          int il = lrscv[2 * ip];
          int ir = lrscv[2 * ip + 1];

          for (int d = 0; d < Hex8Traits::nDim_; ++d) {
            Traits::DblType fac = qIp * v_areav(ip, d);
            Traits::DblType valL = fac / v_dnv(il);
            Traits::DblType valR = fac / v_dnv(ir);

            gradPhiOps(edata, il, di * Hex8Traits::nDim_ + d) += valL;
            gradPhiOps(edata, ir, di * Hex8Traits::nDim_ + d) -= valR;
          }
        }
      }
    });
}

template void
calc_projected_nodal_gradient_interior<ScalarFieldType, VectorFieldType>(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnv,
  const ScalarFieldType& phi,
  const VectorFieldType& gradPhi);

template void
calc_projected_nodal_gradient_interior<VectorFieldType, GenericFieldType>(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnv,
  const VectorFieldType& phi,
  const GenericFieldType& gradPhi);

#if 0
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  auto meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(topo);

  dataNeeded.add_cvfem_surface_me(meSCS);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(scalarField, 1);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  stk::mesh::NgpMesh ngpMesh(bulk);
  sierra::nalu::nalu_ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension(), sierra::nalu::ElemReqType::ELEM) ;

  auto v_shape_function = Kokkos::View<double**>("shape_function", meSCS->num_integration_points(), meSCS->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
    auto& b = *buckets[team.league_rank()];
    const auto length = b.size();

    EXPECT_EQ(b.topology(), topo);

    sierra::nalu::ScratchViews<DoubleType> preReqData(
      team, ndim, topo.num_nodes(), dataNeededNGP);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
      stk::mesh::Entity element = b[k];
      sierra::nalu::fill_pre_req_data(
        dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, element,
        preReqData);
      sierra::nalu::fill_master_element_views(dataNeededNGP, preReqData);

      meSCS->shape_fcn<>(v_shape_function);
      auto v_dnv = preReqData.get_scratch_view_1D(dnv);
      auto v_scalar = preReqData.get_scratch_view_1D(scalarField);
      auto v_scs_areav = preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scs_areav;
      const stk::mesh::NgpMesh::ConnectedNodes node_rels = preReqData.elemNodes;
      const int* lrscv = meSCS->adjacentNodes();

      for (int ip = 0; ip < meSCS->num_integration_points(); ++ip) {
        double qIp = 0.0;
        for (int n = 0; n < meSCS->nodesPerElement_; ++n) {
          qIp += v_shape_function(ip, n) * v_scalar(n);
        }

        int il = lrscv[2*ip + 0];
        int ir = lrscv[2*ip + 1];
        double* dqdxL = stk::mesh::field_data(gradField, node_rels[il]);
        double* dqdxR = stk::mesh::field_data(gradField, node_rels[ir]);

        for (int d = 0; d < ndim; ++d) {
          double fac = qIp * v_scs_areav(ip, d);
          double valL = fac / v_dnv(il);
          double valR = fac / v_dnv(ir);
          Kokkos::atomic_add(dqdxL + d, +valL);
          Kokkos::atomic_add(dqdxR + d, -valR);
        }
      }
    });
  });
}

void calc_projected_nodal_gradient_interior(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnv,
  const VectorFieldType& vectorField,
  const GenericFieldType& gradField)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  auto meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(topo);

  dataNeeded.add_cvfem_surface_me(meSCS);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(vectorField, ndim);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  stk::mesh::NgpMesh ngpMesh(bulk);
  sierra::nalu::nalu_ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension(), sierra::nalu::ElemReqType::ELEM) ;

  auto v_shape_function = Kokkos::View<double**>("shape_function", meSCS->num_integration_points(), meSCS->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
    auto& b = *buckets[team.league_rank()];
    const auto length = b.size();

    EXPECT_EQ(b.topology(), topo);

    sierra::nalu::ScratchViews<DoubleType> preReqData(
      team, ndim, topo.num_nodes(), dataNeededNGP);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
      stk::mesh::Entity element = b[k];
      sierra::nalu::fill_pre_req_data(
        dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, element,
        preReqData);
      sierra::nalu::fill_master_element_views(dataNeededNGP, preReqData);

      meSCS->shape_fcn<>(v_shape_function);
      auto v_dnv = preReqData.get_scratch_view_1D(dnv);
      auto v_vector = preReqData.get_scratch_view_2D(vectorField);
      auto v_scs_areav = preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scs_areav;
      const stk::mesh::NgpMesh::ConnectedNodes node_rels = preReqData.elemNodes;
      const int* lrscv = meSCS->adjacentNodes();

      for (int di = 0; di < ndim; ++di) {
        for (int ip = 0; ip < meSCS->num_integration_points(); ++ip) {
          double qIp = 0.0;
          for (int n = 0; n < meSCS->nodesPerElement_; ++n) {
            qIp += v_shape_function(ip, n) * v_vector(n,di);
          }

          int il = lrscv[2*ip + 0];
          int ir = lrscv[2*ip + 1];
          double* dqdxL = stk::mesh::field_data(gradField, node_rels[il]);
          double* dqdxR = stk::mesh::field_data(gradField, node_rels[ir]);

          for (int d = 0; d < ndim; ++d) {
            double fac = qIp * v_scs_areav(ip, d);
            double valL = fac / v_dnv(il);
            double valR = fac / v_dnv(ir);
            Kokkos::atomic_add(dqdxL + di*ndim + d, +valL);
            Kokkos::atomic_add(dqdxR + di*ndim + d, -valR);
          }
        }
      }
    });
  });
}
#endif

void
calc_projected_nodal_gradient_boundary(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnv,
  const ScalarFieldType& scalarField,
  const VectorFieldType& gradField)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);
  EXPECT_EQ(topo.rank(), meta.side_rank());

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  auto meBC =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(topo);

  dataNeeded.add_cvfem_surface_me(meBC);
  dataNeeded.add_coordinates_field(
    coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(scalarField, 1);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(
    sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(meta.side_rank(), selector);

  stk::mesh::NgpMesh ngpMesh(bulk);
  sierra::nalu::nalu_ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(
    fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(
    dataNeededNGP, meta.spatial_dimension(), sierra::nalu::ElemReqType::ELEM);

  auto v_shape_function = Kokkos::View<DoubleType**>(
    "shape_function", meBC->num_integration_points(), meBC->nodesPerElement_);

  sierra::nalu::SharedMemView<
    sierra::nalu::DoubleType**, sierra::nalu::DeviceShmem>
    shape_function(
      v_shape_function.data(), meBC->num_integration_points(),
      meBC->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, [&](const sierra::nalu::TeamHandleType& team) {
      auto& b = *buckets[team.league_rank()];
      const auto length = b.size();

      EXPECT_EQ(b.topology(), topo);

      sierra::nalu::ScratchViews<DoubleType> preReqData(
        team, ndim, topo.num_nodes(), dataNeededNGP);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
          stk::mesh::Entity face = b[k];
          sierra::nalu::fill_pre_req_data(
            dataNeededNGP, ngpMesh, meta.side_rank(), face, preReqData);
          sierra::nalu::fill_master_element_views(dataNeededNGP, preReqData);

          meBC->shape_fcn<>(shape_function);
          auto v_dnv = preReqData.get_scratch_view_1D(dnv);
          auto v_scalar = preReqData.get_scratch_view_1D(scalarField);
          auto v_scs_areav =
            preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES)
              .scs_areav;
          const stk::mesh::NgpMesh::ConnectedNodes node_rels =
            preReqData.elemNodes;
          const int* ipNodeMap = meBC->ipNodeMap();

          for (int ip = 0; ip < meBC->num_integration_points(); ++ip) {
            DoubleType qIp = 0.0;
            for (int n = 0; n < meBC->nodesPerElement_; ++n) {
              qIp += v_shape_function(ip, n) * v_scalar(n);
            }

            const int nn = ipNodeMap[ip];
            double* dqdxNN = stk::mesh::field_data(gradField, node_rels[nn]);

            for (int d = 0; d < ndim; ++d) {
              const double fac =
                stk::simd::get_data(qIp * v_scs_areav(ip, d) / v_dnv(nn), 0);
              Kokkos::atomic_add(dqdxNN + d, fac);
            }
          }
        });
    });
}

void
calc_projected_nodal_gradient_boundary(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnv,
  const VectorFieldType& vectorField,
  const GenericFieldType& gradField)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);
  EXPECT_EQ(topo.rank(), meta.side_rank());

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  auto meBC =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(topo);

  dataNeeded.add_cvfem_surface_me(meBC);
  dataNeeded.add_coordinates_field(
    coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(vectorField, ndim);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(
    sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(meta.side_rank(), selector);

  stk::mesh::NgpMesh ngpMesh(bulk);
  sierra::nalu::nalu_ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(
    fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread =
    sierra::nalu::get_num_bytes_pre_req_data<DoubleType>(
      dataNeededNGP, meta.spatial_dimension(), sierra::nalu::ElemReqType::ELEM);

  auto v_shape_function = Kokkos::View<DoubleType**>(
    "shape_function", meBC->num_integration_points(), meBC->nodesPerElement_);
  sierra::nalu::SharedMemView<
    sierra::nalu::DoubleType**, sierra::nalu::DeviceShmem>
    shape_function(
      v_shape_function.data(), meBC->num_integration_points(),
      meBC->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, [&](const sierra::nalu::TeamHandleType& team) {
      auto& b = *buckets[team.league_rank()];
      const auto length = b.size();

      EXPECT_EQ(b.topology(), topo);

      sierra::nalu::ScratchViews<DoubleType> preReqData(
        team, ndim, topo.num_nodes(), dataNeededNGP);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
          stk::mesh::Entity face = b[k];
          sierra::nalu::fill_pre_req_data(
            dataNeededNGP, ngpMesh, meta.side_rank(), face, preReqData);
          sierra::nalu::fill_master_element_views(dataNeededNGP, preReqData);

          meBC->shape_fcn<>(shape_function);
          auto v_dnv = preReqData.get_scratch_view_1D(dnv);
          auto v_vector = preReqData.get_scratch_view_2D(vectorField);
          auto v_scs_areav =
            preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES)
              .scs_areav;
          const stk::mesh::NgpMesh::ConnectedNodes node_rels =
            preReqData.elemNodes;
          const int* ipNodeMap = meBC->ipNodeMap();

          for (int di = 0; di < ndim; ++di) {
            for (int ip = 0; ip < meBC->num_integration_points(); ++ip) {
              DoubleType qIp = 0.0;
              for (int n = 0; n < meBC->nodesPerElement_; ++n) {
                qIp += v_shape_function(ip, n) * v_vector(n, di);
              }

              const int nn = ipNodeMap[ip];
              double* dqdxNN = stk::mesh::field_data(gradField, node_rels[nn]);

              for (int d = 0; d < ndim; ++d) {
                const double fac =
                  stk::simd::get_data(qIp * v_scs_areav(ip, d) / v_dnv(nn), 0);
                Kokkos::atomic_add(dqdxNN + di * ndim + d, fac);
              }
            }
          }
        });
    });
}

void
calc_dual_nodal_volume(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnvField)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  auto meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element_on_host(topo);

  dataNeeded.add_cvfem_volume_me(meSCV);
  dataNeeded.add_coordinates_field(
    coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_master_element_call(
    sierra::nalu::SCV_VOLUME, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  stk::mesh::NgpMesh ngpMesh(bulk);
  sierra::nalu::nalu_ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(
    fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread =
    sierra::nalu::get_num_bytes_pre_req_data<DoubleType>(
      dataNeededNGP, meta.spatial_dimension(), sierra::nalu::ElemReqType::ELEM);

  auto v_shape_function = Kokkos::View<DoubleType**>(
    "shape_function", meSCV->num_integration_points(), meSCV->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, [&](const sierra::nalu::TeamHandleType& team) {
      auto& b = *buckets[team.league_rank()];
      const auto length = b.size();

      EXPECT_EQ(b.topology(), topo);

      sierra::nalu::ScratchViews<DoubleType> preReqData(
        team, ndim, topo.num_nodes(), dataNeededNGP);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
          stk::mesh::Entity element = b[k];
          sierra::nalu::fill_pre_req_data(
            dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, element,
            preReqData);
          sierra::nalu::fill_master_element_views(dataNeededNGP, preReqData);

          auto v_scv_vol =
            preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES)
              .scv_volume;
          const stk::mesh::NgpMesh::ConnectedNodes node_rels =
            preReqData.elemNodes;
          const int* ipNodeMap = meSCV->ipNodeMap();

          for (int ip = 0; ip < meSCV->num_integration_points(); ++ip) {
            DoubleType volIp = v_scv_vol(ip);
            Kokkos::atomic_add(
              stk::mesh::field_data(dnvField, node_rels[ipNodeMap[ip]]),
              stk::simd::get_data(volIp, 0));
          }
        });
    });
}

void
calc_projected_nodal_gradient(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  ScalarFieldType& dnv,
  const ScalarFieldType& scalarField,
  VectorFieldType& gradField)
{
  // for now
  EXPECT_TRUE(
    topo != stk::topology::PYRAMID_5 && topo != stk::topology::WEDGE_6);
  stk::mesh::field_fill(0.0, dnv);
  stk::mesh::field_fill(0.0, gradField);

  calc_dual_nodal_volume(bulk, topo, coordinates, dnv);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&dnv});
  }

  calc_projected_nodal_gradient_interior(
    bulk, topo, coordinates, dnv, scalarField, gradField);
  calc_projected_nodal_gradient_boundary(
    bulk, topo.side_topology(0), coordinates, dnv, scalarField, gradField);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&gradField});
  }
}

void
calc_projected_nodal_gradient(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  ScalarFieldType& dnv,
  const VectorFieldType& vectorField,
  GenericFieldType& gradField)
{
  // for now
  EXPECT_TRUE(
    topo != stk::topology::PYRAMID_5 && topo != stk::topology::WEDGE_6);
  stk::mesh::field_fill(0.0, dnv);

  calc_dual_nodal_volume(bulk, topo, coordinates, dnv);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&dnv});
  }

  stk::mesh::field_fill(0.0, gradField);
  calc_projected_nodal_gradient_interior(
    bulk, topo, coordinates, dnv, vectorField, gradField);
  calc_projected_nodal_gradient_boundary(
    bulk, topo.side_topology(0), coordinates, dnv, vectorField, gradField);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&gradField});
  }
}

#endif // KOKKOS_ENABLE_CUDA

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
