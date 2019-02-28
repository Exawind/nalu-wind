/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestKokkosUtils.h"

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_ngp/NgpFieldManager.hpp>

#include <cmath>

namespace {

/** Trigonometric field functions for unit testing
 *
 * This convenience class generates trigonometric test functions for primitive
 * variables based on Steady Taylor Vortex and Steady Thermal MMS.
 */
struct TrigFieldFunction
{
  TrigFieldFunction()
    : pi(std::acos(-1.0))
  {}

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

    qField[0] = -pnot/4.0 * (
      std::cos(2.0 * a * pi * x) + std::cos(2.0 * a * pi * y));
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

    qField[0] = (k/4.0) * (
      std::cos(2.0 * aT * pi * x) +
      std::cos(2.0 * aT * pi * y) +
      std::cos(2.0 * aT * pi * z));
  }

  void density(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] = rhonot * (
      std::cos(a * pi * x) *
      std::cos(a * pi * y) *
      std::cos(a * pi * z));
  }

  void tke(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] = 2*tkenot + tkenot * (
      std::cos(a * pi * x) *
      std::sin(a * pi * y) *
      std::cos(a * pi * z));
  }

  void alpha(const double* coords, double* qField) const
  {
    double x = coords[0];
    double y = coords[1];
    double z = coords[2];

    qField[0] = alphanot + alphanot * (
      std::cos(a * pi * x) *
      std::sin(a * pi * y) *
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

    qField[0] = 2*sdrnot + sdrnot * (
      std::cos(a * pi * x) *
      std::sin(a * pi * y) *
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

    qField[0] = 2*tviscnot + tviscnot * (
      std::cos(a * pi * x) *
      std::cos(a * pi * y) *
      std::cos(a * pi * z));
  }

  void tensor_turbulent_viscosity(const double*  /* coords */, double* qField) const
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

    qField[0] = sst_f_one_blendingnot * (
      std::sin(a * pi * x) *
      std::cos(a * pi * y) *
      std::sin(a * pi * z));
  }

  void minimum_distance_to_wall(const double* coords, double* qField) const
  {
    double x = coords[0];
    qField[0] = 10*x+10;
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
  static constexpr double alphanot{1.0};

  /// Factor for sdr field
  static constexpr double sdrnot{1.0};

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
template<typename T>
void init_trigonometric_field(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  T& qField)
{
  using FieldInitFunction = void (TrigFieldFunction::*)(const double*, double*) const;
  const TrigFieldFunction stv;
  const auto fieldName = qField.name();
  FieldInitFunction funcPtr = nullptr;

  if (fieldName == "velocity")
    funcPtr = &TrigFieldFunction::velocity;
  else if (fieldName == "dudx")
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
  else if (fieldName == "adaptivity_parameter")
    funcPtr = &TrigFieldFunction::alpha;
  else if (fieldName == "dkdx")
    funcPtr = &TrigFieldFunction::dkdx;
  else if (fieldName == "specific_dissipation_rate")
    funcPtr = &TrigFieldFunction::sdr;
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
  else
    funcPtr = nullptr;

  EXPECT_TRUE(funcPtr != nullptr);

  const auto& meta = bulk.mesh_meta_data();
  EXPECT_EQ(meta.spatial_dimension(), 3u);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);

  for(const stk::mesh::Bucket* bptr : buckets)
  {
    for(stk::mesh::Entity node : *bptr)
    {
      const double* coords = stk::mesh::field_data(coordinates, node);
      double* qNode = stk::mesh::field_data(qField, node);

      ((stv).*(funcPtr))(coords, qNode);
    }
  }
}

template<typename LOOP_BODY>
void init_trigonometric_field(
    const stk::mesh::BulkData& bulk,
    const LOOP_BODY& inner_loop_body)
{
    const auto& meta = bulk.mesh_meta_data();
    EXPECT_EQ(meta.spatial_dimension(), 3u);

    const stk::mesh::Selector selector =
        meta.locally_owned_part() | meta.globally_shared_part();
    const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);

    for(const stk::mesh::Bucket* bptr : buckets)
    {
      for(stk::mesh::Entity node : *bptr)
      {
        inner_loop_body(node);
      }
    }
}

} // anonymous namespace

namespace unit_test_kernel_utils {

void velocity_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& velocity)
{
  // Add additional test functions in future?
  init_trigonometric_field(bulk, coordinates, velocity);
}

void dudx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  GenericFieldType& dudx)
{
  // Add additional test functions in future?
  init_trigonometric_field(bulk, coordinates, dudx);
}

void pressure_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& pressure)
{
  init_trigonometric_field(bulk, coordinates, pressure);
}

void dpdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dpdx)
{
  init_trigonometric_field(bulk, coordinates, dpdx);
}

void temperature_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& temperature)
{
  init_trigonometric_field(bulk, coordinates, temperature);
}

void density_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& density)
{
  init_trigonometric_field(bulk, coordinates, density);
}

void tke_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& tke)
{
  init_trigonometric_field(bulk, coordinates, tke);
}

void alpha_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& alpha)
{
  init_trigonometric_field(bulk, coordinates, alpha);
}

void dkdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dkdx)
{
  init_trigonometric_field(bulk, coordinates, dkdx);
}

void sdr_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& sdr)
{
  init_trigonometric_field(bulk, coordinates, sdr);
}

void dwdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dwdx)
{
  init_trigonometric_field(bulk, coordinates, dwdx);
}

void turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& turbulent_viscosity)
{
  init_trigonometric_field(bulk, coordinates, turbulent_viscosity);
}

void tensor_turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  GenericFieldType& mutij)
{
  init_trigonometric_field(bulk, coordinates, mutij);
}
void sst_f_one_blending_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& sst_f_one_blending)
{
  init_trigonometric_field(bulk, coordinates, sst_f_one_blending);
}

void minimum_distance_to_wall_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& minimum_distance_to_wall)
{
  init_trigonometric_field(bulk, coordinates, minimum_distance_to_wall);
}

void property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const ScalarFieldType& mixFraction,
  ScalarFieldType& property,
  const double primary,
  const double secondary)
{  
  init_trigonometric_field(
    bulk,
    [&](stk::mesh::Entity node){
      const double mixFrac = *stk::mesh::field_data(mixFraction, node);
      double *theProp = stk::mesh::field_data(property, node);
      *theProp = primary*mixFrac + secondary*(1.0-mixFrac);
    });
}

void inverse_property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const ScalarFieldType& mixFraction,
  ScalarFieldType& property,
  const double primary,
  const double secondary)
{  
  init_trigonometric_field(
    bulk,
    [&](stk::mesh::Entity node){
      const double z = *stk::mesh::field_data(mixFraction, node);
      double *theProp = stk::mesh::field_data(property, node);
      *theProp = 1.0/(z/primary + (1.0-z)/secondary);
    });
}

void mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  const ScalarFieldType& mixtureFrac,
  const double znot,
  const double amf)
{  
  const double pi = acos(-1.0);
  init_trigonometric_field(
    bulk,
    [&](stk::mesh::Entity node){
      const double *coords = stk::mesh::field_data(coordinates, node);
      double *mixFrac = stk::mesh::field_data(mixtureFrac, node);      
      const double x = coords[0];
      const double y = coords[1];
      const double z = coords[2];
      *mixFrac = znot*cos(amf*pi*x)*cos(amf*pi*y)*cos(amf*pi*z);
    });
}

void dhdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dhdx)
{
  init_trigonometric_field(bulk, coordinates, dhdx);
}

#ifndef KOKKOS_ENABLE_CUDA

void calc_mass_flow_rate_scs(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& density,
  const VectorFieldType& velocity,
  const GenericFieldType& massFlowRate)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  const ScalarFieldType& densityNp1 = density.field_of_state(stk::mesh::StateNP1);
  const VectorFieldType& velocityNp1 = velocity.field_of_state(stk::mesh::StateNP1);
  auto meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

  dataNeeded.add_cvfem_surface_me(meSCS);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(densityNp1, 1);
  dataNeeded.add_gathered_nodal_field(velocityNp1, ndim);
  dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV,
                                     sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector =
    meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  ngp::Mesh ngpMesh(bulk);
  ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(
    dataNeededNGP, meta.spatial_dimension()) ;

  auto team_exec = sierra::nalu::get_host_team_policy(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
      auto& b = *buckets[team.league_rank()];
      const auto length = b.size();

      EXPECT_EQ(b.topology(), topo);

      sierra::nalu::ScratchViews<double> preReqData(
        team, ndim, topo.num_nodes(), dataNeededNGP);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
          stk::mesh::Entity element = b[k];
          sierra::nalu::fill_pre_req_data(
            dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, element,
            preReqData);

          std::vector<double> rhoU(ndim);
          std::vector<double> v_shape_function(
            meSCS->numIntPoints_ * meSCS->nodesPerElement_);
          meSCS->shape_fcn(v_shape_function.data());
          double *mdot = stk::mesh::field_data(massFlowRate, element);
          auto& v_densityNp1 = preReqData.get_scratch_view_1D(densityNp1);
          auto& v_velocityNp1 = preReqData.get_scratch_view_2D(velocityNp1);
          auto& v_scs_areav = preReqData.get_me_views(
            sierra::nalu::CURRENT_COORDINATES).scs_areav;

          for (int ip=0; ip < meSCS->numIntPoints_; ++ip) {
            for (int j=0; j < ndim; ++j) {
              rhoU[j] = 0.0;
            }
            const int offset = ip * meSCS->nodesPerElement_;

            for (int ic=0; ic < meSCS->nodesPerElement_; ++ic) {
              const double r = v_shape_function[offset+ic];
              for (int j=0; j < ndim; ++j) {
                rhoU[j] += r * v_densityNp1(ic) * v_velocityNp1(ic, j);
              }
            }

            double tmdot = 0.0;
            for (int j=0; j < ndim; j++) {
              tmdot += rhoU[j] * v_scs_areav(ip, j);
            }
            mdot[ip] = tmdot;
          }
        });
    });
}

void calc_projected_nodal_gradient_interior(
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

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  auto meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

  dataNeeded.add_cvfem_surface_me(meSCS);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(scalarField, 1);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  ngp::Mesh ngpMesh(bulk);
  ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension()) ;

  auto v_shape_function = Kokkos::View<double**>("shape_function", meSCS->numIntPoints_, meSCS->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
    auto& b = *buckets[team.league_rank()];
    const auto length = b.size();

    EXPECT_EQ(b.topology(), topo);

    sierra::nalu::ScratchViews<double> preReqData(
      team, ndim, topo.num_nodes(), dataNeededNGP);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
      stk::mesh::Entity element = b[k];
      sierra::nalu::fill_pre_req_data(
        dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, element,
        preReqData);

      meSCS->shape_fcn(v_shape_function.data());
      auto v_dnv = preReqData.get_scratch_view_1D(dnv);
      auto v_scalar = preReqData.get_scratch_view_1D(scalarField);
      auto v_scs_areav = preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scs_areav;
      const ngp::Mesh::ConnectedNodes node_rels = preReqData.elemNodes;
      const int* lrscv = meSCS->adjacentNodes();

      for (int ip = 0; ip < meSCS->numIntPoints_; ++ip) {
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

  auto meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

  dataNeeded.add_cvfem_surface_me(meSCS);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(vectorField, ndim);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  ngp::Mesh ngpMesh(bulk);
  ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension()) ;

  auto v_shape_function = Kokkos::View<double**>("shape_function", meSCS->numIntPoints_, meSCS->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
    auto& b = *buckets[team.league_rank()];
    const auto length = b.size();

    EXPECT_EQ(b.topology(), topo);

    sierra::nalu::ScratchViews<double> preReqData(
      team, ndim, topo.num_nodes(), dataNeededNGP);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
      stk::mesh::Entity element = b[k];
      sierra::nalu::fill_pre_req_data(
        dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, element,
        preReqData);

      meSCS->shape_fcn(v_shape_function.data());
      auto v_dnv = preReqData.get_scratch_view_1D(dnv);
      auto v_vector = preReqData.get_scratch_view_2D(vectorField);
      auto v_scs_areav = preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scs_areav;
      const ngp::Mesh::ConnectedNodes node_rels = preReqData.elemNodes;
      const int* lrscv = meSCS->adjacentNodes();

      for (int di = 0; di < ndim; ++di) {
        for (int ip = 0; ip < meSCS->numIntPoints_; ++ip) {
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

void calc_projected_nodal_gradient_boundary(
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

  auto meBC = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

  dataNeeded.add_cvfem_surface_me(meBC);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(scalarField, 1);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(meta.side_rank(), selector);

  ngp::Mesh ngpMesh(bulk);
  ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension()) ;

  auto v_shape_function = Kokkos::View<double**>("shape_function", meBC->numIntPoints_, meBC->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
    auto& b = *buckets[team.league_rank()];
    const auto length = b.size();

    EXPECT_EQ(b.topology(), topo);

    sierra::nalu::ScratchViews<double> preReqData(team, ndim, topo.num_nodes(), dataNeededNGP);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
      stk::mesh::Entity face = b[k];
      sierra::nalu::fill_pre_req_data(
        dataNeededNGP, ngpMesh, meta.side_rank(), face, preReqData);

      meBC->shape_fcn(v_shape_function.data());
      auto v_dnv = preReqData.get_scratch_view_1D(dnv);
      auto v_scalar = preReqData.get_scratch_view_1D(scalarField);
      auto v_scs_areav = preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scs_areav;
      const ngp::Mesh::ConnectedNodes node_rels = preReqData.elemNodes;
      const int* ipNodeMap = meBC->ipNodeMap();

      for (int ip = 0; ip < meBC->numIntPoints_; ++ip) {
        double qIp = 0.0;
        for (int n = 0; n < meBC->nodesPerElement_; ++n) {
          qIp += v_shape_function(ip, n) * v_scalar(n);
        }

        const int nn = ipNodeMap[ip];
        double* dqdxNN = stk::mesh::field_data(gradField, node_rels[nn]);

        for (int d = 0; d < ndim; ++d) {
          double fac  = qIp * v_scs_areav(ip, d) / v_dnv(nn);
          Kokkos::atomic_add(dqdxNN + d, fac);
        }
      }
    });
  });
}

void calc_projected_nodal_gradient_boundary(
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

  auto meBC = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

  dataNeeded.add_cvfem_surface_me(meBC);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_gathered_nodal_field(vectorField, ndim);
  dataNeeded.add_gathered_nodal_field(dnv, 1);
  dataNeeded.add_gathered_nodal_field(gradField, ndim);
  dataNeeded.add_master_element_call(sierra::nalu::SCS_AREAV, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(meta.side_rank(), selector);

  ngp::Mesh ngpMesh(bulk);
  ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension()) ;

  auto v_shape_function = Kokkos::View<double**>("shape_function", meBC->numIntPoints_, meBC->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
    auto& b = *buckets[team.league_rank()];
    const auto length = b.size();

    EXPECT_EQ(b.topology(), topo);

    sierra::nalu::ScratchViews<double> preReqData(
      team, ndim, topo.num_nodes(), dataNeededNGP);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
      stk::mesh::Entity face = b[k];
      sierra::nalu::fill_pre_req_data(
        dataNeededNGP, ngpMesh, meta.side_rank(), face, preReqData);

      meBC->shape_fcn(v_shape_function.data());
      auto v_dnv = preReqData.get_scratch_view_1D(dnv);
      auto v_vector = preReqData.get_scratch_view_2D(vectorField);
      auto v_scs_areav = preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scs_areav;
      const ngp::Mesh::ConnectedNodes node_rels = preReqData.elemNodes;
      const int* ipNodeMap = meBC->ipNodeMap();

      for (int di = 0; di < ndim; ++di) {
        for (int ip = 0; ip < meBC->numIntPoints_; ++ip) {
          double qIp = 0.0;
          for (int n = 0; n < meBC->nodesPerElement_; ++n) {
            qIp += v_shape_function(ip, n) * v_vector(n,di);
          }

          const int nn = ipNodeMap[ip];
          double* dqdxNN = stk::mesh::field_data(gradField, node_rels[nn]);

          for (int d = 0; d < ndim; ++d) {
            double fac  = qIp * v_scs_areav(ip, d) / v_dnv(nn);
            Kokkos::atomic_add(dqdxNN + di*ndim + d, fac);
          }
        }
      }
    });
  });
}

void calc_dual_nodal_volume(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  const ScalarFieldType& dnvField)
{
  const auto& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  EXPECT_EQ(ndim, 3);

  sierra::nalu::ElemDataRequests dataNeeded(meta);

  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(topo);

  dataNeeded.add_cvfem_volume_me(meSCV);
  dataNeeded.add_coordinates_field(coordinates, ndim, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_master_element_call(sierra::nalu::SCV_VOLUME, sierra::nalu::CURRENT_COORDINATES);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  ngp::Mesh ngpMesh(bulk);
  ngp::FieldManager fieldMgr(bulk);
  sierra::nalu::ElemDataRequestsGPU dataNeededNGP(fieldMgr, dataNeeded, meta.get_fields().size());

  const int bytes_per_team = 0;
  const int bytes_per_thread = sierra::nalu::get_num_bytes_pre_req_data<double>(dataNeededNGP, meta.spatial_dimension()) ;

  auto v_shape_function = Kokkos::View<double**>("shape_function", meSCV->numIntPoints_, meSCV->nodesPerElement_);

  auto team_exec = sierra::nalu::get_host_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team) {
    auto& b = *buckets[team.league_rank()];
    const auto length = b.size();

    EXPECT_EQ(b.topology(), topo);

    sierra::nalu::ScratchViews<double> preReqData(
      team, ndim, topo.num_nodes(), dataNeededNGP);

    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, length), [&](const size_t& k) {
      stk::mesh::Entity element = b[k];
      sierra::nalu::fill_pre_req_data(
        dataNeededNGP, ngpMesh, stk::topology::ELEMENT_RANK, element,
        preReqData);

      auto v_scv_vol = preReqData.get_me_views(sierra::nalu::CURRENT_COORDINATES).scv_volume;
      const ngp::Mesh::ConnectedNodes node_rels = preReqData.elemNodes;
      const int* ipNodeMap = meSCV->ipNodeMap();

      for (int ip = 0; ip < meSCV->numIntPoints_; ++ip) {
        double volIp = v_scv_vol(ip);
        Kokkos::atomic_add(stk::mesh::field_data(dnvField, node_rels[ipNodeMap[ip]]),volIp);
      }
    });
  });
}

void calc_projected_nodal_gradient(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  ScalarFieldType& dnv,
  const ScalarFieldType& scalarField,
  VectorFieldType& gradField)
{
  // for now
  EXPECT_TRUE(topo != stk::topology::PYRAMID_5 && topo != stk::topology::WEDGE_6);
  stk::mesh::field_fill(0.0, dnv);
  stk::mesh::field_fill(0.0, gradField);

  calc_dual_nodal_volume(bulk, topo, coordinates, dnv);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&dnv});
  }

  calc_projected_nodal_gradient_interior(bulk, topo, coordinates, dnv, scalarField, gradField);
  calc_projected_nodal_gradient_boundary(bulk, topo.side_topology(0), coordinates, dnv, scalarField, gradField);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&gradField});
  }

}

void calc_projected_nodal_gradient(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  ScalarFieldType& dnv,
  const VectorFieldType& vectorField,
  GenericFieldType& gradField)
{
  // for now
  EXPECT_TRUE(topo != stk::topology::PYRAMID_5 && topo != stk::topology::WEDGE_6);
  stk::mesh::field_fill(0.0, dnv);

  calc_dual_nodal_volume(bulk, topo, coordinates, dnv);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&dnv});
  }

  stk::mesh::field_fill(0.0, gradField);
  calc_projected_nodal_gradient_interior(bulk, topo, coordinates, dnv, vectorField, gradField);
  calc_projected_nodal_gradient_boundary(bulk, topo.side_topology(0), coordinates, dnv, vectorField, gradField);
  if (bulk.parallel_size() > 1) {
    stk::mesh::parallel_sum(bulk, {&gradField});
  }
}

#endif // KOKKOS_ENABLE_CUDA

void expect_all_near(
  const Kokkos::View<double*>& calcValue,
  const double* exactValue,
  const double tol)
{
  const int length = calcValue.extent(0);

  for (int i=0; i < length; ++i) {
    EXPECT_NEAR(calcValue[i], exactValue[i], tol);
  }
}

void expect_all_near(
  const Kokkos::View<double*>& calcValue,
  const double exactValue,
  const double tol)
{
  const int length = calcValue.extent(0);

  for (int i=0; i < length; ++i) {
    EXPECT_NEAR(calcValue[i], exactValue, tol);
  }
}

void expect_all_near(
  const Kokkos::View<double**>& calcValue,
  const double* exactValue,
  const double tol)
{
  const int dim1 = calcValue.extent(0);
  const int dim2 = calcValue.extent(1);

  for (int i=0; i < dim1; i++)
    for (int j=0; j < dim2; j++)
      EXPECT_NEAR(calcValue(i,j),exactValue[i*dim2+j], tol);
}

} // unit_test_kernel_utils

