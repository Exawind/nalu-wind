/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef UNITTESTKERNELUTILS_H
#define UNITTESTKERNELUTILS_H

#include "UnitTestUtils.h"

#include "SolutionOptions.h"
#include "kernel/Kernel.h"
#include "ElemDataRequests.h"
#include "ScratchViews.h"
#include "CopyAndInterleave.h"
#include "AlgTraits.h"
#include "KokkosInterface.h"
#include "TimeIntegrator.h"

#include <gtest/gtest.h>

#include <mpi.h>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace unit_test_kernel_utils {

void velocity_test_function(
  const stk::mesh::BulkData&,
  const VectorFieldType& coordinates,
  VectorFieldType& velocity);

void dudx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  GenericFieldType& dudx);

void pressure_test_function(
  const stk::mesh::BulkData&,
  const VectorFieldType& coordinates,
  ScalarFieldType& pressure);

void dpdx_test_function(
  const stk::mesh::BulkData&,
  const VectorFieldType& coordinates,
  VectorFieldType& dpdx);

void temperature_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& temperature);

void density_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& density);

void tke_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& tke);

void alpha_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& alpha);

void dkdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dkdx);

void sdr_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& sdr);

void dwdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dwdx);

void turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& turbulent_viscosity);

void tensor_turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  GenericFieldType& mutij);

void sst_f_one_blending_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& sst_f_one_blending);

void minimum_distance_to_wall_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& minimum_distance_to_wall);

void property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData&,
  const ScalarFieldType& mixFraction,
  ScalarFieldType& property,
  const double primary,
  const double secondary);

void inverse_property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData&,
  const ScalarFieldType& mixFraction,
  ScalarFieldType& property,
  const double primary,
  const double secondary);

void mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  const ScalarFieldType& mixFrac,
  const double znot,
  const double amf);

void dhdx_test_function(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& dhdx);

void calc_mass_flow_rate_scs(
  stk::mesh::BulkData&,
  const stk::topology&,
  const VectorFieldType&,
  const ScalarFieldType&,
  const VectorFieldType&,
  const GenericFieldType&);

void calc_projected_nodal_gradient(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  ScalarFieldType& dualNodalVolume,
  const ScalarFieldType& scalarField,
  VectorFieldType& vectorField);

void calc_projected_nodal_gradient(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const VectorFieldType& coordinates,
  ScalarFieldType& dualNodalVolume,
  const VectorFieldType& vectorField,
  GenericFieldType& tensorField);

void expect_all_near(
  const Kokkos::View<double*>& calcValue,
  const double* exactValue,
  const double tol = 1.0e-15);

void expect_all_near(
  const Kokkos::View<double*>& calcValue,
  const double exactValue,
  const double tol = 1.0e-15);

void expect_all_near(
  const Kokkos::View<double**>& calcValue,
  const double* exactValue,
  const double tol = 1.0e-15);

template<int N>
void expect_all_near(
  const Kokkos::View<double**>& calcValue,
  const double (*exactValue)[N],
  const double tol = 1.0e-15)
{
  const int dim1 = calcValue.extent(0);
  const int dim2 = calcValue.extent(1);
  EXPECT_EQ(dim2, N);

  for (int i=0; i < dim1; ++i) {
    for (int j=0; j < dim2; ++j) {
      EXPECT_NEAR(calcValue(i,j), exactValue[i][j], tol);
    }
  }
}

template<int N>
void expect_all_near(
  const Kokkos::View<double**>& calcValue,
  const double exactValue,
  const double tol = 1.0e-15)
{
  const int dim1 = calcValue.extent(0);
  const int dim2 = calcValue.extent(1);
  EXPECT_EQ(dim2, N);

  for (int i=0; i < dim1; ++i) {
    for (int j=0; j < dim2; ++j) {
      EXPECT_NEAR(calcValue(i,j), exactValue, tol);
    }
  }
}

}

/** Base class for all computational kernel testing setups
 *
 *  Initializes the STK mesh data structures and stores a handle to the
 *  coordinates field. Subclasses must declare the necessary computational
 *  fields necessary for the tests.
 */
class TestKernelHex8Mesh : public ::testing::Test
{
public:
  TestKernelHex8Mesh()
    : comm_(MPI_COMM_WORLD),
      spatialDim_(3),
      meta_(spatialDim_),
      bulk_(meta_, comm_),
      solnOpts_()
  {}

  virtual ~TestKernelHex8Mesh() {}

  void fill_mesh(bool doPerturb = false)
  {

    unit_test_utils::fill_mesh_1_elem_per_proc_hex8(bulk_);
    if (doPerturb) {
      unit_test_utils::perturb_coord_hex_8(bulk_, 0.125);
    }

    partVec_ = {meta_.get_part("block_1")};

    coordinates_ = static_cast<const VectorFieldType*>(meta_.coordinate_field());

    EXPECT_TRUE(coordinates_ != nullptr);
  }

  stk::ParallelMachine comm_;
  unsigned spatialDim_;
  stk::mesh::MetaData meta_;
  stk::mesh::BulkData bulk_;
  stk::mesh::PartVector partVec_;

  sierra::nalu::SolutionOptions solnOpts_;

  const VectorFieldType* coordinates_{nullptr};
};

/** Test Fixture for Low-Mach Kernels
 *
 *  This test fixture performs the following actions:
 *    - Create a HEX8 mesh with one element
 *    - Declare `velocity`, `pressure`, `density`, and `dpdx` fields
 *    - Initialize the fields with SteadyTaylorVortex solution
 *    - `density` is initialized to 1.0
 */
class LowMachKernelHex8Mesh : public TestKernelHex8Mesh
{
public:
  LowMachKernelHex8Mesh()
    : TestKernelHex8Mesh(),
      velocity_(
        &meta_.declare_field<VectorFieldType>(
          stk::topology::NODE_RANK, "velocity",2)),
      dpdx_(
        &meta_.declare_field<VectorFieldType>(
          stk::topology::NODE_RANK, "dpdx",2)),
      density_(
        &meta_.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "density",2)),
      pressure_(
        &meta_.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "pressure",2)),
      Udiag_(
        &meta_.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "momentum_diag", 2))
  {
    stk::mesh::put_field_on_mesh(*velocity_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*dpdx_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*density_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*pressure_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*Udiag_, meta_.universal_part(), 1, nullptr);
  }

  virtual ~LowMachKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(bool doPerturb = false)
  {
    fill_mesh(doPerturb);

    unit_test_kernel_utils::velocity_test_function(bulk_, *coordinates_, *velocity_);
    unit_test_kernel_utils::pressure_test_function(bulk_, *coordinates_, *pressure_);
    unit_test_kernel_utils::dpdx_test_function(bulk_, *coordinates_, *dpdx_);
    stk::mesh::field_fill(1.0, *density_);
    stk::mesh::field_fill(1.0, *Udiag_);
  }

  VectorFieldType* velocity_{nullptr};
  VectorFieldType* dpdx_{nullptr};
  ScalarFieldType* density_{nullptr};
  ScalarFieldType* pressure_{nullptr};
  ScalarFieldType* Udiag_{nullptr};
};

class ContinuityKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  virtual ~ContinuityKernelHex8Mesh() {}
};

class MomentumKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  MomentumKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      massFlowRate_(
        &meta_.declare_field<GenericFieldType>(
          stk::topology::ELEM_RANK, "mass_flow_rate_scs")),
      viscosity_(
        &meta_.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "viscosity")),
      dudx_(
        &meta_.declare_field<GenericFieldType>(
          stk::topology::NODE_RANK, "dudx")),
     temperature_(
        &meta_.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "temperature"))
  {
    const auto& meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(stk::topology::HEX_8);
    stk::mesh::put_field_on_mesh(*massFlowRate_, meta_.universal_part(), meSCS->num_integration_points(), nullptr);
    stk::mesh::put_field_on_mesh(*viscosity_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*dudx_, meta_.universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*temperature_, meta_.universal_part(), 1, nullptr);
  }

  virtual ~MomentumKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(bool doPerturb = false)
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb);
    unit_test_kernel_utils::calc_mass_flow_rate_scs(
      bulk_, stk::topology::HEX_8, *coordinates_, *density_, *velocity_, *massFlowRate_);
    unit_test_kernel_utils::dudx_test_function(bulk_, *coordinates_, *dudx_);
    stk::mesh::field_fill(0.1, *viscosity_);
    stk::mesh::field_fill(300.0, *temperature_);
  }

  GenericFieldType* massFlowRate_{nullptr};
  ScalarFieldType* viscosity_{nullptr};
  GenericFieldType* dudx_{nullptr};
  ScalarFieldType* temperature_{nullptr};
};

/** Test Fixture for the SST Kernels
 *
 */
class SSTKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  SSTKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      tke_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      sdr_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "specific_dissipation_rate")),
      tvisc_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "turbulent_viscosity")),
      maxLengthScale_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "sst_max_length_scale")),
      fOneBlend_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "sst_f_one_blending"))
  {
    stk::mesh::put_field_on_mesh(*tke_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*sdr_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*tvisc_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*maxLengthScale_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*fOneBlend_, meta_.universal_part(), 1, nullptr);
  }

  virtual ~SSTKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(bool doPerturb = false)
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb);
    stk::mesh::field_fill(0.3, *tvisc_);
    stk::mesh::field_fill(0.5, *maxLengthScale_);
    unit_test_kernel_utils::density_test_function(
      bulk_, *coordinates_, *density_);
    unit_test_kernel_utils::tke_test_function(bulk_, *coordinates_, *tke_);
    unit_test_kernel_utils::sdr_test_function(bulk_, *coordinates_, *sdr_);
    unit_test_kernel_utils::sst_f_one_blending_test_function(
      bulk_, *coordinates_, *fOneBlend_);
  }

  ScalarFieldType* tke_{nullptr};
  ScalarFieldType* sdr_{nullptr};
  ScalarFieldType* tvisc_{nullptr};
  ScalarFieldType* maxLengthScale_{nullptr};
  ScalarFieldType* fOneBlend_{nullptr};
};

/** Test Fixture for the hybrid turbulence Kernels
 *
 */
class HybridTurbKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  HybridTurbKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      tke_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      alpha_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "adaptivity_parameter")),
      mutij_(&meta_.declare_field<GenericFieldType>(
        stk::topology::NODE_RANK, "tensor_turbulent_viscosity"))
  {
    stk::mesh::put_field_on_mesh(*tke_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*alpha_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(
      *mutij_, meta_.universal_part(), spatialDim_ * spatialDim_, nullptr);
  }

  virtual ~HybridTurbKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(bool doPerturb = false)
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb);
    stk::mesh::field_fill(0.0, *tke_);
    stk::mesh::field_fill(1.0, *alpha_);
    unit_test_kernel_utils::tensor_turbulent_viscosity_test_function(bulk_, *coordinates_, *mutij_);
    /* unit_test_kernel_utils::tke_test_function(bulk_, *coordinates_, *tke_); */
    /* unit_test_kernel_utils::alpha_test_function(bulk_, *coordinates_, *alpha_); */
  }

  ScalarFieldType* tke_{nullptr};
  ScalarFieldType* alpha_{nullptr};
  GenericFieldType* mutij_{nullptr};
};

/** Text fixture for heat conduction equation kernels
 *
 *  This test fixture performs the following actions:
 *    - Create a HEX8 mesh with one element
 *    - Declare `temperature` and `thermal_conductivity` fields
 *    - Initialize the fields with steady 3-D thermal solution
 *    - `thermal_conductivity` is initialized to 1.0
 */
class HeatCondKernelHex8Mesh : public TestKernelHex8Mesh
{
public:
  HeatCondKernelHex8Mesh()
    : TestKernelHex8Mesh(),
      temperature_(
        &meta_.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "temperature",2)),
      thermalCond_(
        &meta_.declare_field<ScalarFieldType>(
          stk::topology::NODE_RANK, "thermal_conductivity",2))
  {
    stk::mesh::put_field_on_mesh(*temperature_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*thermalCond_, meta_.universal_part(), 1, nullptr);
  }

  void fill_mesh_and_init_fields(bool doPerturb = false)
  {
    fill_mesh(doPerturb);

    unit_test_kernel_utils::temperature_test_function(bulk_, *coordinates_, *temperature_);
    stk::mesh::field_fill(1.0, *thermalCond_);
  }

  ScalarFieldType* temperature_{nullptr};
  ScalarFieldType* thermalCond_{nullptr};
};

/** Text fixture for mixture fraction equation kernels
 *
 *  This test fixture performs the following actions:
 *    - Create a HEX8 mesh with one element
 *    - Declare all of the set of fields required (autonomous from LowMach/Mom/Cont)
 *    - Initialize the fields with steady 3-D solution; properties of helium/air
 */
class MixtureFractionKernelHex8Mesh : public TestKernelHex8Mesh
{
public:
  MixtureFractionKernelHex8Mesh()
    : TestKernelHex8Mesh(),
    mixFraction_(&meta_.declare_field<ScalarFieldType>(stk::topology::NODE_RANK,
                                                       "mixture_fraction")),
    scalarQ_(&meta_.declare_field<ScalarFieldType>(stk::topology::NODE_RANK,
                                                    "scalarQ", 2)),
    velocity_(&meta_.declare_field<VectorFieldType>(stk::topology::NODE_RANK,
                                                    "velocity")),
    density_(&meta_.declare_field<ScalarFieldType>(stk::topology::NODE_RANK,
                                                   "density", 2)),
    viscosity_(&meta_.declare_field<ScalarFieldType>(stk::topology::NODE_RANK,
                                                     "viscosity")),
    effectiveViscosity_(&meta_.declare_field<ScalarFieldType>(stk::topology::NODE_RANK,
                                                              "effective_viscosity")),
    massFlowRate_(&meta_.declare_field<GenericFieldType>(stk::topology::ELEM_RANK,
                                                         "mass_flow_rate_scs")),

    znot_(1.0),
    amf_(2.0),
    lamSc_(0.9),
    trbSc_(1.1),
    rhoPrimary_(0.163),
    rhoSecondary_(1.18),
    viscPrimary_(1.967e-5),
    viscSecondary_(1.85e-5)
  {
    const auto& meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(stk::topology::HEX_8);
    stk::mesh::put_field_on_mesh(*mixFraction_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*scalarQ_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*velocity_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*density_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*viscosity_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*massFlowRate_, meta_.universal_part(), meSCS->num_integration_points(), nullptr);
  }
  virtual ~MixtureFractionKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(bool doPerturb = false)
  {
    fill_mesh(doPerturb);

    unit_test_kernel_utils::mixture_fraction_test_function(bulk_, *coordinates_, *mixFraction_, amf_, znot_);
    unit_test_kernel_utils::mixture_fraction_test_function(bulk_, *coordinates_, *scalarQ_, amf_, znot_);
    unit_test_kernel_utils::velocity_test_function(bulk_, *coordinates_, *velocity_);
    unit_test_kernel_utils::inverse_property_from_mixture_fraction_test_function(bulk_, *mixFraction_, *density_,
                                                                                 rhoPrimary_, rhoSecondary_);
    unit_test_kernel_utils::property_from_mixture_fraction_test_function(bulk_, *mixFraction_, *viscosity_,
                                                                         viscPrimary_, viscSecondary_);
    unit_test_kernel_utils::calc_mass_flow_rate_scs(
      bulk_, stk::topology::HEX_8, *coordinates_, *density_, *velocity_, *massFlowRate_);
  }

  ScalarFieldType* mixFraction_{nullptr};
  ScalarFieldType* scalarQ_{nullptr};
  VectorFieldType* velocity_{nullptr};
  ScalarFieldType* density_{nullptr};
  ScalarFieldType* viscosity_{nullptr};
  ScalarFieldType* effectiveViscosity_{nullptr};
  GenericFieldType* massFlowRate_{nullptr};

  const double znot_;
  const double amf_;
  const double lamSc_;
  const double trbSc_;
  const double rhoPrimary_;
  const double rhoSecondary_;
  const double viscPrimary_;
  const double viscSecondary_;
};

/** Text fixture for actuator source kernels
 *
 *  This test fixture performs the following actions:
 *    - Create a HEX8 mesh with one element
 *    - Declare all of the set of fields required (actuator_source)
 *    - Initialize the field with steady 3-D solution;
 */
class ActuatorSourceKernelHex8Mesh : public TestKernelHex8Mesh
{
public:
  ActuatorSourceKernelHex8Mesh()
    : TestKernelHex8Mesh(),
    actuator_source_(&meta_.declare_field<VectorFieldType>(stk::topology::NODE_RANK,
                                                          "actuator_source")),
    actuator_source_lhs_(&meta_.declare_field<VectorFieldType>(stk::topology::NODE_RANK,
                                                           "actuator_source_lhs"))
  {
    stk::mesh::put_field_on_mesh(*actuator_source_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*actuator_source_lhs_, meta_.universal_part(), spatialDim_, nullptr);
  }

  virtual ~ActuatorSourceKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(bool doPerturb = false)
  {
    fill_mesh(doPerturb);

    std::vector<double> act_source(spatialDim_,0.0);
    for(size_t j=0;j<spatialDim_;j++) act_source[j] = j+1;
    stk::mesh::field_fill_component(act_source.data(), *actuator_source_);

    std::vector<double> act_source_lhs(spatialDim_,0.0);
    for(size_t j=0;j<spatialDim_;j++) act_source_lhs[j] = 0.1*(j+1);
    stk::mesh::field_fill_component(act_source_lhs.data(), *actuator_source_lhs_);
  }

  VectorFieldType* actuator_source_{nullptr};
  VectorFieldType* actuator_source_lhs_{nullptr};

};

#endif /* UNITTESTKERNELUTILS_H */
