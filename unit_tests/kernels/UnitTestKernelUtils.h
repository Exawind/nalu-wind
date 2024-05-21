// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNITTESTKERNELUTILS_H
#define UNITTESTKERNELUTILS_H

#include "UnitTestUtils.h"

#include "FieldTypeDef.h"
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
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& velocity);

void dudx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::TensorFieldType& dudx);

void pressure_test_function(
  const stk::mesh::BulkData&,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& pressure);

void dpdx_test_function(
  const stk::mesh::BulkData&,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dpdx);

void temperature_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& temperature);

void density_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& density);

void tke_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& tke);

void alpha_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& alpha);

void dkdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dkdx);

void sdr_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& sdr);

void tdr_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& tdr);

void dwdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dwdx);

void turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& turbulent_viscosity);

void tensor_turbulent_viscosity_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::GenericFieldType& mutij);

void sst_f_one_blending_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& sst_f_one_blending);

void iddes_rans_indicator_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& iddes_rans_indicator);

void minimum_distance_to_wall_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& minimum_distance_to_wall);

void dplus_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::ScalarFieldType& dplus);

void property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData&,
  const sierra::nalu::ScalarFieldType& mixFraction,
  sierra::nalu::ScalarFieldType& property,
  const double primary,
  const double secondary);

void inverse_property_from_mixture_fraction_test_function(
  const stk::mesh::BulkData&,
  const sierra::nalu::ScalarFieldType& mixFraction,
  sierra::nalu::ScalarFieldType& property,
  const double primary,
  const double secondary);

void mixture_fraction_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  const sierra::nalu::ScalarFieldType& mixFrac,
  const double znot,
  const double amf);

void dhdx_test_function(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::VectorFieldType& dhdx);

void calc_edge_area_vec(
  const stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const sierra::nalu::VectorFieldType& coordinates,
  const sierra::nalu::VectorFieldType& edgeAreaVec);

void calc_exposed_area_vec(
  const stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const sierra::nalu::VectorFieldType& coordinates,
  sierra::nalu::GenericFieldType& exposedAreaVec);

void calc_mass_flow_rate(
  const stk::mesh::BulkData&,
  const sierra::nalu::VectorFieldType&,
  const sierra::nalu::ScalarFieldType&,
  const sierra::nalu::VectorFieldType&,
  sierra::nalu::ScalarFieldType&);

void calc_mass_flow_rate_scs(
  stk::mesh::BulkData&,
  const stk::topology&,
  const sierra::nalu::VectorFieldType&,
  const sierra::nalu::ScalarFieldType&,
  const sierra::nalu::VectorFieldType&,
  const sierra::nalu::GenericFieldType&);

void calc_open_mass_flow_rate(
  stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  const sierra::nalu::VectorFieldType& coordinates,
  const sierra::nalu::ScalarFieldType& density,
  const sierra::nalu::VectorFieldType& velocity,
  const sierra::nalu::GenericFieldType& exposedAreaVec,
  const sierra::nalu::GenericFieldType& massFlowRate);

void expect_all_near(
  const Kokkos::View<double*>& calcValue,
  const double* exactValue,
  const double tol = 1.0e-15);

void expect_all_near(
  const Kokkos::View<double*>& calcValue,
  const double exactValue,
  const double tol = 1.0e-15);

void expect_all_near_2d(
  const Kokkos::View<double**>& calcValue,
  const double* exactValue,
  const double tol = 1.0e-15);

template <int N>
void
expect_all_near(
  const Kokkos::View<double**>& devCalcValue,
  const double (*exactValue)[N],
  const double tol = 1.0e-15)
{
  const int dim1 = devCalcValue.extent(0);
  const int dim2 = devCalcValue.extent(1);
  EXPECT_EQ(dim2, N);

  auto hostCalcValue =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devCalcValue);

  for (int i = 0; i < dim1; ++i) {
    for (int j = 0; j < dim2; ++j) {
      EXPECT_NEAR(hostCalcValue(i, j), exactValue[i][j], tol);
    }
  }
}

template <int N>
void
expect_all_near(
  const Kokkos::View<double**>& devCalcValue,
  const double exactValue,
  const double tol = 1.0e-15)
{
  const int dim1 = devCalcValue.extent(0);
  const int dim2 = devCalcValue.extent(1);
  EXPECT_EQ(dim2, N);

  auto hostCalcValue =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devCalcValue);

  for (int i = 0; i < dim1; ++i) {
    for (int j = 0; j < dim2; ++j) {
      EXPECT_NEAR(hostCalcValue(i, j), exactValue, tol);
    }
  }
}

} // namespace unit_test_kernel_utils

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
    : comm_(MPI_COMM_WORLD), spatialDim_(3), solnOpts_(), coordinates_(nullptr)
  {
    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(spatialDim_);
    bulk_ = meshBuilder.create();
    meta_ = &bulk_->mesh_meta_data();
    meta_->use_simple_fields();

    naluGlobalId_ = &meta_->declare_field<stk::mesh::EntityId>(
      stk::topology::NODE_RANK, "nalu_global_id", 1);
    tpetGlobalId_ = &meta_->declare_field<sierra::nalu::TpetIdType>(
      stk::topology::NODE_RANK, "tpet_global_id", 1);
    dnvField_ = &meta_->declare_field<double>(
      stk::topology::NODE_RANK, "dual_nodal_volume", 3);
    divMeshVelField_ = &meta_->declare_field<double>(
      stk::topology::NODE_RANK, "div_mesh_velocity");
    edgeAreaVec_ = &meta_->declare_field<double>(
      stk::topology::EDGE_RANK, "edge_area_vector");
    elementVolume_ =
      &meta_->declare_field<double>(stk::topology::ELEM_RANK, "element_volume");
    exposedAreaVec_ =
      &meta_->declare_field<double>(meta_->side_rank(), "exposed_area_vector");

    stk::mesh::put_field_on_mesh(
      *naluGlobalId_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *tpetGlobalId_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*dnvField_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *divMeshVelField_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *edgeAreaVec_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *edgeAreaVec_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *elementVolume_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *exposedAreaVec_, meta_->universal_part(),
      spatialDim_ * sierra::nalu::AlgTraitsQuad4::numScsIp_, nullptr);
  }

  virtual ~TestKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false)
  {
    std::string meshSpec =
      "generated:1x1x" + std::to_string(bulk_->parallel_size());
    if (generateSidesets)
      meshSpec += "|sideset:xXyYzZ";
    unit_test_utils::fill_hex8_mesh(meshSpec, *bulk_);
    if (doPerturb) {
      unit_test_utils::perturb_coord_hex_8(*bulk_, 0.125);
    }

    partVec_ = {meta_->get_part("block_1")};

    coordinates_ = static_cast<const sierra::nalu::VectorFieldType*>(
      meta_->coordinate_field());

    EXPECT_TRUE(coordinates_ != nullptr);

    stk::mesh::field_fill(0.125, *dnvField_);
    stk::mesh::field_fill(1.25, *divMeshVelField_);
    unit_test_kernel_utils::calc_edge_area_vec(
      *bulk_, sierra::nalu::AlgTraitsHex8::topo_, *coordinates_, *edgeAreaVec_);
    unit_test_kernel_utils::calc_exposed_area_vec(
      *bulk_, sierra::nalu::AlgTraitsQuad4::topo_, *coordinates_,
      *exposedAreaVec_);
  }

  stk::ParallelMachine comm_;
  unsigned spatialDim_;
  stk::mesh::MetaData* meta_;
  std::shared_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::PartVector partVec_;

  sierra::nalu::SolutionOptions solnOpts_;
#ifdef NALU_USES_TRILINOS_SOLVERS
  using GlobalOrdinal = Tpetra::Details::DefaultTypes::global_ordinal_type;
#else
  using GlobalOrdinal = int64_t;
#endif
  using TpetIDFieldType = stk::mesh::Field<GlobalOrdinal>;

  const sierra::nalu::VectorFieldType* coordinates_{nullptr};
  sierra::nalu::GlobalIdFieldType* naluGlobalId_{nullptr};
  TpetIDFieldType* tpetGlobalId_{nullptr};
  sierra::nalu::ScalarFieldType* dnvField_{nullptr};
  sierra::nalu::ScalarFieldType* divMeshVelField_{nullptr};
  sierra::nalu::VectorFieldType* edgeAreaVec_{nullptr};
  sierra::nalu::ScalarFieldType* elementVolume_{nullptr};
  sierra::nalu::GenericFieldType* exposedAreaVec_{nullptr};
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
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "velocity", 2)),
      dpdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dpdx", 2)),
      density_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "density", 2)),
      pressure_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "pressure", 2)),
      Udiag_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "momentum_diag", 2)),
      velocityBC_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "velocity_bc")),
      dynP_(
        &meta_->declare_field<double>(meta_->side_rank(), "dynamic_pressure"))
  {
    stk::mesh::put_field_on_mesh(
      *velocity_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *velocity_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *dpdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dpdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(*density_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*pressure_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*Udiag_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *velocityBC_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *velocityBC_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *dynP_, meta_->universal_part(), sierra::nalu::AlgTraitsQuad4::numScsIp_,
      nullptr);
  }

  virtual ~LowMachKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    TestKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb, generateSidesets);

    unit_test_kernel_utils::velocity_test_function(
      *bulk_, *coordinates_, *velocity_);
    unit_test_kernel_utils::pressure_test_function(
      *bulk_, *coordinates_, *pressure_);
    unit_test_kernel_utils::dpdx_test_function(*bulk_, *coordinates_, *dpdx_);
    stk::mesh::field_fill(0.0, *dynP_);
    stk::mesh::field_fill(1.0, *density_);
    stk::mesh::field_fill(1.0, *Udiag_);
    unit_test_kernel_utils::velocity_test_function(
      *bulk_, *coordinates_, *velocityBC_);
  }

  sierra::nalu::VectorFieldType* velocity_{nullptr};
  sierra::nalu::VectorFieldType* dpdx_{nullptr};
  sierra::nalu::ScalarFieldType* density_{nullptr};
  sierra::nalu::ScalarFieldType* pressure_{nullptr};
  sierra::nalu::ScalarFieldType* Udiag_{nullptr};
  sierra::nalu::VectorFieldType* velocityBC_{nullptr};
  sierra::nalu::GenericFieldType* dynP_{nullptr};
};

class ContinuityKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  ContinuityKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      pressureBC_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "pressure_bc")),
      dynP_(
        &meta_->declare_field<double>(meta_->side_rank(), "dynamic_pressure"))
  {
    stk::mesh::put_field_on_mesh(
      *pressureBC_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dynP_, meta_->universal_part(), sierra::nalu::AlgTraitsQuad4::numScsIp_,
      nullptr);
  }

  virtual ~ContinuityKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(0.0, *pressureBC_);
    stk::mesh::field_fill(0.0, *dynP_);
  }

private:
  sierra::nalu::ScalarFieldType* pressureBC_{nullptr};
  sierra::nalu::GenericFieldType* dynP_{nullptr};
};

// Provide separate namespace for Edge kernel tests
class ContinuityEdgeHex8Mesh : public ContinuityKernelHex8Mesh
{
};

class ContinuityNodeHex8Mesh : public ContinuityKernelHex8Mesh
{
};

class MomentumKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  MomentumKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      massFlowRate_(&meta_->declare_field<double>(
        stk::topology::ELEM_RANK, "mass_flow_rate_scs")),
      viscosity_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      dudx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx")),
      temperature_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "temperature")),
      openMassFlowRate_(&meta_->declare_field<double>(
        meta_->side_rank(), "open_mass_flow_rate")),
      dynP_(
        &meta_->declare_field<double>(meta_->side_rank(), "dynamic_pressure")),
      openVelocityBC_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "open_velocity_bc"))
  {
    const auto& meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        stk::topology::HEX_8);
    stk::mesh::put_field_on_mesh(
      *massFlowRate_, meta_->universal_part(), meSCS->num_integration_points(),
      nullptr);
    stk::mesh::put_field_on_mesh(*viscosity_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *dudx_, stk::io::FieldOutputType::FULL_TENSOR_36);
    stk::mesh::put_field_on_mesh(
      *temperature_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *openMassFlowRate_, meta_->universal_part(),
      sierra::nalu::AlgTraitsQuad4::numScsIp_, nullptr);
    stk::mesh::put_field_on_mesh(
      *dynP_, meta_->universal_part(), sierra::nalu::AlgTraitsQuad4::numScsIp_,
      nullptr);
    stk::mesh::put_field_on_mesh(
      *openVelocityBC_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *openVelocityBC_, stk::io::FieldOutputType::VECTOR_3D);
  }

  virtual ~MomentumKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(0., *dynP_);
    unit_test_kernel_utils::calc_mass_flow_rate_scs(
      *bulk_, stk::topology::HEX_8, *coordinates_, *density_, *velocity_,
      *massFlowRate_);
    unit_test_kernel_utils::dudx_test_function(*bulk_, *coordinates_, *dudx_);
    stk::mesh::field_fill(0.1, *viscosity_);
    stk::mesh::field_fill(300.0, *temperature_);
    unit_test_kernel_utils::calc_open_mass_flow_rate(
      *bulk_, stk::topology::QUAD_4, *coordinates_, *density_, *velocity_,
      *exposedAreaVec_, *openMassFlowRate_);
  }

  sierra::nalu::GenericFieldType* massFlowRate_{nullptr};
  sierra::nalu::ScalarFieldType* viscosity_{nullptr};
  sierra::nalu::TensorFieldType* dudx_{nullptr};
  sierra::nalu::ScalarFieldType* temperature_{nullptr};
  sierra::nalu::GenericFieldType* openMassFlowRate_{nullptr};
  sierra::nalu::GenericFieldType* dynP_{nullptr};
  sierra::nalu::VectorFieldType* openVelocityBC_{nullptr};
};

// Provide separate namespace for Edge kernel tests
class MomentumEdgeHex8Mesh : public MomentumKernelHex8Mesh
{
public:
  MomentumEdgeHex8Mesh()
    : MomentumKernelHex8Mesh(),
      massFlowRateEdge_(&meta_->declare_field<double>(
        stk::topology::EDGE_RANK, "mass_flow_rate")),
      pecletFactor_(&meta_->declare_field<double>(
        stk::topology::EDGE_RANK, "peclet_factor")),
      ablWallNodeMask_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "abl_wall_no_slip_wall_func_node_mask"))
  {
    stk::mesh::put_field_on_mesh(
      *massFlowRateEdge_, meta_->universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *pecletFactor_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *ablWallNodeMask_, meta_->universal_part(), nullptr);
  }

  virtual ~MomentumEdgeHex8Mesh() = default;

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    MomentumKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    unit_test_kernel_utils::calc_mass_flow_rate(
      *bulk_, *velocity_, *density_, *edgeAreaVec_, *massFlowRateEdge_);
    stk::mesh::field_fill(1.0, *ablWallNodeMask_);
    ablWallNodeMask_->modify_on_host();
    ablWallNodeMask_->sync_to_device();
  }

  sierra::nalu::ScalarFieldType* massFlowRateEdge_{nullptr};
  sierra::nalu::ScalarFieldType* pecletFactor_{nullptr};
  sierra::nalu::ScalarFieldType* maxPecletFactor_{nullptr};
  sierra::nalu::ScalarFieldType* ablWallNodeMask_{nullptr};
};

class MomentumABLKernelHex8Mesh : public MomentumKernelHex8Mesh
{
public:
  MomentumABLKernelHex8Mesh()
    : MomentumKernelHex8Mesh(),
      wallVelocityBC_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "wall_velocity_bc")),
      bcHeatFlux_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "heat_flux_bc")),
      specificHeat_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "specific_heat")),
      wallFricVel_(&meta_->declare_field<double>(
        meta_->side_rank(), "wall_friction_velocity_bip")),
      wallNormDist_(&meta_->declare_field<double>(
        meta_->side_rank(), "wall_normal_distance_bip")),
      tGradBC_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "temperature_gradient_bc")),
      ustar_(kappa_ * uh_ / std::log(zh_ / z0_))
  {

    stk::mesh::put_field_on_mesh(
      *wallVelocityBC_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *wallVelocityBC_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *bcHeatFlux_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *specificHeat_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *wallFricVel_, meta_->universal_part(), 4, nullptr);
    stk::mesh::put_field_on_mesh(
      *wallNormDist_, meta_->universal_part(), 4, nullptr);
    stk::mesh::put_field_on_mesh(*tGradBC_, meta_->universal_part(), nullptr);
  }

  virtual ~MomentumABLKernelHex8Mesh() = default;

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    const double vel[3] = {uh_, 0.0, 0.0};
    const double bcVel[3] = {0.0, 0.0, 0.0};
    MomentumKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill_component(vel, *velocity_);
    velocity_->modify_on_host();
    velocity_->sync_to_device();

    stk::mesh::field_fill_component(bcVel, *wallVelocityBC_);
    wallVelocityBC_->modify_on_host();
    wallVelocityBC_->sync_to_device();

    stk::mesh::field_fill(0.0, *bcHeatFlux_);
    bcHeatFlux_->modify_on_host();
    bcHeatFlux_->sync_to_device();

    stk::mesh::field_fill(1000.0, *specificHeat_);
    specificHeat_->modify_on_host();
    specificHeat_->sync_to_device();

    stk::mesh::field_fill(ustar_, *wallFricVel_);
    wallFricVel_->modify_on_host();
    wallFricVel_->sync_to_device();

    stk::mesh::field_fill(zh_, *wallNormDist_);
    wallNormDist_->modify_on_host();
    wallNormDist_->sync_to_device();

    stk::mesh::field_fill(-0.003, *tGradBC_);
    tGradBC_->modify_on_host();
    tGradBC_->sync_to_device();
  }

  sierra::nalu::VectorFieldType* wallVelocityBC_{nullptr};
  sierra::nalu::ScalarFieldType* bcHeatFlux_{nullptr};
  sierra::nalu::ScalarFieldType* specificHeat_{nullptr};
  sierra::nalu::ScalarFieldType* wallFricVel_{nullptr};
  sierra::nalu::ScalarFieldType* wallNormDist_{nullptr};
  sierra::nalu::ScalarFieldType* tGradBC_{nullptr};

  const double z0_{0.1};
  const double zh_{0.25};
  const double uh_{0.15};
  const double kappa_{0.41};
  const double gravity_{9.81};
  const double Tref_{300.0};
  const double ustar_;
};

class MomentumNodeHex8Mesh : public MomentumKernelHex8Mesh
{
};

class EnthalpyABLKernelHex8Mesh : public MomentumABLKernelHex8Mesh
{
public:
  EnthalpyABLKernelHex8Mesh()
    : MomentumABLKernelHex8Mesh(),
      thermalCond_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "thermal_conductivity")),
      evisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "effective_viscosity")),
      tvisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_viscosity")),
      heatFluxBC_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "heat_flux_bc"))
  {
    stk::mesh::put_field_on_mesh(
      *thermalCond_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*evisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tvisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *heatFluxBC_, meta_->universal_part(), nullptr);
  }

  virtual ~EnthalpyABLKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    MomentumABLKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(0.0, *thermalCond_);
    stk::mesh::field_fill(0.0, *evisc_);
    stk::mesh::field_fill(0.3, *tvisc_);
    stk::mesh::field_fill(100.0, *heatFluxBC_);
  }

  sierra::nalu::ScalarFieldType* thermalCond_{nullptr};
  sierra::nalu::ScalarFieldType* evisc_{nullptr};
  sierra::nalu::ScalarFieldType* tvisc_{nullptr};
  sierra::nalu::ScalarFieldType* heatFluxBC_{nullptr};
};

/** Test Fixture for the SST Kernels
 *
 */
class SSTKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  SSTKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      tke_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      tkebc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "bc_turbulent_ke")),
      sdr_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "specific_dissipation_rate")),
      sdrbc_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "sdr_bc")),
      visc_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      tvisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_viscosity")),
      maxLengthScale_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "sst_max_length_scale")),
      minDistance_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "minimum_distance_to_wall")),
      fOneBlend_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "sst_f_one_blending")),
      iddes_rans_indicator_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "iddes_rans_indicator")),
      dudx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx")),
      dkdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dkdx")),
      dwdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dwdx")),
      openMassFlowRate_(&meta_->declare_field<double>(
        meta_->side_rank(), "open_mass_flow_rate")),
      sdrWallbc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "wall_model_sdr_bc")),
      sdrWallArea_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "assembled_wall_area_sdr")),
      wallFricVel_(&meta_->declare_field<double>(
        meta_->side_rank(), "wall_friction_velocity_bip")),
      pecletFactor_(&meta_->declare_field<double>(
        stk::topology::EDGE_RANK, "peclet_factor"))
  {
    stk::mesh::put_field_on_mesh(*tke_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tkebc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*sdr_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*sdrbc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*visc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tvisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *maxLengthScale_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *minDistance_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*fOneBlend_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *iddes_rans_indicator_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *dudx_, stk::io::FieldOutputType::FULL_TENSOR_36);
    stk::mesh::put_field_on_mesh(
      *dkdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dkdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *dwdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dwdx_, stk::io::FieldOutputType::VECTOR_3D);
    double initOpenMassFlowRate[sierra::nalu::AlgTraitsQuad4::numScsIp_];
    for (int i = 0; i < sierra::nalu::AlgTraitsQuad4::numScsIp_; ++i) {
      initOpenMassFlowRate[i] = 10.0;
    }
    stk::mesh::put_field_on_mesh(
      *openMassFlowRate_, meta_->universal_part(),
      sierra::nalu::AlgTraitsQuad4::numScsIp_, initOpenMassFlowRate);

    stk::mesh::put_field_on_mesh(*sdrWallbc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *sdrWallArea_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *wallFricVel_, meta_->universal_part(), 4, nullptr);
    stk::mesh::put_field_on_mesh(
      *pecletFactor_, meta_->universal_part(), nullptr);
  }

  virtual ~SSTKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(0.2, *visc_);
    stk::mesh::field_fill(0.3, *tvisc_);
    stk::mesh::field_fill(0.5, *maxLengthScale_);
    unit_test_kernel_utils::density_test_function(
      *bulk_, *coordinates_, *density_);
    unit_test_kernel_utils::tke_test_function(*bulk_, *coordinates_, *tke_);
    unit_test_kernel_utils::sdr_test_function(*bulk_, *coordinates_, *sdr_);
    unit_test_kernel_utils::minimum_distance_to_wall_test_function(
      *bulk_, *coordinates_, *minDistance_);
    unit_test_kernel_utils::sst_f_one_blending_test_function(
      *bulk_, *coordinates_, *fOneBlend_);
    unit_test_kernel_utils::iddes_rans_indicator_test_function(
      *bulk_, *coordinates_, *iddes_rans_indicator_);
    unit_test_kernel_utils::dudx_test_function(*bulk_, *coordinates_, *dudx_);
    stk::mesh::field_fill(0.0, *dkdx_);
    stk::mesh::field_fill(0.0, *dwdx_);
    stk::mesh::field_fill(0.0, *pecletFactor_);
  }

  sierra::nalu::ScalarFieldType* tke_{nullptr};
  sierra::nalu::ScalarFieldType* tkebc_{nullptr};
  sierra::nalu::ScalarFieldType* sdr_{nullptr};
  sierra::nalu::ScalarFieldType* sdrbc_{nullptr};
  sierra::nalu::ScalarFieldType* visc_{nullptr};
  sierra::nalu::ScalarFieldType* tvisc_{nullptr};
  sierra::nalu::ScalarFieldType* maxLengthScale_{nullptr};
  sierra::nalu::ScalarFieldType* minDistance_{nullptr};
  sierra::nalu::ScalarFieldType* fOneBlend_{nullptr};
  sierra::nalu::ScalarFieldType* iddes_rans_indicator_{nullptr};
  sierra::nalu::TensorFieldType* dudx_{nullptr};
  sierra::nalu::VectorFieldType* dkdx_{nullptr};
  sierra::nalu::VectorFieldType* dwdx_{nullptr};
  sierra::nalu::GenericFieldType* openMassFlowRate_{nullptr};

  sierra::nalu::ScalarFieldType* sdrWallbc_{nullptr};
  sierra::nalu::ScalarFieldType* sdrWallArea_{nullptr};
  sierra::nalu::GenericFieldType* wallFricVel_{nullptr};
  sierra::nalu::ScalarFieldType* pecletFactor_{nullptr};
};

/** Test Fixture for the KE Kernels
 *
 */
class KEKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  KEKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      tke_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      tdr_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "total_dissipation_rate")),
      visc_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      tvisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_viscosity")),
      minDistance_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "minimum_distance_to_wall")),
      dplus_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "dplus_wall_function")),
      dudx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx"))
  {
    stk::mesh::put_field_on_mesh(*tke_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tdr_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*visc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tvisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *minDistance_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*dplus_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *dudx_, stk::io::FieldOutputType::FULL_TENSOR_36);
  }

  virtual ~KEKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(0.2, *visc_);
    stk::mesh::field_fill(0.3, *tvisc_);
    unit_test_kernel_utils::density_test_function(
      *bulk_, *coordinates_, *density_);
    unit_test_kernel_utils::tke_test_function(*bulk_, *coordinates_, *tke_);
    unit_test_kernel_utils::tdr_test_function(*bulk_, *coordinates_, *tdr_);
    unit_test_kernel_utils::minimum_distance_to_wall_test_function(
      *bulk_, *coordinates_, *minDistance_);
    unit_test_kernel_utils::dplus_test_function(*bulk_, *coordinates_, *dplus_);
    unit_test_kernel_utils::dudx_test_function(*bulk_, *coordinates_, *dudx_);
  }

  sierra::nalu::ScalarFieldType* tke_{nullptr};
  sierra::nalu::ScalarFieldType* tdr_{nullptr};
  sierra::nalu::ScalarFieldType* visc_{nullptr};
  sierra::nalu::ScalarFieldType* tvisc_{nullptr};
  sierra::nalu::ScalarFieldType* minDistance_{nullptr};
  sierra::nalu::ScalarFieldType* dplus_{nullptr};
  sierra::nalu::TensorFieldType* dudx_{nullptr};
};

/** Test Fixture for the KO Kernels
 *
 */
class KOKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  KOKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      tke_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      sdr_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "specific_dissipation_rate")),
      visc_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      tvisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_viscosity")),
      minDistance_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "minimum_distance_to_wall")),
      dudx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx")),
      dkdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dkdx")),
      dwdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dwdx"))
  {
    stk::mesh::put_field_on_mesh(*tke_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*sdr_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*visc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tvisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *minDistance_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *dudx_, stk::io::FieldOutputType::FULL_TENSOR_36);
    stk::mesh::put_field_on_mesh(
      *dkdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dkdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *dwdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dwdx_, stk::io::FieldOutputType::VECTOR_3D);
  }

  virtual ~KOKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(0.2, *visc_);
    stk::mesh::field_fill(0.3, *tvisc_);
    unit_test_kernel_utils::density_test_function(
      *bulk_, *coordinates_, *density_);
    unit_test_kernel_utils::tke_test_function(*bulk_, *coordinates_, *tke_);
    unit_test_kernel_utils::sdr_test_function(*bulk_, *coordinates_, *sdr_);
    unit_test_kernel_utils::minimum_distance_to_wall_test_function(
      *bulk_, *coordinates_, *minDistance_);
    unit_test_kernel_utils::dudx_test_function(*bulk_, *coordinates_, *dudx_);
    stk::mesh::field_fill(0.0, *dkdx_);
    stk::mesh::field_fill(0.0, *dwdx_);
  }

  sierra::nalu::ScalarFieldType* tke_{nullptr};
  sierra::nalu::ScalarFieldType* sdr_{nullptr};
  sierra::nalu::ScalarFieldType* visc_{nullptr};
  sierra::nalu::ScalarFieldType* tvisc_{nullptr};
  sierra::nalu::ScalarFieldType* minDistance_{nullptr};
  sierra::nalu::TensorFieldType* dudx_{nullptr};
  sierra::nalu::VectorFieldType* dkdx_{nullptr};
  sierra::nalu::VectorFieldType* dwdx_{nullptr};
};

/** Test Fixture for the Turbulence Kernels
 *
 */
class KsgsKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  KsgsKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      viscosity_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      tke_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      sdr_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "specific_dissipation_rate")),
      minDistance_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "minimum_distance_to_wall")),
      dudx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx")),
      tvisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_viscosity")),
      maxLengthScale_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "sst_max_length_scale")),
      fOneBlend_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "sst_f_one_blending")),
      evisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "effective_viscosity")),
      dualNodalVolume_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "dual_nodal_volume", 3)),
      dkdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dkdx")),
      dwdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dwdx")),
      dhdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dhdx")),
      specificHeat_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "specific_heat")),
      wallNormDistBip_(&meta_->declare_field<double>(
        stk::topology::FACE_RANK, "wall_normal_distance_bip")),
      wallArea_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "assembled_wall_area_wf")),
      wallNormDist_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "assembled_wall_normal_distance"))
  {
    stk::mesh::put_field_on_mesh(*viscosity_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tke_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*sdr_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *minDistance_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *dudx_, stk::io::FieldOutputType::FULL_TENSOR_36);
    stk::mesh::put_field_on_mesh(*tvisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *maxLengthScale_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*fOneBlend_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*evisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dualNodalVolume_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dkdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dkdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *dwdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dwdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *dhdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dhdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *specificHeat_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *wallNormDistBip_, meta_->universal_part(),
      sierra::nalu::AlgTraitsQuad4::numFaceIp_, nullptr);
    stk::mesh::put_field_on_mesh(*wallArea_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *wallNormDist_, meta_->universal_part(), nullptr);
  }

  virtual ~KsgsKernelHex8Mesh() = default;

  using LowMachKernelHex8Mesh::fill_mesh_and_init_fields;
  virtual void fill_mesh_and_init_fields(
    bool doPerturb = false,
    bool generateSidesets = false,
    const bool perturb_turbulent_viscosity_and_dual_nodal_volume = false)
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    if (perturb_turbulent_viscosity_and_dual_nodal_volume) {
      unit_test_kernel_utils::turbulent_viscosity_test_function(
        *bulk_, *coordinates_, *tvisc_);
      stk::mesh::field_fill(0.2, *dualNodalVolume_);
    } else {
      stk::mesh::field_fill(0.3, *tvisc_);
    }
    unit_test_kernel_utils::density_test_function(
      *bulk_, *coordinates_, *density_);
    stk::mesh::field_fill(0.2, *viscosity_);
    unit_test_kernel_utils::tke_test_function(*bulk_, *coordinates_, *tke_);
    unit_test_kernel_utils::sdr_test_function(*bulk_, *coordinates_, *sdr_);
    unit_test_kernel_utils::minimum_distance_to_wall_test_function(
      *bulk_, *coordinates_, *minDistance_);
    unit_test_kernel_utils::dudx_test_function(*bulk_, *coordinates_, *dudx_);
    stk::mesh::field_fill(0.5, *maxLengthScale_);
    unit_test_kernel_utils::sst_f_one_blending_test_function(
      *bulk_, *coordinates_, *fOneBlend_);
    stk::mesh::field_fill(0.0, *evisc_);
    unit_test_kernel_utils::dkdx_test_function(*bulk_, *coordinates_, *dkdx_);
    unit_test_kernel_utils::dwdx_test_function(*bulk_, *coordinates_, *dwdx_);
    unit_test_kernel_utils::dhdx_test_function(*bulk_, *coordinates_, *dhdx_);
    stk::mesh::field_fill(1000.0, *specificHeat_);
  }

  sierra::nalu::ScalarFieldType* viscosity_{nullptr};
  sierra::nalu::ScalarFieldType* tke_{nullptr};
  sierra::nalu::ScalarFieldType* sdr_{nullptr};
  sierra::nalu::ScalarFieldType* minDistance_{nullptr};
  sierra::nalu::TensorFieldType* dudx_{nullptr};
  sierra::nalu::ScalarFieldType* tvisc_{nullptr};
  sierra::nalu::ScalarFieldType* maxLengthScale_{nullptr};
  sierra::nalu::ScalarFieldType* fOneBlend_{nullptr};
  sierra::nalu::ScalarFieldType* evisc_{nullptr};
  sierra::nalu::ScalarFieldType* dualNodalVolume_{nullptr};
  sierra::nalu::VectorFieldType* dkdx_{nullptr};
  sierra::nalu::VectorFieldType* dwdx_{nullptr};
  sierra::nalu::VectorFieldType* dhdx_{nullptr};
  sierra::nalu::ScalarFieldType* specificHeat_{nullptr};
  sierra::nalu::GenericFieldType* wallNormDistBip_{nullptr};
  sierra::nalu::ScalarFieldType* wallArea_{nullptr};
  sierra::nalu::ScalarFieldType* wallNormDist_{nullptr};
};

/** Test Fixture for the AMS Kernels
 *
 */
class AMSKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  AMSKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      tke_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      sdr_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "specific_dissipation_rate")),
      visc_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      tvisc_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_viscosity")),
      alpha_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "k_ratio")),
      avgVelocity_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "average_velocity")),
      avgResAdeq_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "avg_res_adequacy_parameter")),
      avgProd_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "average_production")),
      avgTime_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "rans_time_scale")),
      minDist_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "minimum_distance_to_wall")),
      Mij_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "metric_tensor")),
      fOneBlend_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "sst_f_one_blending")),
      dudx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx")),
      avgDudx_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "average_dudx")),
      dkdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dkdx")),
      dwdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dwdx")),
      forcingComp_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "forcing_components"))
  {
    stk::mesh::put_field_on_mesh(*tke_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*sdr_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*visc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tvisc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*alpha_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *avgVelocity_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *avgVelocity_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *avgResAdeq_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*avgProd_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*avgTime_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*minDist_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *Mij_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*fOneBlend_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *dudx_, stk::io::FieldOutputType::FULL_TENSOR_36);
    stk::mesh::put_field_on_mesh(
      *avgDudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *avgDudx_, stk::io::FieldOutputType::FULL_TENSOR_36);
    stk::mesh::put_field_on_mesh(
      *dkdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dkdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *dwdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(*dwdx_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *forcingComp_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *forcingComp_, stk::io::FieldOutputType::VECTOR_3D);
  }

  virtual ~AMSKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(1.e-4, *visc_);
    stk::mesh::field_fill(0.3, *tvisc_);
    stk::mesh::field_fill(0.5, *avgVelocity_);
    stk::mesh::field_fill(1.0, *density_);
    stk::mesh::field_fill(0.7, *avgResAdeq_);
    stk::mesh::field_fill(0.6, *avgProd_);
    stk::mesh::field_fill(1.0, *avgTime_);
    stk::mesh::field_fill(0.7, *minDist_);
    stk::mesh::field_fill(0.2, *Mij_);
    unit_test_kernel_utils::tke_test_function(*bulk_, *coordinates_, *tke_);
    unit_test_kernel_utils::sdr_test_function(*bulk_, *coordinates_, *sdr_);
    unit_test_kernel_utils::alpha_test_function(*bulk_, *coordinates_, *alpha_);
    unit_test_kernel_utils::sst_f_one_blending_test_function(
      *bulk_, *coordinates_, *fOneBlend_);
    unit_test_kernel_utils::dudx_test_function(*bulk_, *coordinates_, *dudx_);
    unit_test_kernel_utils::dudx_test_function(
      *bulk_, *coordinates_, *avgDudx_);
    stk::mesh::field_fill(0.0, *dkdx_);
    stk::mesh::field_fill(0.0, *dwdx_);
    stk::mesh::field_fill(0.0, *forcingComp_);
  }

  sierra::nalu::ScalarFieldType* tke_{nullptr};
  sierra::nalu::ScalarFieldType* sdr_{nullptr};
  sierra::nalu::ScalarFieldType* visc_{nullptr};
  sierra::nalu::ScalarFieldType* tvisc_{nullptr};
  sierra::nalu::ScalarFieldType* alpha_{nullptr};
  sierra::nalu::VectorFieldType* avgVelocity_{nullptr};
  sierra::nalu::ScalarFieldType* avgResAdeq_{nullptr};
  sierra::nalu::ScalarFieldType* avgProd_{nullptr};
  sierra::nalu::ScalarFieldType* avgTime_{nullptr};
  sierra::nalu::ScalarFieldType* minDist_{nullptr};
  sierra::nalu::GenericFieldType* Mij_{nullptr};
  sierra::nalu::ScalarFieldType* fOneBlend_{nullptr};
  sierra::nalu::TensorFieldType* dudx_{nullptr};
  sierra::nalu::TensorFieldType* avgDudx_{nullptr};
  sierra::nalu::VectorFieldType* dkdx_{nullptr};
  sierra::nalu::VectorFieldType* dwdx_{nullptr};
  sierra::nalu::VectorFieldType* forcingComp_{nullptr};
};

/** Test Fixture for the hybrid turbulence Kernels
 *
 */
class HybridTurbKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  HybridTurbKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      tke_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "turbulent_ke")),
      alpha_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "adaptivity_parameter")),
      mutij_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "tensor_turbulent_viscosity"))
  {
    stk::mesh::put_field_on_mesh(*tke_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*alpha_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *mutij_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
  }

  virtual ~HybridTurbKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields(
      doPerturb, generateSidesets);
    stk::mesh::field_fill(0.0, *tke_);
    stk::mesh::field_fill(1.0, *alpha_);
    unit_test_kernel_utils::tensor_turbulent_viscosity_test_function(
      *bulk_, *coordinates_, *mutij_);
    /* unit_test_kernel_utils::tke_test_function(bulk_, *coordinates_, *tke_);
     */
    /* unit_test_kernel_utils::alpha_test_function(bulk_, *coordinates_,
     * *alpha_); */
  }

  sierra::nalu::ScalarFieldType* tke_{nullptr};
  sierra::nalu::ScalarFieldType* alpha_{nullptr};
  sierra::nalu::GenericFieldType* mutij_{nullptr};
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
      temperature_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "temperature", 2)),
      thermalCond_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "thermal_conductivity", 2))
  {
    stk::mesh::put_field_on_mesh(
      *temperature_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *thermalCond_, meta_->universal_part(), nullptr);
  }

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    TestKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb, generateSidesets);

    unit_test_kernel_utils::temperature_test_function(
      *bulk_, *coordinates_, *temperature_);
    stk::mesh::field_fill(1.0, *thermalCond_);
  }

  sierra::nalu::ScalarFieldType* temperature_{nullptr};
  sierra::nalu::ScalarFieldType* thermalCond_{nullptr};
};

/** Text fixture for mixture fraction equation kernels
 *
 *  This test fixture performs the following actions:
 *    - Create a HEX8 mesh with one element
 *    - Declare all of the set of fields required (autonomous from
 * LowMach/Mom/Cont)
 *    - Initialize the fields with steady 3-D solution; properties of helium/air
 */
class MixtureFractionKernelHex8Mesh : public TestKernelHex8Mesh
{
public:
  MixtureFractionKernelHex8Mesh()
    : TestKernelHex8Mesh(),
      mixFraction_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "mixture_fraction", 2)),
      velocity_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "velocity")),
      density_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "density", 2)),
      viscosity_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      effectiveViscosity_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "effective_viscosity")),
      massFlowRate_(&meta_->declare_field<double>(
        stk::topology::ELEM_RANK, "mass_flow_rate_scs")),
      dzdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dzdx")),
      dpdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dpdx", 2)),
      openMassFlowRate_(&meta_->declare_field<double>(
        meta_->side_rank(), "open_mass_flow_rate")),
      massFlowRateEdge_(&meta_->declare_field<double>(
        stk::topology::EDGE_RANK, "mass_flow_rate")),
      znot_(1.0),
      amf_(2.0),
      lamSc_(0.9),
      trbSc_(1.1),
      rhoPrimary_(0.163),
      rhoSecondary_(1.18),
      viscPrimary_(1.967e-5),
      viscSecondary_(1.85e-5)
  {
    const auto& meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        stk::topology::HEX_8);
    stk::mesh::put_field_on_mesh(
      *mixFraction_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *velocity_, meta_->universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*density_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*viscosity_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *massFlowRate_, meta_->universal_part(), meSCS->num_integration_points(),
      nullptr);
    stk::mesh::put_field_on_mesh(
      *dzdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *dpdx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *openMassFlowRate_, meta_->universal_part(),
      sierra::nalu::AlgTraitsQuad4::numScsIp_, nullptr);
    stk::mesh::put_field_on_mesh(
      *massFlowRateEdge_, meta_->universal_part(), spatialDim_, nullptr);
  }
  virtual ~MixtureFractionKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    TestKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb, generateSidesets);

    unit_test_kernel_utils::mixture_fraction_test_function(
      *bulk_, *coordinates_, *mixFraction_, amf_, znot_);
    unit_test_kernel_utils::velocity_test_function(
      *bulk_, *coordinates_, *velocity_);
    unit_test_kernel_utils::
      inverse_property_from_mixture_fraction_test_function(
        *bulk_, *mixFraction_, *density_, rhoPrimary_, rhoSecondary_);
    unit_test_kernel_utils::property_from_mixture_fraction_test_function(
      *bulk_, *mixFraction_, *viscosity_, viscPrimary_, viscSecondary_);
    unit_test_kernel_utils::calc_mass_flow_rate_scs(
      *bulk_, stk::topology::HEX_8, *coordinates_, *density_, *velocity_,
      *massFlowRate_);
    unit_test_kernel_utils::calc_open_mass_flow_rate(
      *bulk_, stk::topology::QUAD_4, *coordinates_, *density_, *velocity_,
      *exposedAreaVec_, *openMassFlowRate_);
    unit_test_kernel_utils::calc_mass_flow_rate(
      *bulk_, *velocity_, *density_, *edgeAreaVec_, *massFlowRateEdge_);
  }

  sierra::nalu::ScalarFieldType* mixFraction_{nullptr};
  sierra::nalu::VectorFieldType* velocity_{nullptr};
  sierra::nalu::ScalarFieldType* density_{nullptr};
  sierra::nalu::ScalarFieldType* viscosity_{nullptr};
  sierra::nalu::ScalarFieldType* effectiveViscosity_{nullptr};
  sierra::nalu::GenericFieldType* massFlowRate_{nullptr};
  sierra::nalu::VectorFieldType* dzdx_{nullptr};
  sierra::nalu::VectorFieldType* dpdx_{nullptr};
  sierra::nalu::GenericFieldType* openMassFlowRate_{nullptr};
  sierra::nalu::ScalarFieldType* massFlowRateEdge_{nullptr};

  const double znot_;
  const double amf_;
  const double lamSc_;
  const double trbSc_;
  const double rhoPrimary_;
  const double rhoSecondary_;
  const double viscPrimary_;
  const double viscSecondary_;
};

/** Text fixture for volume of fluid equation kernels
 *
 *  This test fixture performs the following actions:
 *    - Create a HEX8 mesh with one element
 *    - Declare all of the set of fields required (autonomous from
 * LowMach/Mom/Cont)
 *    - Initialize the fields with 3-D solution; properties of water/air
 */
class VOFKernelHex8Mesh : public TestKernelHex8Mesh
{
public:
  VOFKernelHex8Mesh()
    : TestKernelHex8Mesh(),
      volumeOfFluid_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "volume_of_fluid", 2)),
      dvolumeOfFluidDx_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "volume_of_fluid_gradient")),
      velocity_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "velocity", 2)),
      density_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "density", 2)),
      viscosity_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      massFlowRateEdge_(&meta_->declare_field<double>(
        stk::topology::EDGE_RANK, "mass_flow_rate")),
      vofBalancedMassFlowRateEdge_(&meta_->declare_field<ScalarFieldType>(
        stk::topology::EDGE_RANK, "mass_vof_balanced_flow_rate")),
      znot_(1.0),
      amf_(2.0),
      rhoPrimary_(1000.0),
      rhoSecondary_(1.0),
      viscPrimary_(1.e-6),
      viscSecondary_(1.e-5)
  {
    const auto& meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        stk::topology::HEX_8);
    stk::mesh::put_field_on_mesh(
      *volumeOfFluid_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dvolumeOfFluidDx_, meta_->universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *velocity_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *velocity_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(*density_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*viscosity_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *massFlowRateEdge_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *vofBalancedMassFlowRateEdge_, meta_->universal_part(), 1, nullptr);
  }
  virtual ~VOFKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    TestKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb, generateSidesets);

    unit_test_kernel_utils::mixture_fraction_test_function(
      *bulk_, *coordinates_, *volumeOfFluid_, amf_, znot_);

    unit_test_kernel_utils::velocity_test_function(
      *bulk_, *coordinates_, *velocity_);

    unit_test_kernel_utils::property_from_mixture_fraction_test_function(
      *bulk_, *volumeOfFluid_, *density_, rhoPrimary_, rhoSecondary_);
    unit_test_kernel_utils::property_from_mixture_fraction_test_function(
      *bulk_, *volumeOfFluid_, *viscosity_, viscPrimary_, viscSecondary_);
    unit_test_kernel_utils::calc_mass_flow_rate(
      *bulk_, *velocity_, *density_, *edgeAreaVec_, *massFlowRateEdge_);
  }

  sierra::nalu::ScalarFieldType* volumeOfFluid_{nullptr};
  sierra::nalu::VectorFieldType* dvolumeOfFluidDx_{nullptr};
  sierra::nalu::VectorFieldType* velocity_{nullptr};
  sierra::nalu::ScalarFieldType* density_{nullptr};
  sierra::nalu::ScalarFieldType* viscosity_{nullptr};
  sierra::nalu::ScalarFieldType* massFlowRateEdge_{nullptr};
  sierra::nalu::ScalarFieldType* vofBalancedMassFlowRateEdge_{nullptr};
  const double znot_;
  const double amf_;
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
      actuator_source_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "actuator_source")),
      actuator_source_lhs_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "actuator_source_lhs"))
  {
    stk::mesh::put_field_on_mesh(
      *actuator_source_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *actuator_source_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *actuator_source_lhs_, meta_->universal_part(), spatialDim_, nullptr);
    stk::io::set_field_output_type(
      *actuator_source_lhs_, stk::io::FieldOutputType::VECTOR_3D);
  }

  virtual ~ActuatorSourceKernelHex8Mesh() {}

  virtual void fill_mesh_and_init_fields(
    const bool doPerturb = false, const bool generateSidesets = false) override
  {
    TestKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb, generateSidesets);

    std::vector<double> act_source(spatialDim_, 0.0);
    for (size_t j = 0; j < spatialDim_; j++)
      act_source[j] = j + 1;
    stk::mesh::field_fill_component(act_source.data(), *actuator_source_);

    std::vector<double> act_source_lhs(spatialDim_, 0.0);
    for (size_t j = 0; j < spatialDim_; j++)
      act_source_lhs[j] = 0.1 * (j + 1);
    stk::mesh::field_fill_component(
      act_source_lhs.data(), *actuator_source_lhs_);
  }

  sierra::nalu::VectorFieldType* actuator_source_{nullptr};
  sierra::nalu::VectorFieldType* actuator_source_lhs_{nullptr};
};

class WallDistKernelHex8Mesh : public TestKernelHex8Mesh
{
};

#endif /* UNITTESTKERNELUTILS_H */
