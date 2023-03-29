// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "UnitTestAlgorithm.h"
#include "UnitTestUtils.h"
#include "UnitTestKokkosUtils.h"
#include "kernels/UnitTestKernelUtils.h"

#include "Realm.h"

void
TestAlgorithm::fill_mesh(const std::string mesh_spec)
{
  this->declare_fields();

  unit_test_utils::fill_hex8_mesh(mesh_spec, bulk());
  meshPart_ = meta().get_part("block_1");
  coordinates_ = static_cast<const VectorFieldType*>(meta().coordinate_field());
  EXPECT_TRUE(coordinates_ != nullptr);
}

double
TestAlgorithm::field_norm(
  const ScalarFieldType& field, stk::mesh::Selector* selector)
{

  auto& meta = this->meta();
  auto& bulk = this->bulk();
  auto sel = (selector == nullptr) ? meta.locally_owned_part() : *selector;

  return unit_test_utils::field_norm(field, bulk, sel);
}

void
TestTurbulenceAlgorithm::declare_fields()
{
  auto& meta = this->meta();
  auto spatialDim = meta.spatial_dimension();

  if (!realm_->fieldManager_) {
    sierra::nalu::TimeIntegrator timeIntegrator;
    realm_->timeIntegrator_ = &timeIntegrator;
    realm_->setup_field_manager();
    realm_->timeIntegrator_ = nullptr;
  }
  const int numStates = 1;
  realm_->fieldManager_->register_field("density", meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("viscosity", meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("turbulent_ke", meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("specific_dissipation_rate", meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("minimum_distance_to_wall", meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("turbulent_viscosity" ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("sst_max_length_scale",       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("sst_f_one_blending"  ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("effective_viscosity" ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("dual_nodal_volume"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("specific_heat"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("rans_time_scale"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("open_tke_bc"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("dkdx"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("dwdx"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("dhdx"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("dudx"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("average_dudx"   ,       meta.universal_part(), numStates);
  realm_->fieldManager_->register_field("open_mass_flow_rate"   ,       meta.universal_part(), numStates);

  density_     = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("density");
  viscosity_   = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("viscosity");
  tke_         = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("turbulent_ke");
  sdr_         = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("specific_dissipation_rate");
  minDistance_ = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("minimum_distance_to_wall");
  tvisc_          = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("turbulent_viscosity" );
  maxLengthScale_ = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("sst_max_length_scale");
  fOneBlend_      = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("sst_f_one_blending"  );
  evisc_          = realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("effective_viscosity" );
  dualNodalVolume_= realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("dual_nodal_volume"   );
  specificHeat_= realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("specific_heat"   );
  avgTime_= realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("rans_time_scale"   );
  tkebc_= realm_->fieldManager_->get_field_ptr<ScalarFieldType*>("open_tke_bc"   );
  dkdx_= realm_->fieldManager_->get_field_ptr<VectorFieldType*>("dkdx");
  dwdx_= realm_->fieldManager_->get_field_ptr<VectorFieldType*>("dwdx");
  dhdx_= realm_->fieldManager_->get_field_ptr<VectorFieldType*>("dhdx");
  dudx_= realm_->fieldManager_->get_field_ptr<TensorFieldType*>("dudx");
  avgDudx_= realm_->fieldManager_->get_field_ptr<TensorFieldType*>("average_dudx"   );
  openMassFlowRate_ = realm_->fieldManager_->get_field_ptr<GenericFieldType*>("open_mass_flow_rate");


  stk::mesh::put_field_on_mesh(
    *openMassFlowRate_, meta.universal_part(),
    sierra::nalu::AlgTraitsQuad4::numScsIp_, nullptr);
}

void
TestTurbulenceAlgorithm::fill_mesh_and_init_fields(const std::string mesh_spec)
{
  fill_mesh(mesh_spec);

  auto& bulk = this->bulk();

  unit_test_kernel_utils::density_test_function(bulk, *coordinates_, *density_);
  stk::mesh::field_fill(0.2, *viscosity_);
  unit_test_kernel_utils::tke_test_function(bulk, *coordinates_, *tke_);
  unit_test_kernel_utils::sdr_test_function(bulk, *coordinates_, *sdr_);
  unit_test_kernel_utils::minimum_distance_to_wall_test_function(
    bulk, *coordinates_, *minDistance_);
  unit_test_kernel_utils::dudx_test_function(bulk, *coordinates_, *dudx_);
  unit_test_kernel_utils::turbulent_viscosity_test_function(
    bulk, *coordinates_, *tvisc_);
  stk::mesh::field_fill(0.5, *maxLengthScale_);
  unit_test_kernel_utils::sst_f_one_blending_test_function(
    bulk, *coordinates_, *fOneBlend_);
  stk::mesh::field_fill(0.0, *evisc_);
  stk::mesh::field_fill(0.2, *dualNodalVolume_);
  unit_test_kernel_utils::dkdx_test_function(bulk, *coordinates_, *dkdx_);
  unit_test_kernel_utils::dwdx_test_function(bulk, *coordinates_, *dwdx_);
  unit_test_kernel_utils::dhdx_test_function(bulk, *coordinates_, *dhdx_);
  stk::mesh::field_fill(1000.0, *specificHeat_);
  stk::mesh::field_fill(10.0, *openMassFlowRate_);
  unit_test_kernel_utils::dudx_test_function(bulk, *coordinates_, *avgDudx_);
  stk::mesh::field_fill(1.0, *avgTime_);
}
