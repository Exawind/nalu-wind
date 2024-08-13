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
#include <variant>
#include <type_traits>

void
TestAlgorithm::fill_mesh(const std::string mesh_spec)
{
  this->declare_fields();

  unit_test_utils::fill_hex8_mesh(mesh_spec, bulk());
  meshPart_ = meta().get_part("block_1");
  coordinates_ = static_cast<const sierra::nalu::VectorFieldType*>(
    meta().coordinate_field());
  EXPECT_TRUE(coordinates_ != nullptr);
}

double
TestAlgorithm::field_norm(
  const sierra::nalu::ScalarFieldType& field, stk::mesh::Selector* selector)
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
    timeIntegrator.secondOrderTimeAccurate_ = false;
    realm_->timeIntegrator_ = &timeIntegrator;
    realm_->setup_field_manager();
    realm_->timeIntegrator_ = nullptr;
  }
  const int numStates = 1;

  // clang-format off
  const std::vector<std::pair<std::string, stk::mesh::Field<double>**>> Fields = {
    {"density",                   &density_          }, 
    {"viscosity",                 &viscosity_        }, 
    {"turbulent_ke",              &tke_              }, 
    {"specific_dissipation_rate", &sdr_              }, 
    {"minimum_distance_to_wall",  &minDistance_      }, 
    {"turbulent_viscosity",       &tvisc_            }, 
    {"sst_max_length_scale",      &maxLengthScale_   }, 
    {"sst_f_one_blending",        &fOneBlend_        }, 
    {"effective_viscosity",       &evisc_            }, 
    {"dual_nodal_volume",         &dualNodalVolume_  }, 
    {"specific_heat",             &specificHeat_     }, 
    {"rans_time_scale",           &avgTime_          }, 
    {"open_tke_bc",               &tkebc_            }, 
    {"dkdx",                      &dkdx_             }, 
    {"dwdx",                      &dwdx_             }, 
    {"dhdx",                      &dhdx_             }, 
    {"dudx",                      &dudx_             }, 
    {"average_dudx",              &avgDudx_          }, 
    {"open_mass_flow_rate",       &openMassFlowRate_ }  
  };
  // clang-format on
  for (auto& Field : Fields) {
    const std::string& name = Field.first;
    const stk::mesh::PartVector universal(1, &meta.universal_part());
    using to_field = typename std::remove_pointer<decltype(Field.second)>::type;
    sierra::nalu::FieldPointerTypes new_field =
      realm_->fieldManager_->register_field(name, universal, numStates);
    std::visit(
      [&](auto fld) {
        using from_field = decltype(fld);
        if constexpr (std::is_same_v<to_field, from_field>) {
          *Field.second = fld;
        }
      },
      new_field);
  }
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
