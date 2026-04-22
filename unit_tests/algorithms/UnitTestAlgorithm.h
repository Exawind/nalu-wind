// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNITTESTALGORITHM_H
#define UNITTESTALGORITHM_H

#include <gtest/gtest.h>
#include "UnitTestRealm.h"
#include "UnitTestUtils.h"
#include "UnitTestFieldUtils.h"

#include <memory>
#include <cassert>

class TestAlgorithm : public ::testing::Test
{
public:
  TestAlgorithm() : comm_(MPI_COMM_WORLD)
  {
    YAML::Node doc = unit_test_utils::get_default_inputs();
    kynema_ugfObj_.reset(new unit_test_utils::KynemaUGFTest(doc));
  }

  virtual ~TestAlgorithm() {}

  inline sierra::kynema_ugf::Realm& create_realm(
    const YAML::Node& realm_node,
    const std::string realm_type = "multi_physics")
  {
    realm_ = &kynema_ugfObj_->create_realm(realm_node, realm_type);
    return *realm_;
  }

  inline sierra::kynema_ugf::Realm&
  create_realm(const std::string realm_type = "multi_physics")
  {
    const YAML::Node realm_node = unit_test_utils::get_realm_default_node();
    realm_ = &kynema_ugfObj_->create_realm(realm_node, realm_type);
    return *realm_;
  }

  void fill_mesh(const std::string mesh_spec = "generated:10x10x10");

  virtual void declare_fields() = 0;

  inline sierra::kynema_ugf::Realm& realm() const
  {
    assert(realm_ != nullptr);
    return *realm_;
  }

  inline stk::mesh::MetaData& meta() const { return realm().meta_data(); }

  inline stk::mesh::BulkData& bulk() const { return realm().bulk_data(); }

  double field_norm(
    const sierra::kynema_ugf::ScalarFieldType& field,
    stk::mesh::Selector* selector = nullptr);

  //! Reference to test KynemaUGF instance used to hold Simulation and Realm
  std::unique_ptr<unit_test_utils::KynemaUGFTest> kynema_ugfObj_;

  //! Reference to realm instance
  sierra::kynema_ugf::Realm* realm_{nullptr};

  stk::mesh::Part* meshPart_{nullptr};
  const sierra::kynema_ugf::VectorFieldType* coordinates_{nullptr};
  stk::ParallelMachine comm_;
};

class TestTurbulenceAlgorithm : public TestAlgorithm
{
public:
  TestTurbulenceAlgorithm() : TestAlgorithm() {}

  virtual ~TestTurbulenceAlgorithm() {}

  virtual void declare_fields();

  virtual void
  fill_mesh_and_init_fields(const std::string mesh_spec = "generated:10x10x10");

  sierra::kynema_ugf::ScalarFieldType* density_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* viscosity_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* tke_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* sdr_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* minDistance_{nullptr};
  sierra::kynema_ugf::TensorFieldType* dudx_{nullptr};
  sierra::kynema_ugf::GenericFieldType* openMassFlowRate_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* tvisc_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* maxLengthScale_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* fOneBlend_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* evisc_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* dualNodalVolume_{nullptr};
  sierra::kynema_ugf::VectorFieldType* dkdx_{nullptr};
  sierra::kynema_ugf::VectorFieldType* dwdx_{nullptr};
  sierra::kynema_ugf::VectorFieldType* dhdx_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* specificHeat_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* tkebc_{nullptr};
  sierra::kynema_ugf::TensorFieldType* avgDudx_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* avgTime_{nullptr};
};

struct NodeSuppHelper
{
  NodeSuppHelper()
    : yamlNode(unit_test_utils::get_default_inputs()),
      realmDefaultNode(unit_test_utils::get_realm_default_node()),
      kynema_ugfObj(
        std::unique_ptr<unit_test_utils::KynemaUGFTest>(
          new unit_test_utils::KynemaUGFTest(yamlNode))),
      realm(kynema_ugfObj->create_realm(realmDefaultNode, "multi_physics"))
  {
  }

  stk::mesh::Entity make_one_node_mesh()
  {
    realm.bulk_data().modification_begin();
    node = realm.bulk_data().declare_node(1u);
    realm.bulk_data().modification_end();

    return node;
  }

  YAML::Node yamlNode;
  YAML::Node realmDefaultNode;
  std::unique_ptr<unit_test_utils::KynemaUGFTest> kynema_ugfObj;
  sierra::kynema_ugf::Realm& realm;

  stk::mesh::Entity node;
};

#endif /* UNITTESTALGORITHM_H */
