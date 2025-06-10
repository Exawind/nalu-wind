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
    naluObj_.reset(new unit_test_utils::NaluTest(doc));
  }

  virtual ~TestAlgorithm() {}

  inline sierra::nalu::Realm& create_realm(
    const YAML::Node& realm_node,
    const std::string realm_type = "multi_physics")
  {
    realm_ = &naluObj_->create_realm(realm_node, realm_type);
    return *realm_;
  }

  inline sierra::nalu::Realm&
  create_realm(const std::string realm_type = "multi_physics")
  {
    const YAML::Node realm_node = unit_test_utils::get_realm_default_node();
    realm_ = &naluObj_->create_realm(realm_node, realm_type);
    return *realm_;
  }

  void fill_mesh(const std::string mesh_spec = "generated:10x10x10");

  virtual void declare_fields() = 0;

  inline sierra::nalu::Realm& realm() const
  {
    assert(realm_ != nullptr);
    return *realm_;
  }

  inline stk::mesh::MetaData& meta() const { return realm().meta_data(); }

  inline stk::mesh::BulkData& bulk() const { return realm().bulk_data(); }

  double field_norm(
    const sierra::nalu::ScalarFieldType& field,
    stk::mesh::Selector* selector = nullptr);

  //! Reference to test Nalu instance used to hold Simulation and Realm
  std::unique_ptr<unit_test_utils::NaluTest> naluObj_;

  //! Reference to realm instance
  sierra::nalu::Realm* realm_{nullptr};

  stk::mesh::Part* meshPart_{nullptr};
  const sierra::nalu::VectorFieldType* coordinates_{nullptr};
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

  sierra::nalu::ScalarFieldType* density_{nullptr};
  sierra::nalu::ScalarFieldType* viscosity_{nullptr};
  sierra::nalu::ScalarFieldType* tke_{nullptr};
  sierra::nalu::ScalarFieldType* sdr_{nullptr};
  sierra::nalu::ScalarFieldType* minDistance_{nullptr};
  sierra::nalu::TensorFieldType* dudx_{nullptr};
  sierra::nalu::GenericFieldType* openMassFlowRate_{nullptr};
  sierra::nalu::ScalarFieldType* tvisc_{nullptr};
  sierra::nalu::ScalarFieldType* maxLengthScale_{nullptr};
  sierra::nalu::ScalarFieldType* fOneBlend_{nullptr};
  sierra::nalu::ScalarFieldType* evisc_{nullptr};
  sierra::nalu::ScalarFieldType* dualNodalVolume_{nullptr};
  sierra::nalu::VectorFieldType* dkdx_{nullptr};
  sierra::nalu::VectorFieldType* dwdx_{nullptr};
  sierra::nalu::VectorFieldType* dhdx_{nullptr};
  sierra::nalu::ScalarFieldType* specificHeat_{nullptr};
  sierra::nalu::ScalarFieldType* tkebc_{nullptr};
  sierra::nalu::TensorFieldType* avgDudx_{nullptr};
  sierra::nalu::ScalarFieldType* avgTime_{nullptr};
};

struct NodeSuppHelper
{
  NodeSuppHelper()
    : yamlNode(unit_test_utils::get_default_inputs()),
      realmDefaultNode(unit_test_utils::get_realm_default_node()),
      naluObj(
        std::unique_ptr<unit_test_utils::NaluTest>(
          new unit_test_utils::NaluTest(yamlNode))),
      realm(naluObj->create_realm(realmDefaultNode, "multi_physics"))
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
  std::unique_ptr<unit_test_utils::NaluTest> naluObj;
  sierra::nalu::Realm& realm;

  stk::mesh::Entity node;
};

#endif /* UNITTESTALGORITHM_H */
