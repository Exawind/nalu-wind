#include <gtest/gtest.h>
#include <limits>

#include <ComputeGeometryAlgorithmDriver.h>
#include "ComputeGeometryInteriorAlgorithm.h"
#include "ComputeGeometryBoundaryAlgorithm.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/ComputeVectorDivergence.h"

#include <stk_mesh/base/FieldParallel.hpp>

#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#include <string>

namespace {
  const std::vector<double> vecCoeff {7.0, 2.5, -3.0};

  const double coeffSum = vecCoeff[0]+vecCoeff[1]+vecCoeff[2];

  const double testTol = 1e-12;
}

TEST(utils, compute_vector_divergence)
{
  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();

  // declare relevant fields
  int nDim = realm.meta_data().spatial_dimension();

  ScalarFieldType *duaNdlVol = &(realm.meta_data().declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume"));
  stk::mesh::put_field_on_mesh(*duaNdlVol, realm.meta_data().universal_part(), nullptr);

  VectorFieldType *meshVec = &(realm.meta_data().declare_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_vector"));
  stk::mesh::put_field_on_mesh(*meshVec, realm.meta_data().universal_part(), nDim, nullptr);

  const sierra::nalu::MasterElement* meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(stk::topology::QUAD_4);
  const int numScsIp = meFC->numIntPoints_;
  GenericFieldType *exposedAreaVec = &(realm.meta_data().declare_field<GenericFieldType>(realm.meta_data().side_rank(), "exposed_area_vector"));
  stk::mesh::put_field_on_mesh(*exposedAreaVec, realm.meta_data().universal_part(), nDim*numScsIp , nullptr);

  ScalarFieldType *divV = &(realm.meta_data().declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_vector"));
  stk::mesh::put_field_on_mesh(*divV, realm.meta_data().universal_part(), nullptr);

  // create mesh
  const std::string meshSpec("generated:4x4x4");
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());

  // create dual volumes
  sierra::nalu::ComputeGeometryInteriorAlgorithm geomAlg(realm, &(realm.meta_data().universal_part()));
  geomAlg.execute();
  stk::mesh::parallel_sum(realm.bulk_data(), {duaNdlVol});

  sierra::nalu::ComputeGeometryBoundaryAlgorithm bndyGeomAlg(realm, realm.meta_data().get_part("surface_1"));
  bndyGeomAlg.execute();

  // get coordinate field
  VectorFieldType* modelCoords = realm.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  // get the universal part
  stk::mesh::Selector sel = stk::mesh::Selector(realm.meta_data().universal_part())
                          & (realm.meta_data().locally_owned_part() | realm.meta_data().globally_shared_part());
  const auto& bkts = realm.bulk_data().get_buckets(stk::topology::NODE_RANK, sel);

  // fill mesh velocity vector
  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node

      double* vecField = stk::mesh::field_data(*meshVec, node);
      double* xyz = stk::mesh::field_data( *modelCoords, node);

      for( int d = 0; d < nDim; d++)
        vecField[d] = vecCoeff[d]*xyz[d];

    } // end for loop - in index
  } // end for loop - bkts

  // compute divergence
  stk::mesh::PartVector partVec;
  partVec.push_back( &(realm.meta_data().universal_part()) );

  stk::mesh::PartVector bndyPartVec;
  bndyPartVec.push_back( realm.meta_data().get_part("surface_1") );

  sierra::nalu::compute_vector_divergence( realm.bulk_data(),
                                           partVec, bndyPartVec,
                                           meshVec, divV );

  // check values
  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node

      double* divVal = stk::mesh::field_data(*divV, node);

      EXPECT_NEAR(divVal[0], coeffSum, testTol);

    } // end for loop - in index
  } // end for loop - bkts
}
