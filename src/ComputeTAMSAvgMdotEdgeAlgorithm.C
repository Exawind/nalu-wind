/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <ComputeTAMSAvgMdotEdgeAlgorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// basic c++
#include <cmath>

namespace sierra {
namespace nalu {

ComputeTAMSAvgMdotEdgeAlgorithm::ComputeTAMSAvgMdotEdgeAlgorithm(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    meshMotion_(realm_.does_mesh_move()),
    avgTime_(NULL),
    massFlowRate_(NULL),
    avgMassFlowRate_(NULL)
{
  // save off field
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  avgTime_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_time");
  massFlowRate_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::EDGE_RANK, "mass_flow_rate");
  avgMassFlowRate_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::EDGE_RANK, "average_mass_flow_rate");
}

void
ComputeTAMSAvgMdotEdgeAlgorithm::execute()
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // time step
  const double dt = realm_.get_time_step();

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part() &
                                              stk::mesh::selectUnion(partVec_) &
                                              !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& edge_buckets =
    realm_.get_buckets(stk::topology::EDGE_RANK, s_locally_owned_union);
  for (stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
       ib != edge_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double * mdot = stk::mesh::field_data(*massFlowRate_, b);
    double* avgMdot = stk::mesh::field_data(*avgMassFlowRate_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      stk::mesh::Entity const* edge_node_rels = b.begin_nodes(k);

      // sanity check on number or nodes
      ThrowAssert(b.num_nodes(k) == 2);

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      // extract nodal fields
      const double avgTimeL = *stk::mesh::field_data(*avgTime_, nodeL);
      const double avgTimeR = *stk::mesh::field_data(*avgTime_, nodeR);

      const double avgTimeIp = 0.5 * (avgTimeR + avgTimeL);

      const double weightAvg = std::max(1.0 - dt / avgTimeIp, 0.0);
      const double weightInst = std::min(dt / avgTimeIp, 1.0);

      avgMdot[k] = weightAvg * avgMdot[k] + weightInst * mdot[k];
    }
  }
}

} // namespace nalu
} // namespace sierra
