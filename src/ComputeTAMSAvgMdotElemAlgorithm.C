/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <ComputeTAMSAvgMdotElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

ComputeTAMSAvgMdotElemAlgorithm::ComputeTAMSAvgMdotElemAlgorithm(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    avgTime_(NULL),
    massFlowRate_(NULL),
    avgMassFlowRate_(NULL),
    shiftTAMSAvgMdot_(realm_.get_cvfem_shifted_mdot())
{
  // extract fields; nodal
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  avgTime_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_time");
  massFlowRate_ = meta_data.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");
  avgMassFlowRate_ = meta_data.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "average_mass_flow_rate_scs");
}

void
ComputeTAMSAvgMdotElemAlgorithm::execute()
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  // time step
  const double dt = realm_.get_time_step();

  // nodal fields to gather
  std::vector<double> ws_avgTime;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_shape_function;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part() &
                                              stk::mesh::selectUnion(partVec_) &
                                              !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union);
  for (stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
       ib != elem_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // extract master element
    MasterElement* meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->num_integration_points();

    // algorithm related
    ws_avgTime.resize(nodesPerElement);
    ws_scs_areav.resize(numScsIp * nDim);
    ws_shape_function.resize(numScsIp * nodesPerElement);

    // pointers
    double* p_avgTime = &ws_avgTime[0];
    double* p_shape_function = &ws_shape_function[0];

    if (shiftTAMSAvgMdot_)
      meSCS->shifted_shape_fcn(&p_shape_function[0]);
    else
      meSCS->shape_fcn(&p_shape_function[0]);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      // pointers to elem data
      double* mdot = stk::mesh::field_data(*massFlowRate_, b, k);
      double* avgMdot = stk::mesh::field_data(*avgMassFlowRate_, b, k);

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const* node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert(num_nodes == nodesPerElement);

      for (int ni = 0; ni < num_nodes; ++ni) {
        stk::mesh::Entity node = node_rels[ni];

        // gather scalars
        p_avgTime[ni] = *stk::mesh::field_data(*avgTime_, node);

      }

      for (int ip = 0; ip < numScsIp; ++ip) {

        // setup for ip values
        double avgTimeIp = 0.0;

        const int offSet = ip * nodesPerElement;
        for (int ic = 0; ic < nodesPerElement; ++ic) {

          const double r = p_shape_function[offSet + ic];
          const double nodalAvgTime = p_avgTime[ic];
          avgTimeIp += r * nodalAvgTime;
        }

        const double weightAvg = std::max(1.0 - dt / avgTimeIp, 0.0);
        const double weightInst = std::min(dt / avgTimeIp, 1.0);

        avgMdot[ip] = weightAvg * avgMdot[ip] + weightInst * mdot[ip];
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
