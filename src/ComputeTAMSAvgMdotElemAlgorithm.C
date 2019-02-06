/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
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

//==========================================================================
// Class Definition
//==========================================================================
// ComputeTAMSAvgMdotElemAlgorithm - interior mdor for elem continuity
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSAvgMdotElemAlgorithm::ComputeTAMSAvgMdotElemAlgorithm(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    velocityRTM_(NULL),
    coordinates_(NULL),
    density_(NULL),
    avgTime_(NULL),
    massFlowRate_(NULL),
    avgMassFlowRate_(NULL),
    shiftTAMSAvgMdot_(realm_.get_cvfem_shifted_mdot())
{
  // extract fields; nodal
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  velocityRTM_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "average_velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_density");
  avgTime_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_time");
  massFlowRate_ = meta_data.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");
  avgMassFlowRate_ = meta_data.get_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "average_mass_flow_rate_scs");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeTAMSAvgMdotElemAlgorithm::~ComputeTAMSAvgMdotElemAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeTAMSAvgMdotElemAlgorithm::execute()
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  // time step
  const double dt = realm_.get_time_step();

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0 - interpTogether;

  // nodal fields to gather
  std::vector<double> ws_vrtm;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_density;
  std::vector<double> ws_avgTime;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_shape_function;

  // integration point data that depends on size
  std::vector<double> uIp(nDim);
  std::vector<double> rho_uIp(nDim);

  // pointers to everyone...
  double* p_uIp = &uIp[0];
  double* p_rho_uIp = &rho_uIp[0];

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
    ws_vrtm.resize(nodesPerElement * nDim);
    ws_coordinates.resize(nodesPerElement * nDim);
    ws_density.resize(nodesPerElement);
    ws_avgTime.resize(nodesPerElement);
    ws_scs_areav.resize(numScsIp * nDim);
    ws_shape_function.resize(numScsIp * nodesPerElement);

    // pointers
    double* p_vrtm = &ws_vrtm[0];
    double* p_coordinates = &ws_coordinates[0];
    double* p_density = &ws_density[0];
    double* p_avgTime = &ws_avgTime[0];
    double* p_scs_areav = &ws_scs_areav[0];
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

        // pointers to real data
        const double* vrtm = stk::mesh::field_data(*velocityRTM_, node);
        const double* coords = stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        p_density[ni] = *stk::mesh::field_data(*density_, node);
        p_avgTime[ni] = *stk::mesh::field_data(*avgTime_, node);

        // gather vectors
        const int offSet = ni * nDim;
        for (int j = 0; j < nDim; ++j) {
          p_vrtm[offSet + j] = vrtm[j];
          p_coordinates[offSet + j] = coords[j];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      for (int ip = 0; ip < numScsIp; ++ip) {

        // setup for ip values
        for (int j = 0; j < nDim; ++j) {
          p_uIp[j] = 0.0;
          p_rho_uIp[j] = 0.0;
        }
        double rhoIp = 0.0;
        double avgTimeIp = 0.0;

        const int offSet = ip * nodesPerElement;
        for (int ic = 0; ic < nodesPerElement; ++ic) {

          const double r = p_shape_function[offSet + ic];
          const double nodalRho = p_density[ic];
          const double nodalAvgTime = p_avgTime[ic];

          rhoIp += r * nodalRho;
          avgTimeIp += r * nodalAvgTime;

          for (int j = 0; j < nDim; ++j) {
            p_uIp[j] += r * p_vrtm[nDim * ic + j];
            p_rho_uIp[j] += r * nodalRho * p_vrtm[nDim * ic + j];
          }
        }

        // assemble mdot
        double tmdot = 0.0;
        for (int j = 0; j < nDim; ++j) {
          tmdot += (interpTogether * p_rho_uIp[j] +
                    om_interpTogether * rhoIp * p_uIp[j]) *
                   p_scs_areav[ip * nDim + j];
        }

        // avgMdot[ip] = tmdot;
        const double weightAvg = std::max(1.0 - dt / avgTimeIp, 0.0);
        const double weightInst = std::min(dt / avgTimeIp, 1.0);

        avgMdot[ip] = weightAvg * avgMdot[ip] + weightInst * mdot[ip];
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
