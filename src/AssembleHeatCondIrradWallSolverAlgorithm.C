/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleHeatCondIrradWallSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleHeatCondIrradWallSolverAlgorithm - provides e(Ib-H) flux
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleHeatCondIrradWallSolverAlgorithm::AssembleHeatCondIrradWallSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  bool useShifted)
  : SolverAlgorithm(realm, part, eqSystem),
    useShifted_(useShifted),
    exposedAreaVec_(NULL),
    temperature_(NULL),
    irradiation_(NULL),
    emissivity_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  temperature_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature");
  irradiation_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "irradiation");
  emissivity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "emissivity");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleHeatCondIrradWallSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleHeatCondIrradWallSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double sigma = realm_.get_stefan_boltzmann();

  // space for LHS/RHS; nodesPerFace*nodesPerFace and nodesPerFace
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // nodal fields to gather
  std::vector<double> ws_irradiation;
  std::vector<double> ws_emissivity;
  std::vector<double> ws_temperature;

  // geometry related to populate
  std::vector<double> ws_shape_function;

  // setup for buckets; union parts and ask for locally owned
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);
  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract master element specifics
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsIp = meFC->num_integration_points();

    // resize some things; matrix related
    const int lhsSize = nodesPerFace*nodesPerFace;
    const int rhsSize = nodesPerFace;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerFace);

    // algorithm related
    ws_irradiation.resize(nodesPerFace);
    ws_emissivity.resize(nodesPerFace);
    ws_temperature.resize(nodesPerFace);
    ws_shape_function.resize(numScsIp*nodesPerFace);

    // pointers
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_irradiation = &ws_irradiation[0];
    double *p_emissivity = &ws_emissivity[0];
    double *p_temperature = &ws_temperature[0];
    double *p_shape_function = &ws_shape_function[0];

    if ( useShifted_ )
      meFC->shifted_shape_fcn(&p_shape_function[0]);
    else
      meFC->shape_fcn(&p_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        p_lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        p_rhs[p] = 0.0;

      // face data
      double * areaVec = stk::mesh::field_data(*exposedAreaVec_, b, k);

      // face node relations for nodal gather
      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);
      
      int num_nodes = b.num_nodes(k);
      for ( int ni = 0; ni < num_nodes; ++ni ) {

        // get the node and form connected_node
        stk::mesh::Entity node = face_node_rels[ni];
        connected_nodes[ni] = node;

        // gather scalar
        p_irradiation[ni] = *stk::mesh::field_data(*irradiation_, node);
        p_emissivity[ni] = *stk::mesh::field_data(*emissivity_, node);
        p_temperature[ni] = *stk::mesh::field_data(*temperature_, node);
      }

      // start the assembly
      for ( int ip = 0; ip < numScsIp; ++ip ) {
	
        double magA = 0.0;
        for ( int j=0; j < nDim; ++j ) {
          magA += areaVec[ip*nDim+j]*areaVec[ip*nDim+j];
        }
        magA = std::sqrt(magA);
	
        const int nn = ip;
        const int offSet = ip*nodesPerFace;
	
        // form boundary ip values
        double irradiationBip = 0.0;
        double emissivityBip = 0.0;
        double tBip = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_shape_function[offSet+ic];
          irradiationBip += r*p_irradiation[ic];
          emissivityBip += r*p_emissivity[ic];
          tBip += r*p_temperature[ic];
        }

        // form rhs contribution
        const double radiation = emissivityBip*(irradiationBip - sigma*std::pow(tBip,4))*magA;
        p_rhs[nn] += radiation;
	
        // sensitivities
        const int rowR = nn*nodesPerFace;
        const double lhsFac = 4.0*sigma*emissivityBip*magA*std::pow(tBip,3);
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_shape_function[offSet+ic];
          p_lhs[rowR+ic] += r*lhsFac;
        }
      }
      
      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
