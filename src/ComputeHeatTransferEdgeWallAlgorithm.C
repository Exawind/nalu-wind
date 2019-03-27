/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeHeatTransferEdgeWallAlgorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeHeatTransferEdgeWallAlgorithm - compute h and Too
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeHeatTransferEdgeWallAlgorithm::ComputeHeatTransferEdgeWallAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    temperature_(NULL),
    dhdx_(NULL),
    coordinates_(NULL),
    density_(NULL),
    thermalCond_(NULL),
    specificHeat_(NULL),
    exposedAreaVec_(NULL),
    assembledWallArea_(NULL),
    referenceTemperature_(NULL),
    heatTransferCoefficient_(NULL),
    normalHeatFlux_(NULL),
    robinCouplingParameter_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  temperature_= meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature");
  dhdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dhdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  thermalCond_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "thermal_conductivity");
  specificHeat_= meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  assembledWallArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_ht");
  referenceTemperature_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "reference_temperature");
  heatTransferCoefficient_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_transfer_coefficient");
  normalHeatFlux_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "normal_heat_flux");
  robinCouplingParameter_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "robin_coupling_parameter");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeHeatTransferEdgeWallAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double dt = realm_.get_time_step();

  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract connected element topology
    b.parent_topology(stk::topology::ELEMENT_RANK, parentTopo);
    ThrowAssert ( parentTopo.size() == 1 );
    stk::topology theElemTopo = parentTopo[0];
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theElemTopo);

    // size some things that are useful
    const int num_face_nodes = b.topology().num_nodes();
    

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, b, k);

      // extract the connected element to this exposed face; should be single in size!
      stk::mesh::Entity const * face_elem_rels = b.begin_elements(k);
      ThrowAssert( b.num_elements(k) == 1 );

      // get element; its face ordinal number and populate face_node_ordinals
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = b.begin_element_ordinals(k)[0];
      const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

      // get the relations
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);

      for ( int ip = 0; ip < num_face_nodes; ++ip ) {

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
        const int nearestNode = face_node_ordinals[ip];

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

        // extract nodal fields
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

        const double tempL = *stk::mesh::field_data(*temperature_, nodeL );
        const double tempR = *stk::mesh::field_data(*temperature_, nodeR );

        // nearest nodes; gathered and to-be-scattered
        const double * dhdxR    =  stk::mesh::field_data(*dhdx_, nodeR );
        const double densityR   = *stk::mesh::field_data(*density_, nodeR );
        const double thermalCondR = *stk::mesh::field_data(*thermalCond_, nodeR );
        const double specificHeatR = *stk::mesh::field_data(*specificHeat_, nodeR );
        double *assembledWallArea = stk::mesh::field_data(*assembledWallArea_, nodeR);
        double *referenceTemperature = stk::mesh::field_data(*referenceTemperature_, nodeR);
        double *heatTransferCoefficient = stk::mesh::field_data(*heatTransferCoefficient_, nodeR);
        double *normalHeatFlux = stk::mesh::field_data(*normalHeatFlux_, nodeR);
        double *robinCouplingParameter = stk::mesh::field_data(*robinCouplingParameter_, nodeR);

        // offset for bip area vector
        const int faceOffSet = ip*nDim;

        // compute geometry
        double axdx = 0.0;
        double asq = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[faceOffSet+j];
          const double dxj = coordR[j] - coordL[j];
          asq += axj*axj;
          axdx += axj*dxj;
        }

        const double inv_axdx = 1.0/axdx;
        const double aMag = std::sqrt(asq);
        const double edgeLen = axdx/aMag;

        // NOC; convert dhdx to dTdx
        double nonOrth = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[faceOffSet+j];
          const double dxj = coordR[j] - coordL[j];
          const double kxj = axj - asq*inv_axdx*dxj;
          const double GjT = dhdxR[j]/specificHeatR;
          nonOrth += -thermalCondR*kxj*GjT;
        }

        // compute coupling parameter
        const double chi = densityR * specificHeatR * edgeLen * edgeLen
          / (2 * thermalCondR * dt);
        const double alpha = compute_coupling_parameter(thermalCondR, edgeLen, chi);

        // assemble the nodal quantities; group NOC on reference temp
        // if NOC is < 0; Too will be greater than Tphyscial
        // if NOC is > 0; Too will be less than Tphysical
        // grouping NOC on H reverses the above, however, who knows which is best..
        *assembledWallArea += aMag;
        *referenceTemperature += thermalCondR*tempL*asq*inv_axdx - nonOrth;
        *heatTransferCoefficient += -thermalCondR*tempR*asq*inv_axdx;
        *normalHeatFlux += thermalCondR*(tempL-tempR)*asq*inv_axdx - nonOrth;
        *robinCouplingParameter += alpha*aMag;
      }
    }
  }
}


//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeHeatTransferEdgeWallAlgorithm::~ComputeHeatTransferEdgeWallAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- compute_coupling_parameter --------------------------------------
//--------------------------------------------------------------------------
double
ComputeHeatTransferEdgeWallAlgorithm::compute_coupling_parameter(const double & kappa,
                                                                 const double & h,
                                                                 const double & chi)
{
  // This function approximates the ideal coupling parameter for Dirichlet-Robin coupling
  const double A = 1.0 + chi - 1.0/(1.0 + chi + std::sqrt(chi*(chi+2)));
  return A * kappa/h;
}


} // namespace nalu
} // namespace Sierra
