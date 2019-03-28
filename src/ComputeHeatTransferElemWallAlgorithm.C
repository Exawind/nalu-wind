/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeHeatTransferElemWallAlgorithm.h>

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
// ComputeHeatTransferElemWallAlgorithm - compute h and Too
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeHeatTransferElemWallAlgorithm::ComputeHeatTransferElemWallAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    temperature_(NULL),
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
  temperature_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  thermalCond_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "thermal_conductivity");
  specificHeat_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat");
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
ComputeHeatTransferElemWallAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double dt = realm_.get_time_step();

  // nodal fields to gather
  std::vector<double> ws_coordinates;
  std::vector<double> ws_temperature;
  std::vector<double> ws_thermalCond;
  std::vector<double> ws_density;
  std::vector<double> ws_specificHeat;

  // master element
  std::vector<double> ws_face_shape_function;
  std::vector<double> ws_dndx;
  std::vector<double> ws_det_j;
  // array for face nodes and nodes off face
  std::vector<double> ws_nodesOnFace;
  std::vector<double> ws_nodesOffFace;

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
    const int nodesPerElement = meSCS->nodesPerElement_;

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // algorithm related; element
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_temperature.resize(nodesPerElement);
    ws_thermalCond.resize(nodesPerFace);
    ws_density.resize(nodesPerFace);
    ws_specificHeat.resize(nodesPerFace);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);
    ws_dndx.resize(nDim*numScsBip*nodesPerElement);
    ws_det_j.resize(numScsBip);
    ws_nodesOnFace.resize(nodesPerElement);
    ws_nodesOffFace.resize(nodesPerElement);

    // pointers
    double *p_coordinates = &ws_coordinates[0];
    double *p_temperature = &ws_temperature[0];
    double *p_thermalCond = &ws_thermalCond[0];
    double *p_density     = &ws_density[0];
    double *p_specificHeat = &ws_specificHeat[0];
    double *p_face_shape_function = &ws_face_shape_function[0];
    double *p_dndx = &ws_dndx[0];
    double *p_nodesOnFace = &ws_nodesOnFace[0];
    double *p_nodesOffFace = &ws_nodesOffFace[0];

    // shape function
    meFC->shape_fcn(&p_face_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get face
      stk::mesh::Entity face = b[k];

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, b, k);

      //======================================
      // gather nodal data off of face
      //======================================
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);
      const int num_face_nodes = bulk_data.num_nodes(face);
      // sanity check on num nodes
      ThrowAssert( num_face_nodes == nodesPerFace );
      for ( int ni = 0; ni < num_face_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];
        // gather scalars
        p_density[ni] = *stk::mesh::field_data(*density_, node);
        const double lambda = *stk::mesh::field_data(*thermalCond_, node);
        const double Cp = *stk::mesh::field_data(*specificHeat_, node);
        p_specificHeat[ni] = Cp;
        p_thermalCond[ni] = lambda;
      }

      // extract the connected element to this exposed face; should be single in size!
      stk::mesh::Entity const * face_elem_rels = b.begin_elements(k);
      ThrowAssert( b.num_elements(k) == 1 );

      // get element; its face ordinal
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = b.begin_element_ordinals(k)[0];

      // mapping from ip to nodes for this ordinal; element perspective (use with elem_node_relations)
      const int *ipNodeMap = meSCS->ipNodeMap(face_ordinal);

      //==========================================
      // gather nodal data off of element
      //==========================================
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        // sneak in nodesOn/offFace
        p_nodesOnFace[ni] = 0.0;
        p_nodesOffFace[ni] = 1.0;
        stk::mesh::Entity node = elem_node_rels[ni];
        // gather scalars
        p_temperature[ni] = *stk::mesh::field_data(*temperature_, node);
        // gather vectors
        double * coords = stk::mesh::field_data(*coordinates_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_coordinates[offSet+j] = coords[j];
        }
      }

      // process on/off while looping over face nodes
      for ( int ip = 0; ip < numScsBip; ++ip ) {
        const int nearestNode = ipNodeMap[ip];
        p_nodesOnFace[nearestNode] = 1.0;
        p_nodesOffFace[nearestNode] = 0.0;
      }

      // compute dndx
      double scs_error = 0.0;
      meSCS->face_grad_op(1, face_ordinal, &p_coordinates[0], &p_dndx[0], &ws_det_j[0], &scs_error);

      // loop over boundary ips
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int nearestNode = ipNodeMap[ip];
        stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

        // pointers to nearest node data
        double *assembledWallArea = stk::mesh::field_data(*assembledWallArea_, nodeR);
        double *referenceTemperature = stk::mesh::field_data(*referenceTemperature_, nodeR);
        double *heatTransferCoefficient = stk::mesh::field_data(*heatTransferCoefficient_, nodeR);
        double *normalHeatFlux = stk::mesh::field_data(*normalHeatFlux_, nodeR);
        double *robinCouplingParameter = stk::mesh::field_data(*robinCouplingParameter_, nodeR);

        // offset for bip area vector and types of shape function
        const int faceOffSet = ip*nDim;
        const int offSetSF_face = ip*nodesPerFace;

        // interpolate to bip
        double thermalCondBip = 0.0;
        double densityBip = 0.0;
        double specificHeatBip = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          thermalCondBip += r*p_thermalCond[ic];
          densityBip += r*p_density[ic];
          specificHeatBip += r*p_specificHeat[ic];
        }

        // handle flux due to on and off face in a single loop (on/off provided above)
        double dndx    = 0.0;
        double dndxOn  = 0.0;
        double dndxOff = 0.0;
        double invEltLen = 0.0;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
          const double nodesOnFace = p_nodesOnFace[ic];
          const double nodesOffFace = p_nodesOffFace[ic];
          const double tempIC = p_temperature[ic];
          for ( int j = 0; j < nDim; ++j ) {
            const double axj = areaVec[faceOffSet+j];
            const double dndxj = p_dndx[offSetDnDx+j];
            const double dTdA = dndxj*axj*tempIC;
            dndx    += dTdA;
            dndxOn  += dTdA*nodesOnFace;
            dndxOff += dTdA*nodesOffFace;
            invEltLen += dndxj*axj*nodesOnFace;
          }
        }

        // compute assembled area
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[faceOffSet+j];
          aMag += axj*axj;
        }
        aMag = std::sqrt(aMag);
        double eltLen = aMag/invEltLen;

        // compute coupling parameter
        const double chi = densityBip * specificHeatBip * eltLen * eltLen 
          / (2 * thermalCondBip * dt);
        const double alpha = compute_coupling_parameter(thermalCondBip, eltLen, chi);

        // assemble the nodal quantities
        *assembledWallArea += aMag;
        *normalHeatFlux -= thermalCondBip*dndx;
        *referenceTemperature -= thermalCondBip*dndxOff;
        *heatTransferCoefficient -= thermalCondBip*dndxOn;
        *robinCouplingParameter += alpha*aMag;
      }
    }
  }
}


//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeHeatTransferElemWallAlgorithm::~ComputeHeatTransferElemWallAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- compute_coupling_parameter --------------------------------------
//--------------------------------------------------------------------------
double
ComputeHeatTransferElemWallAlgorithm::compute_coupling_parameter(const double & kappa,
                                                                 const double & h,
                                                                 const double & chi)
{
  // This function approximates the ideal coupling parameter for Dirichlet-Robin coupling
  const double A = 1.0 + chi - 1.0/(1.0 + chi + std::sqrt(chi*(chi+2)));
  return A * kappa/h;
}

} // namespace nalu
} // namespace Sierra
