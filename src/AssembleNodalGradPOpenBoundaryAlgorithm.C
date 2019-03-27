/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodalGradPOpenBoundaryAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"
#include <SolutionOptions.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradPBoundaryAlgorithm - adds in boundary contribution
//                                      for elem/edge proj nodal gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradPOpenBoundaryAlgorithm::AssembleNodalGradPOpenBoundaryAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  const bool useShifted)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    zeroGrad_(realm_.solutionOptions_->explicitlyZeroOpenPressureGradient_),
    massCorr_(realm_.solutionOptions_->activateOpenMdotCorrection_)
{
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradPOpenBoundaryAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const auto& bulk = realm_.bulk_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields
  GenericFieldType& exposedAreaVec =
      *meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  ScalarFieldType& dualNodalVolume =
      *meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  std::string exposedPressurefieldName =  massCorr_ ? "pressure" : "pressure_bc";
  ThrowRequire(meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, exposedPressurefieldName) != nullptr);

  ScalarFieldType& exposedPressureField =
      meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, exposedPressurefieldName)
      ->field_of_state(stk::mesh::StateNone);

  ScalarFieldType& pressureField =
      meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure")
      ->field_of_state(stk::mesh::StateNone);

  VectorFieldType& GpField =
      meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx")->field_of_state(stk::mesh::StateNone);

  std::vector<double> ws_face_pressure;
  std::vector<double> ws_elem_pressure;

  std::vector<double> ws_face_shape_function;
  std::vector<double> ws_elem_shape_function;

  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  std::vector<stk::topology> topoVec(1);

  stk::mesh::BucketVector const& face_buckets = realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for (const auto* ib : face_buckets ) {
    const auto& b = *ib;
    const stk::mesh::Bucket::size_type length   = b.size();

    MasterElement& meFC = *MasterElementRepo::get_surface_master_element(b.topology());

    b.parent_topology(stk::topology::ELEM_RANK, topoVec);
    MasterElement& meSCS = *MasterElementRepo::get_surface_master_element(topoVec[0]);

    const int nodesPerFace = meFC.nodesPerElement_;
    const int numFaceScsIp = meFC.numIntPoints_;
    const int nodesPerElement = meSCS.nodesPerElement_;
    const int numElemScsIp = meSCS.numIntPoints_;

    const int *ipNodeMap = meFC.ipNodeMap();

    ws_face_pressure.resize(nodesPerFace);
    double *p_facePressure = ws_face_pressure.data();

    ws_elem_pressure.resize(nodesPerElement);
    double *p_elemPressure = ws_elem_pressure.data();

    ws_face_shape_function.resize(numFaceScsIp*nodesPerFace);
    double *p_face_shape_function = ws_face_shape_function.data();

    ws_elem_shape_function.resize(nodesPerElement*numElemScsIp);
    double *p_elem_shape_function = ws_elem_shape_function.data();

    if ( useShifted_ ) {
      meFC.shifted_shape_fcn(p_face_shape_function);
      meSCS.shifted_shape_fcn(p_elem_shape_function);
    }
    else {
      meFC.shape_fcn(p_face_shape_function);
      meSCS.shape_fcn(p_elem_shape_function);
    }

    for ( size_t k = 0 ; k < length ; ++k ) {
      const double * areaVec = stk::mesh::field_data(exposedAreaVec, b, k);

      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);
      for ( int n = 0; n < nodesPerFace; ++n ) {
        p_facePressure[n] = *stk::mesh::field_data(exposedPressureField, face_node_rels[n]);
      }

      const stk::mesh::Entity elem = bulk.begin_elements(b[k])[0];
      const auto* elem_node_rels = bulk.begin_nodes(elem);
      for (int n = 0; n < nodesPerElement; ++n) {
        const stk::mesh::Entity node = elem_node_rels[n];
        p_elemPressure[n] = *stk::mesh::field_data(pressureField, node);
      }

      const int faceOrdinal = bulk.begin_element_ordinals(b[k])[0];
      for (int ip = 0; ip < numFaceScsIp; ++ip) {
        const auto* areav = &areaVec[ip * nDim];
        const int nn = ipNodeMap[ip];

        stk::mesh::Entity nodeNN = face_node_rels[nn];
        double *gradQNN = stk::mesh::field_data(GpField, nodeNN);
        const double volNN = *stk::mesh::field_data(dualNodalVolume, nodeNN);

        double pIp = 0.0;
        const int offSet = ip * nodesPerFace;
        for (int ic = 0; ic < nodesPerFace; ++ic) {
          pIp += p_face_shape_function[offSet + ic] * p_facePressure[ic];
        }

        // evaluate pressure at opposing face.  If zeroGrad_ option is used, then
        // we'll copy this value to the exposed face.  Otherwise, it's unused
        double pOpp = 0.0;
        const int opp_offset = meSCS.opposingFace(faceOrdinal, ip) * nodesPerElement;
        const double* oppShapeFns = &p_elem_shape_function[opp_offset];
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          pOpp += oppShapeFns[ic] * p_elemPressure[ic];
        }

        double press_div_vol = (zeroGrad_ ? pOpp : pIp) / volNN;
        for ( int j = 0; j < nDim; ++j ) {
          gradQNN[j] += press_div_vol * areav[j];
        }
      }
    }
  }
}

} // namespace nalu
} // namespace Sierra
