/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "utils/ComputeVectorDivergence.h"

#include <master_element/MasterElement.h>

#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/GetBuckets.hpp>

namespace sierra {
namespace nalu {

void compute_vector_divergence(
  stk::mesh::BulkData & bulk,
  stk::mesh::PartVector & partVec,
  stk::mesh::FieldBase * vectorField,
  stk::mesh::FieldBase * scalarField,
  const bool hasMeshDeformation )
{
  stk::mesh::MetaData & meta = bulk.mesh_meta_data();
  const int nDim = meta.spatial_dimension();

  const std::string coordName = hasMeshDeformation ? "current_coordinates" : "coordinates";
  VectorFieldType* coordinates = meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, coordName);

  ScalarFieldType* dualVol = meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  stk::mesh::field_fill(0.0, *scalarField);

  std::vector<double> wsCoordinates;
  std::vector<double> wsScsArea;
  std::vector<double> wsMeshVector;

  std::vector<double> ws_shape_function;

  std::array<double,3> mvIp;

  stk::mesh::Selector sel = meta.locally_owned_part()
          & stk::mesh::selectUnion(partVec);
  const auto& bkts =
      bulk.get_buckets( stk::topology::ELEMENT_RANK, sel );

  for (auto b: bkts) {
    MasterElement *meSCS =
        MasterElementRepo::get_surface_master_element(b->topology());
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
    const int *lrscv = meSCS->adjacentNodes();

    wsMeshVector.resize(nodesPerElement*nDim);
    wsCoordinates.resize(nodesPerElement*nDim);
    wsScsArea.resize(numScsIp*nDim);

    ws_shape_function.resize(numScsIp*nodesPerElement);
    meSCS->shape_fcn(ws_shape_function.data());


    size_t length = b->size();
    for ( size_t k = 0 ; k < length ; ++k ) {

      stk::mesh::Entity const * node_rels = b->begin_nodes(k);
      int num_nodes = b->num_nodes(k);

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        const double * coords =  stk::mesh::field_data(*coordinates, node);
        const double * mv   =  (double*)stk::mesh::field_data(*vectorField, node);

        for (int iDim=0; iDim < nDim; iDim++) {
          wsCoordinates[ni*nDim+iDim] = coords[iDim];
          wsMeshVector[ni*nDim+iDim] = mv[iDim];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, wsCoordinates.data(), wsScsArea.data(), &scs_error);

      for ( int ip = 0; ip < numScsIp; ++ip ) {

        const int ipNdim = ip*nDim;

        const int offSetSF = ip*nodesPerElement;

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        stk::mesh::Entity nodeL = node_rels[il];
        stk::mesh::Entity nodeR = node_rels[ir];

        // pointer to fields to assemble
        double *divMVL = (double*)stk::mesh::field_data(*scalarField, nodeL);
        double *divMVR = (double*)stk::mesh::field_data(*scalarField, nodeR);

        double *dualVolL = stk::mesh::field_data(*dualVol, nodeL);
        double *dualVolR = stk::mesh::field_data(*dualVol, nodeR);

        //Compute mesh vector at this ip
        for ( int j = 0; j < nDim; ++j )
          mvIp[j] = 0.0;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = ws_shape_function[offSetSF+ic];
          for (int j=0; j < nDim; j++)
            mvIp[j] += r * wsMeshVector[ic*nDim+j];
        }

        //Compute dot product with area
        double mvDotArea = 0.0;
        for (int j=0; j < nDim; j++)
          mvDotArea += mvIp[j] * wsScsArea[ipNdim+j];

        *divMVL += mvDotArea/ (*dualVolL);
        *divMVR -= mvDotArea/ (*dualVolR);
      }

    }

  }

}

}
}
