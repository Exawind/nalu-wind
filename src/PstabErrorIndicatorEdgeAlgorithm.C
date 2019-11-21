// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



// nalu
#include <PstabErrorIndicatorEdgeAlgorithm.h>

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

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// PstabErrorIndicatorEdgeAlgorithm - error indicator for edge fluids
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PstabErrorIndicatorEdgeAlgorithm::PstabErrorIndicatorEdgeAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *pressure,
  VectorFieldType *Gpdx,
  const bool simpleGradApproach)
  : Algorithm(realm, part),
    pressure_(pressure),
    Gpdx_(Gpdx),
    coordinates_(NULL),
    edgeAreaVec_(NULL),
    pstabEI_(NULL),
    simpleGradApproachScale_(simpleGradApproach ? 0.0 : 1.0)
{
  // save off field
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
  pstabEI_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "error_indicator");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
PstabErrorIndicatorEdgeAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // time step
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double projTimeScale = dt/gamma1;

  // area vector; gather into
  std::vector<double> areaVec(nDim);

  // pointers for fast access
  double *p_areaVec = &areaVec[0];

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // pointer to elem data
    double * pstabEI = stk::mesh::field_data(*pstabEI_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // iterate edges
      stk::mesh::Entity const * elem_edge_rels = b.begin_edges(k);
      int num_edges = b.num_edges(k);
      
      // initialize error indicator
      double errorIndicator = 0.0;

      for ( int nedge = 0; nedge < num_edges; ++nedge ) {
        // get edge and area_vector
        stk::mesh::Entity edge = elem_edge_rels[nedge];
        const double * av = stk::mesh::field_data(*edgeAreaVec_, edge );

        // extract edge->node relations
        stk::mesh::Entity const * edge_node_rels = bulk_data.begin_nodes(edge);
        ThrowAssert( 2 == bulk_data.num_nodes(edge) );

        // pointer to edge area vector; fill
        for ( int j = 0; j < nDim; ++j )
          p_areaVec[j] = av[j];

        // left and right nodes
        stk::mesh::Entity nodeL = edge_node_rels[0];
        stk::mesh::Entity nodeR = edge_node_rels[1];

        // extract nodal fields
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

        const double * GpdxL = stk::mesh::field_data(*Gpdx_, nodeL );
        const double * GpdxR = stk::mesh::field_data(*Gpdx_, nodeR );

        const double pressureL = *stk::mesh::field_data(*pressure_, nodeL );
        const double pressureR = *stk::mesh::field_data(*pressure_, nodeR );

        // compute geometry
        double axdx = 0.0;
        double asq = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = p_areaVec[j];
          const double dxj = coordR[j] - coordL[j];
          asq += axj*axj;
          axdx += axj*dxj;
        }

        const double inv_axdx = 1.0/axdx;

        // initialize local grad p contribution
        double eiEdge = -projTimeScale*(pressureR-pressureL)*asq*inv_axdx;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = p_areaVec[j];
          const double dxj = coordR[j] - coordL[j];
          const double kxj = axj - asq*inv_axdx*dxj; // NOC
          const double GjIp = 0.5*(GpdxR[j] + GpdxL[j])*simpleGradApproachScale_;
          const double theEI = projTimeScale*(GjIp*axj - kxj*GjIp);
          eiEdge += theEI;
        }
        errorIndicator += eiEdge*eiEdge;
      }

      // scatter to error indicator
      pstabEI[k] = std::sqrt(errorIndicator);
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
PstabErrorIndicatorEdgeAlgorithm::~PstabErrorIndicatorEdgeAlgorithm()
{
  // does nothing
}



} // namespace nalu
} // namespace Sierra
