// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



// nalu
#include <pmr/AssembleRadTransEdgeSolverAlgorithm.h>
#include <pmr/RadiativeTransportEquationSystem.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>

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
// AssembleRadTransEdgeSolverAlgorithm - add LHS/RHS for Intensity (SUCV)
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleRadTransEdgeSolverAlgorithm::AssembleRadTransEdgeSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  RadiativeTransportEquationSystem *radEqSystem)
  : SolverAlgorithm(realm, part, radEqSystem),
    radEqSystem_(radEqSystem),
    intensity_(NULL),
    edgeAreaVec_(NULL),
    coordinates_(NULL),
    absorption_(NULL),
    scattering_(NULL),
    scalarFlux_(NULL),
    radiationSource_(NULL),
    dualNodalVolume_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
  // residual contribution
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  absorption_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "absorption_coefficient");
  scattering_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scattering_coefficient");
  scalarFlux_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_flux");
  radiationSource_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "radiation_source");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleRadTransEdgeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleRadTransEdgeSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // use edge-based length scale
  const bool useEdgeH = true;

  // extract current ordinate direction
  std::vector<double> Sk(nDim,0.0);
  radEqSystem_->get_current_ordinate(&Sk[0]);
  const double *p_Sk = &Sk[0];
  intensity_ = radEqSystem_->get_intensity();

  const double invPi = 1.0/(std::acos(-1.0));

  // space for LHS/RHS; always nodesPerEdge*nodesPerEdge and nodesPerEdge
  std::vector<double> lhs(4);
  std::vector<double> rhs(2);
  std::vector<int> scratchIds(2);
  std::vector<double> scratchVals(2);
  std::vector<stk::mesh::Entity> connected_nodes(2);

  // area vector; gather into
  std::vector<double> areaVec(nDim);

  // pointers for fast access
  double *p_lhs = &lhs[0];
  double *p_rhs = &rhs[0];
  double *p_areaVec = &areaVec[0];

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& edge_buckets =
    realm_.get_buckets( stk::topology::EDGE_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
        ib != edge_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();

    // pointer to edge area vector
    const double * av = stk::mesh::field_data(*edgeAreaVec_, b);

    for ( size_t k = 0 ; k < length ; ++k ) {

      // set ordinal for edge
      unsigned edge_ordinal = k;
      // sanity check on number or nodes
      ThrowAssert( b.num_nodes(edge_ordinal) == 2 );

      stk::mesh::Entity const * edge_node_rels = b.begin_nodes(edge_ordinal);

      // pointer to edge area vector
      for ( int j = 0; j < nDim; ++j )
        p_areaVec[j] = av[k*nDim+j];

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      connected_nodes[0] = nodeL;
      connected_nodes[1] = nodeR;

      // extract nodal fields
      const double * coordL = stk::mesh::field_data(*coordinates_, nodeL);
      const double * coordR = stk::mesh::field_data(*coordinates_, nodeR);

      const double intensityL = *stk::mesh::field_data(*intensity_, nodeL);
      const double intensityR = *stk::mesh::field_data(*intensity_, nodeR);

      const double absorptionL = *stk::mesh::field_data(*absorption_, nodeL);
      const double absorptionR = *stk::mesh::field_data(*absorption_, nodeR);

      const double scatteringL = *stk::mesh::field_data(*scattering_, nodeL);
      const double scatteringR = *stk::mesh::field_data(*scattering_, nodeR);

      const double scalarFluxL = *stk::mesh::field_data(*scalarFlux_, nodeL);
      const double scalarFluxR = *stk::mesh::field_data(*scalarFlux_, nodeR);

      const double radiationSourceL = *stk::mesh::field_data(*radiationSource_, nodeL);
      const double radiationSourceR = *stk::mesh::field_data(*radiationSource_, nodeR);

      const double dualNodalVolumeL = *stk::mesh::field_data(*dualNodalVolume_, nodeL);
      const double dualNodalVolumeR = *stk::mesh::field_data(*dualNodalVolume_, nodeR);

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
      const double aMag = std::sqrt(asq);
      
      // extinction coefficient; always include scattering (likely zero)
      const double extinctionL = absorptionL + scatteringL;
      const double extinctionR = absorptionR + scatteringR;
      const double extinctionIp = 0.5*(extinctionL + extinctionR);

      // construct part of the residual
      const double muI = extinctionIp*0.5*(intensityL + intensityR);
      const double eP = 0.5*(radiationSourceL + radiationSourceR);
      const double isotropicScatter = 0.5*(scatteringL*scalarFluxL + scatteringR*scalarFluxR)/4.0*invPi;

      // compute Sj*njdS; fill in residual; compute length scale
      double sjaj = 0.0;
      double h_edge = 0.0;
      double residual = muI - eP - isotropicScatter;
      for ( int j = 0; j < nDim; ++j ) {
        const double axj = p_areaVec[j];
        const double nj = axj/aMag;
        const double dxj = coordR[j] - coordL[j];
        sjaj += p_Sk[j]*axj;
        residual += p_Sk[j]*(intensityR-intensityL)*axj*inv_axdx;
        h_edge += nj*dxj;
      }

      // alternative h
      const double dualNodalIp = 0.5*(dualNodalVolumeL+dualNodalVolumeR);
      const double h_vol = std::pow(dualNodalIp, 1.0/(double)nDim);

      // pick one
      const double h = (useEdgeH) ? h_edge : h_vol;

      // form tau
      const double tau = std::sqrt(1.0/((2.0/h)*(2.0/h) + extinctionIp*extinctionIp));

      /*
        lhs[0] = IL,IL; lhs[1] = IL,IR; IR,IL; IR,IR
      */

      // pure central term; Iip*sj*njdS
      double lhsfac = 0.5*sjaj;

      // left node
      p_lhs[0] = +lhsfac;
      p_lhs[1] = +lhsfac;
      // now right node
      p_lhs[2] = -lhsfac;
      p_lhs[3] = -lhsfac;

      // residual
      const double intensityIp = 0.5*(intensityL+intensityR);
      p_rhs[0] = -sjaj*intensityIp;
      p_rhs[1] = +sjaj*intensityIp;

      // SUCV diffusion-like term; tau*si*dI/dxi*sjnj; complete residual below
      lhsfac = -tau*sjaj*sjaj*inv_axdx;
      // left node
      p_lhs[0] -= lhsfac;
      p_lhs[1] += lhsfac;
      // now right node
      p_lhs[2] += lhsfac;
      p_lhs[3] -= lhsfac;

      // SUCV muI term; complete residual below
      lhsfac = -tau*sjaj*0.5*extinctionIp;
      // left node
      p_lhs[0] += lhsfac;
      p_lhs[1] += lhsfac;
      // now right node
      p_lhs[2] -= lhsfac;
      p_lhs[3] -= lhsfac;

      // final SUCV residual
      const double sucv = -tau*sjaj*residual;
      p_rhs[0] -= sucv;
      p_rhs[1] += sucv;

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
