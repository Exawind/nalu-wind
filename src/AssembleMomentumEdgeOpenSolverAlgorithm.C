/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumEdgeOpenSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SolutionOptions.h>

#include <master_element/MasterElement.h>

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
// AssembleMomentumEdgeOpenSolverAlgorithm - momentum at open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumEdgeOpenSolverAlgorithm::AssembleMomentumEdgeOpenSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    includeDivU_(realm_.get_divU()),
    velocity_(NULL),
    dudx_(NULL),
    coordinates_(NULL),
    viscosity_(NULL),
    exposedAreaVec_(NULL),
    openMassFlowRate_(NULL),
    velocityBc_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  // extract viscosity  name
  const std::string viscName = realm_.is_turbulent()
    ? "effective_viscosity_u" : "viscosity";
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");
  velocityBc_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "open_velocity_bc");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeOpenSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeOpenSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // nearest face entrainment
  const std::string dofName = "velocity";
  const double nfEntrain = realm_.solutionOptions_->nearestFaceEntrain_;
  const double om_nfEntrain = 1.0-nfEntrain;
  const double relaxFacU = realm_.solutionOptions_->get_relaxation_factor(dofName);

  // space for dui/dxj; the modified gradient with NOC
  std::vector<double> duidxj(nDim*nDim);

  // lhs/rhs space
  std::vector<double> rhs;
  std::vector<double> lhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  std::vector<double> nx(nDim);
  std::vector<double> fx(nDim);

  // pointers
  double *p_duidxj = &duidxj[0];
  double *p_nx = &nx[0];
  double *p_fx = &fx[0];

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);

  // define vector of parent topos
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

    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nDim*nodesPerElement*nDim;
    const int rhsSize = nodesPerElement*nDim;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerElement);

    // pointer to lhs/rhs
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];

    // size some things that are useful
    const int num_face_nodes = b.topology().num_nodes();

    const stk::mesh::Bucket::size_type length = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        p_lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        p_rhs[p] = 0.0;

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, b, k);
      const double * mdot    = stk::mesh::field_data(*openMassFlowRate_, b, k);

      // extract the connected element to this exposed face; should be single in size!
      stk::mesh::Entity const * face_elem_rels = b.begin_elements(k);
      ThrowAssert( b.num_elements(k) == 1 );

      // get element; its face ordinal number and populate face_node_ordinals
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = b.begin_element_ordinals(k)[0];
      const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

      // get the relations; populate connected nodes
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        // set connected nodes
        connected_nodes[ni] = elem_node_rels[ni];
      }

      // loop over face nodes (maps directly to ips)
      for ( int ip = 0; ip < num_face_nodes; ++ip ) {

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);  // "Left"
        const int nearestNode = face_node_ordinals[ip];  // "Right"

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

        // extract nodal fields
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

        const double * uNp1L = stk::mesh::field_data(velocityNp1, nodeL );
        const double * uNp1R = stk::mesh::field_data(velocityNp1, nodeR );

        const double viscosityR = *stk::mesh::field_data(*viscosity_, nodeR );
        const double viscBip = viscosityR;

        // a few only required (or even defined) on nodeR
        const double *bcVelocity =  stk::mesh::field_data(*velocityBc_, nodeR );
        const double * dudxR     =  stk::mesh::field_data(*dudx_, nodeR );

        // offset for bip area vector
        const int faceOffSet = ip*nDim;

        // compute geometry
        double axdx = 0.0;
        double asq = 0.0;
        double udotx = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[faceOffSet+j];
          const double dxj = coordR[j]  - coordL[j];
          asq += axj*axj;
          axdx += axj*dxj;
          udotx += 0.5*dxj*(uNp1L[j] + uNp1R[j]);
        }

        const double inv_axdx = 1.0/axdx;
        const double amag = std::sqrt(asq);

        // form duidxj with over-relaxed procedure of Jasak:
        for ( int i = 0; i < nDim; ++i ) {

          // difference between R and L nodes for component i
          const double uidiff = uNp1R[i] - uNp1L[i];

          // offset into all forms of dudx
          const int offSetI = nDim*i;

          // start sum for NOC contribution
          double GlUidxl = 0.0;
          for ( int l = 0; l< nDim; ++l ) {
            const int offSetIL = offSetI+l;
            const double dxl = coordR[l] - coordL[l];
            const double GlUi = dudxR[offSetIL];
            GlUidxl += GlUi*dxl;
          }

          // form full tensor dui/dxj with NOC
          for ( int j = 0; j < nDim; ++j ) {
            const int offSetIJ = offSetI+j;
            const double axj = areaVec[faceOffSet+j];
            const double GjUi = dudxR[offSetIJ];
            p_duidxj[offSetIJ] = GjUi + (uidiff - GlUidxl)*axj*inv_axdx;
          }
        }

        // divU
        double divU = 0.0;
        for ( int j = 0; j < nDim; ++j)
          divU += p_duidxj[j*nDim+j];

        double fxnx = 0.0;
        double uxnx = 0.0;
        double uxnxip = 0.0;
        double uspecxnx = 0.0;
        for (int i = 0; i < nDim; ++i ) {
          const double axi = areaVec[faceOffSet+i];
          double fxi = 2.0/3.0*viscBip*divU*axi*includeDivU_;
          const int offSetI = nDim*i;
          const double nxi = axi/amag;
          for ( int j = 0; j < nDim; ++j ) {
            const int offSetTrans = nDim*j+i;
            const double axj = areaVec[faceOffSet+j];
            fxi += -viscBip*(p_duidxj[offSetI+j] + p_duidxj[offSetTrans])*axj;
          }

          fxnx += nxi*fxi;
          uxnx += nxi*uNp1R[i];
          uxnxip += 0.5*nxi*(uNp1L[i]+uNp1R[i]);
          uspecxnx += nxi*bcVelocity[i];

          // save off normal and force for each component i
          p_nx[i] = nxi;
          p_fx[i] = fxi;
        }

        // full stress, sigma_ij
        for (int i = 0; i < nDim; ++i ) {

          // setup for matrix contribution assembly
          const int indexL = opposingNode*nDim + i;
          const int indexR = nearestNode*nDim + i;
          const int rowR = indexR*nodesPerElement*nDim;

          const int rRiL_i = rowR+indexL;
          const int rRiR_i = rowR+indexR;

          // subtract normal component
          const double diffFlux = p_fx[i] - p_nx[i]*fxnx;

          const double om_nxinxi = 1.0-p_nx[i]*p_nx[i];

          p_rhs[indexR] -= diffFlux;
          double lhsFac = -viscBip*asq*inv_axdx*om_nxinxi;
          p_lhs[rRiL_i] -= lhsFac;
          p_lhs[rRiR_i] += lhsFac / relaxFacU;

          const double axi = areaVec[faceOffSet+i];

          for ( int j = 0; j < nDim; ++j ) {
            const double axj = areaVec[faceOffSet+j];
            lhsFac = -viscBip*axi*axj*inv_axdx*om_nxinxi;

            const int colL = opposingNode*nDim + j;
            const int colR = nearestNode*nDim + j;

            const int rRiL_j = rowR+colL;
            const int rRiR_j = rowR+colR;

            p_lhs[rRiL_j] -= lhsFac;
            p_lhs[rRiR_j] += lhsFac / relaxFacU;

            if ( i == j ) {
              // nothing
            }
            else {
	      const double nxinxj = p_nx[i]*p_nx[j];

              lhsFac = viscBip*asq*inv_axdx*nxinxj;
              p_lhs[rRiL_j] -= lhsFac;
              p_lhs[rRiR_j] += lhsFac / relaxFacU;

              lhsFac = viscBip*axj*axj*inv_axdx*nxinxj;
              p_lhs[rRiL_j] -= lhsFac;
              p_lhs[rRiR_j] += lhsFac / relaxFacU;

              lhsFac = viscBip*axj*axi*inv_axdx*nxinxj;
              p_lhs[rRiL_i] -= lhsFac;
              p_lhs[rRiR_i] += lhsFac / relaxFacU;
            }
          }
        }

        // advection
        const double tmdot = mdot[ip];
        if ( tmdot > 0 ) {
          // leaving the domain
          for ( int i = 0; i < nDim; ++i ) {
            // setup for matrix contribution assembly
            const int indexR = nearestNode*nDim + i;
            const int rowR = indexR*nodesPerElement*nDim;
            const int rRiR = rowR+indexR;

            p_rhs[indexR] -= tmdot*uNp1R[i];
            p_lhs[rRiR] += tmdot / relaxFacU;
          }
        }
        else {
          // entraining; constrain to be normal
          for ( int i = 0; i < nDim; ++i ) {

            // setup for matrix contribution assembly
            const int indexR = nearestNode*nDim + i;
            const int rowR = indexR*nodesPerElement*nDim;

            // constrain to be normal
            p_rhs[indexR] -= tmdot*(nfEntrain*uxnx + om_nfEntrain*uxnxip)*p_nx[i];

            // user spec entrainment (tangential)
            p_rhs[indexR] -= tmdot*(bcVelocity[i]-uspecxnx*p_nx[i]);

            for ( int j = 0; j < nDim; ++j ) {

              const int colL = opposingNode*nDim + j;
              const int colR = nearestNode*nDim + j;

              p_lhs[rowR+colR] +=  tmdot*(nfEntrain + om_nfEntrain*0.5)*p_nx[i]*p_nx[j] / relaxFacU;
              p_lhs[rowR+colL] +=  tmdot*om_nfEntrain*0.5*p_nx[i]*p_nx[j];

            }

          }
        }
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
