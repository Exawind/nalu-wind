/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumElemABLWallFunctionSolverAlgorithm.h>
#include <SolverAlgorithm.h>
#include <EquationSystem.h>
#include <LinearSystem.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"
#include <ABLProfileFunction.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
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
// AssembleMomentumElemABLWallFunctionSolverAlgorithm - ABL elem wall function
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemABLWallFunctionSolverAlgorithm::AssembleMomentumElemABLWallFunctionSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  const bool &useShifted,
  const double &gravity,
  const double &z0,
  const double &Tref)
  : SolverAlgorithm(realm, part, eqSystem),
    useShifted_(useShifted),
    z0_(z0), 
    Tref_(Tref), 
    gravity_(gravity), 
    alpha_h_(1.0), 
    beta_m_(16.0), 
    beta_h_(16.0),
    gamma_m_(5.0),
    gamma_h_(5.0),
    kappa_(realm.get_turb_model_constant(TM_kappa))
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "wall_velocity_bc");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  bcHeatFlux_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_flux_bc");
  specificHeat_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  wallFrictionVelocityBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_friction_velocity_bip");
  wallNormalDistanceBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_normal_distance_bip");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemABLWallFunctionSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemABLWallFunctionSolverAlgorithm::execute()
{

  ABLProfileFunction *p_ABLProfFun;
  StableABLProfileFunction StableProfFun(gamma_m_, gamma_h_);
  UnstableABLProfileFunction UnstableProfFun(beta_m_, beta_h_);
  NeutralABLProfileFunction NeutralProfFun;

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // space for LHS/RHS; nodesPerFace*nDim*nodesPerFace*nDim and nodesPerFace*nDim
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // bip values
  std::vector<double> uBip(nDim);
  std::vector<double> uBcBip(nDim);
  std::vector<double> unitNormal(nDim);

  // pointers to fixed values
  double *p_uBip = &uBip[0];
  double *p_uBcBip = &uBcBip[0];
  double *p_unitNormal= &unitNormal[0];

  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_bcVelocity;
  std::vector<double> ws_bcHeatFlux;
  std::vector<double> ws_density;
  std::vector<double> ws_specificHeat;


  // master element
  std::vector<double> ws_shape_function;
  std::vector<double> ws_face_shape_function;

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal; face perspective (use with face_node_relations)
    const int *faceIpNodeMap = meFC->ipNodeMap();

    // resize some things; matrix related
    const int lhsSize = nodesPerFace*nDim*nodesPerFace*nDim;
    const int rhsSize = nodesPerFace*nDim;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerFace);

    // algorithm related; element
    ws_velocityNp1.resize(nodesPerFace*nDim);
    ws_bcVelocity.resize(nodesPerFace*nDim);
    ws_bcHeatFlux.resize(nodesPerFace);
    ws_density.resize(nodesPerFace);
    ws_specificHeat.resize(nodesPerFace);

    ws_face_shape_function.resize(numScsBip*nodesPerFace);

    // pointers
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_velocityNp1 = &ws_velocityNp1[0];
    double *p_bcVelocity = &ws_bcVelocity[0];
    double *p_bcHeatFlux = &ws_bcHeatFlux[0];
    double *p_density = &ws_density[0];
    double *p_specificHeat = &ws_specificHeat[0];
    double *p_face_shape_function = &ws_face_shape_function[0];

    // shape functions
    if ( useShifted_ )
      meFC->shifted_shape_fcn(&p_face_shape_function[0]);
    else
      meFC->shape_fcn(&p_face_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        p_lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        p_rhs[p] = 0.0;

      // get face
      stk::mesh::Entity face = b[k];

      //======================================
      // gather nodal data off of face
      //======================================
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);
      for ( int ni = 0; ni < nodesPerFace; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];
        connected_nodes[ni] = node;

        // gather scalars
        p_bcHeatFlux[ni] = *stk::mesh::field_data(*bcHeatFlux_, node);
        p_density[ni]    = *stk::mesh::field_data(densityNp1, node);
        p_specificHeat[ni] = *stk::mesh::field_data(*specificHeat_, node);

        // gather vectors
        double * uNp1 = stk::mesh::field_data(velocityNp1, node);
        double * uBc = stk::mesh::field_data(*bcVelocity_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_velocityNp1[offSet+j] = uNp1[j];
          p_bcVelocity[offSet+j] = uBc[j];
        }
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
      const double *wallNormalDistanceBip = stk::mesh::field_data(*wallNormalDistanceBip_, face);
      const double *wallFrictionVelocityBip = stk::mesh::field_data(*wallFrictionVelocityBip_, face);

      // loop over face nodes
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int offSetAveraVec = ip*nDim;
        const int offSetSF_face = ip*nodesPerFace;

        const int localFaceNode = faceIpNodeMap[ip];

        // zero out vector quantities; squeeze in aMag
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          p_uBip[j] = 0.0;
          p_uBcBip[j] = 0.0;
          const double axj = areaVec[offSetAveraVec+j];
          aMag += axj*axj;
        }
        aMag = std::sqrt(aMag);

        // interpolate to bip
        double heatFluxBip = 0.0;
        double rhoBip = 0.0;
        double CpBip = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          rhoBip += r*p_density[ic];
          CpBip += r*p_specificHeat[ic];
          heatFluxBip += r*p_bcHeatFlux[ic];
          const int offSetFN = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uBip[j] += r*p_velocityNp1[offSetFN+j];
            p_uBcBip[j] += r*p_bcVelocity[offSetFN+j];
          }
        }

        // form unit normal
        for ( int j = 0; j < nDim; ++j ) {
          p_unitNormal[j] = areaVec[offSetAveraVec+j]/aMag;
        }

        // extract bip data
        const double yp = wallNormalDistanceBip[ip];
        const double utau= wallFrictionVelocityBip[ip];

        // determine lambda = tau_w / (rho * u_* * u_tan)
        // NOTE:
        // density should be obtained from the wall function for
        // temperature and a relation rho = rho(T)
        const double TfluxBip = heatFluxBip / (rhoBip * CpBip);

        const double eps_heat_flux = 1.0e-8;
        const double largenum = 1.0e8;
        double Lfac;
        if (TfluxBip < eps_heat_flux) {
          p_ABLProfFun = &StableProfFun;
        }
        else if (TfluxBip > eps_heat_flux) {
          p_ABLProfFun = &UnstableProfFun;
        }
        else {
          p_ABLProfFun = &NeutralProfFun;
        }
        if (std::abs(TfluxBip) < eps_heat_flux) {
          Lfac = largenum;
        }
        else {
          Lfac = -Tref_ / (kappa_ * gravity_ * TfluxBip);
        }
        double L = utau*utau*utau * Lfac;
        // limit the values of L...
        //   to be negative and finite for q>0 (unstable)
        //   to be positive and finite for q<0 (stable)
        double sgnq = (TfluxBip > 0.0)?1.0:-1.0;
        L = - sgnq * std::max(1.0e-10,std::abs(L));

        //const double theta_star = -TfluxBip / utau;
        //need TBip from temperature field - this is T(yp);
        //const double Tsurf = TBip - theta_star/kappa_*(alpha_h_*std::log(yp/z0_) + gamma_h_*yp/L);
        // evaluate rhosurf = f(Tsurf) and use rhosurf in place of rhoBip below
        double lambda = (rhoBip*kappa_*utau/(std::log(yp/z0_) - p_ABLProfFun->velocity(yp/L))) * aMag;

        // start the lhs assembly
        for ( int i = 0; i < nDim; ++i ) {

          int indexR = localFaceNode*nDim + i;
          int rowR = indexR*nodesPerFace*nDim;

          double uiTan = 0.0;
          double uiBcTan = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            const double ninj = p_unitNormal[i]*p_unitNormal[j];
            if ( i==j ) {
              const double om_nini = 1.0 - ninj;
              uiTan += om_nini*p_uBip[j];
              uiBcTan += om_nini*p_uBcBip[j];
              for (int ic = 0; ic < nodesPerFace; ++ic)
                p_lhs[rowR+ic*nDim+i] += lambda*om_nini*p_face_shape_function[offSetSF_face+ic];
            }
            else {
              uiTan -= ninj*p_uBip[j];
              uiBcTan -= ninj*p_uBcBip[j];
              for (int ic = 0; ic < nodesPerFace; ++ic)
                p_lhs[rowR+ic*nDim+j] -= lambda*ninj*p_face_shape_function[offSetSF_face+ic];
            }
          }
          p_rhs[indexR] -= lambda*(uiTan-uiBcTan);
        }
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}


} // namespace nalu
} // namespace Sierra
