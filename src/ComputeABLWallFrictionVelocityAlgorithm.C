/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeABLWallFrictionVelocityAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

#include <ABLProfileFunction.h>
#include <wind_energy/BdyLayerStatistics.h>

// stk_mesh/base/fem/util
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeABLWallFrictionVelocityAlgorithm - utau at ABL wall bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeABLWallFrictionVelocityAlgorithm::ComputeABLWallFrictionVelocityAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  const bool &useShifted,
  const double &gravity,
  const double &z0,
  const double &Tref)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    z0_(z0), 
    Tref_(Tref), 
    gravity_(gravity), 
    alpha_h_(1.0), 
    beta_m_(16.0), 
    beta_h_(16.0),
    gamma_m_(5.0),
    gamma_h_(5.0), 
    kappa_(realm.get_turb_model_constant(TM_kappa)),
    maxIteration_(40),
    tolerance_(1.0e-7)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "wall_velocity_bc");
  bcHeatFlux_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_flux_bc");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  specificHeat_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat");
  //viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  wallFrictionVelocityBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_friction_velocity_bip");
  wallNormalDistanceBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_normal_distance_bip");
  assembledWallArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_wf");
  assembledWallNormalDistance_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_normal_distance");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeABLWallFrictionVelocityAlgorithm::~ComputeABLWallFrictionVelocityAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeABLWallFrictionVelocityAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  ABLProfileFunction *p_ABLProfFun;
  StableABLProfileFunction StableProfFun(gamma_m_, gamma_h_);
  UnstableABLProfileFunction UnstableProfFun(beta_m_, beta_h_);
  NeutralABLProfileFunction NeutralProfFun;

  //Quantities to get the averate utau over the whole sideset
  std::array<double, 2> uTauAreaSumLocal{ {0.0, 0.0} }; // First value hold uTau * area, second value holds area. To get average utau, sum both quantities over all processes and divide one by another.
  std::array<double, 2> uTauAreaSumGlobal{ {0.0, 0.0} };
  double utau_calc;

  // zero out assembled nodal quantities
  zero_nodal_fields();

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

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theElemTopo);

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = b.topology().num_nodes();
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal; face perspective (use with face_node_relations)
    const int *faceIpNodeMap = meFC->ipNodeMap();

    // algorithm related; element
    ws_velocityNp1.resize(nodesPerFace*nDim);
    ws_bcVelocity.resize(nodesPerFace*nDim);
    ws_bcHeatFlux.resize(nodesPerFace);
    ws_density.resize(nodesPerFace);
    ws_specificHeat.resize(nodesPerFace);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);

    // pointers
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

      // get face
      stk::mesh::Entity face = b[k];

      //======================================
      // gather nodal data off of face
      //======================================
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);
      int num_face_nodes = bulk_data.num_nodes(face);
      // sanity check on num nodes
      ThrowAssert( num_face_nodes == nodesPerFace );

      for ( int ni = 0; ni < num_face_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

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
      double *wallNormalDistanceBip = stk::mesh::field_data(*wallNormalDistanceBip_, face);
      double *wallFrictionVelocityBip = stk::mesh::field_data(*wallFrictionVelocityBip_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];

      // get the relations off of element
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);

      // loop over face nodes
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int offSetAveraVec = ip*nDim;

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
        const int localFaceNode = faceIpNodeMap[ip];

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = face_node_rels[localFaceNode];

        // extract nodal fields
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

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
        const int offSetSF_face = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
	  heatFluxBip += r*p_bcHeatFlux[ic];
	  rhoBip += r*p_density[ic];
	  CpBip += r*p_specificHeat[ic];
          const int offSetFN = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uBip[j] += r*p_velocityNp1[offSetFN+j];
            p_uBcBip[j] += r*p_bcVelocity[offSetFN+j];
          }
        }
	
	const double eps_heat_flux = 1.0e-8;
	if (heatFluxBip < -eps_heat_flux) {
	  p_ABLProfFun = &StableProfFun;
	}
	else if (heatFluxBip > eps_heat_flux) {
	  p_ABLProfFun = &UnstableProfFun;
	}
	else {
	  p_ABLProfFun = &NeutralProfFun;
	}

        // form unit normal and determine yp (approximated by 1/4 distance along edge)
        double ypBip = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double nj = areaVec[offSetAveraVec+j]/aMag;
          const double ej = 0.25*(coordR[j] - coordL[j]);
          ypBip += nj*ej*nj*ej;
          p_unitNormal[j] = nj;
        }
        ypBip = std::sqrt(ypBip);
        wallNormalDistanceBip[ip] = ypBip;

        // assemble to nodal quantities
        double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, nodeR );
        double * assembledWallNormalDistance = stk::mesh::field_data(*assembledWallNormalDistance_, nodeR );

        *assembledWallArea += aMag;
        *assembledWallNormalDistance += aMag*ypBip;

        // determine tangential velocity
        double uTangential = 0.0;
        for ( int i = 0; i < nDim; ++i ) {
          double uiTan = 0.0;
          double uiBcTan = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            const double ninj = p_unitNormal[i]*p_unitNormal[j];
            if ( i==j ) {
              const double om_nini = 1.0 - ninj;
              uiTan += om_nini*p_uBip[j];
              uiBcTan += om_nini*p_uBcBip[j];
            }
            else {
              uiTan -= ninj*p_uBip[j];
              uiBcTan -= ninj*p_uBcBip[j];
            }
          }
          uTangential += (uiTan-uiBcTan)*(uiTan-uiBcTan);
        }
        uTangential = std::sqrt(uTangential);

	const double TfluxBip = heatFluxBip / (rhoBip * CpBip);
        compute_utau(uTangential, ypBip, TfluxBip, p_ABLProfFun, wallFrictionVelocityBip[ip]);
        uTauAreaSumLocal[0] += wallFrictionVelocityBip[ip] * aMag ;
        uTauAreaSumLocal[1] += aMag ;
      }
    }
  }
  
  if ((realm_.bdyLayerStats_ != nullptr) &&
      (realm_.isFinalOuterIter_)) {
    stk::all_reduce_sum(
      NaluEnv::self().parallel_comm(), uTauAreaSumLocal.data(),
      uTauAreaSumGlobal.data(), 2);
    utau_calc = uTauAreaSumGlobal[0] / uTauAreaSumGlobal[1];
    realm_.bdyLayerStats_->set_utau_avg(utau_calc);
  }

  // parallel assemble and normalize
  normalize_nodal_fields();
}

//--------------------------------------------------------------------------
//-------- zero_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeABLWallFrictionVelocityAlgorithm::zero_nodal_fields()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    double * assembledWallNormalDistance = stk::mesh::field_data(*assembledWallNormalDistance_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      assembledWallArea[k] = 0.0;
      assembledWallNormalDistance[k] = 0.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- compute_utau----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeABLWallFrictionVelocityAlgorithm::compute_utau(
    const double &up, const double &zp, const double &qsurf, const ABLProfileFunction *ABLProfFun, double &utau )
{
  bool converged = false;

  double Lfac;
  if (qsurf == 0.0) {
    Lfac = 0.0;
  }
  else {
    Lfac = -Tref_ / (kappa_ * gravity_ * qsurf);
  }
  const double log_z_over_z0 = std::log(zp / z0_);

  // Initial guesses for utau
  const double eps_u = 1.0e-8;
  const double perturb = 1.0e-3;
 
  double utau0;
  if (std::abs(up) < eps_u) {
    utau0 = eps_u;
    utau = eps_u;
    return;
  }
  else {
    utau0 = kappa_ * up / log_z_over_z0;
  }
  if (qsurf > 0.0) { // if unstable ABL
    utau0 = 3*utau0; // push initial guess above the singularity in the function to be zero'd
  }
  double utau1 = (1.0+perturb) * utau0;

  for ( int k = 0; k < maxIteration_; ++k) {

    // calculate Monin-Obukhov length
    double L0 = utau0*utau0*utau0 * Lfac;
    double L1 = utau1*utau1*utau1 * Lfac;

    // limit the values of L...
    //   to be negative and finite for qsurf>0 (unstable)
    //   to be positive and finite for qsurf<0 (stable)
    double sgnq = (qsurf > 0.0)?1.0:-1.0;
    L0 = - sgnq * std::max(1.0e-10,std::abs(L0));
    L1 = - sgnq * std::max(1.0e-10,std::abs(L1));

    // calculate normalized coordinate
    const double znorm0 = zp / L0;
    const double znorm1 = zp / L1;

    const double denom0 = log_z_over_z0 - ABLProfFun->velocity(znorm0);
    const double denom1 = log_z_over_z0 - ABLProfFun->velocity(znorm1);

    // form function to be zeroed
    const double f0 = utau0 - up * kappa_ / denom0;
    const double f1 = utau1 - up * kappa_ / denom1;

    // estimate slope, d f/d utau
    double dutau = utau1 - utau0;
    if (dutau > 0.0) {
      dutau  = std::max(1.0e-15, dutau);
    }
    else  {
      dutau = std::min(-1.0e-15, dutau);
    }
    double fPrime = (f1 - f0) / dutau;
    if (fPrime > 0.0) {
      fPrime  = std::max(1.0e-15, fPrime);
    }
    else  {
      fPrime = std::min(-1.0e-15, fPrime);
    }

    // update utau
    const double utau_tmp = utau1;
    utau1 = utau0 - f0 / fPrime;
    //enforce non-negativity of utau
    //utau1 = std::max(0.0, utau1);
    utau0 = utau_tmp;

    // check for convergence
    if ( (std::abs(f1) < tolerance_) ) {
      converged = true;
      //enforce non-negativity of utau
      utau = std::max(0.0, utau1);
      break;
    }
  }

  // report trouble
  if ( !converged ) {
    NaluEnv::self().naluOutputP0() << "Issue with ABL utau: not converged " << std::endl;
    NaluEnv::self().naluOutputP0() << up << " " << zp << " " << utau0 << " " << utau1 << std::endl;
  }


  // NOTE: SOWFA implementation contains another block to check if utau
  // converged to a small value.  In this case, a different zero to
  // the function is sought using a different algorithm.
}

//--------------------------------------------------------------------------
//-------- normalize_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeABLWallFrictionVelocityAlgorithm::normalize_nodal_fields()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // parallel assemble
  std::vector<const stk::mesh::FieldBase*> fields = {
    assembledWallArea_,
    assembledWallNormalDistance_
  };
  const std::vector<const stk::mesh::FieldBase*>& const_fields_ref = fields;
  stk::mesh::parallel_sum(bulk_data, const_fields_ref);

  // periodic assemble
  if ( realm_.hasPeriodic_) {
    const unsigned fieldSize = 1;
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    realm_.periodic_field_update(assembledWallArea_, fieldSize, bypassFieldCheck);
    realm_.periodic_field_update(assembledWallNormalDistance_, fieldSize, bypassFieldCheck);
  }

  // normalize
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    const double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    double * assembledWallNormalDistance = stk::mesh::field_data(*assembledWallNormalDistance_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      assembledWallNormalDistance[k] /= assembledWallArea[k];
    }
  }
}

} // namespace nalu
} // namespace Sierra
