/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumEdgeABLTopBC.h>
#include <SolverAlgorithm.h>
#include <EquationSystem.h>
#include <LinearSystem.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// fftw
#include <complex.h> // Must proceed fftw3.h in order to get native c complex
#include <fftw3.h>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleMomentumEdgeABLTopBC - Top boundary condition for inflow/outflow
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumEdgeABLTopBC::AssembleMomentumEdgeABLTopBC(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem, std::vector<int>& grid_dims)
  : SolverAlgorithm(realm, part, eqSystem),
  imax_(grid_dims[0]), jmax_(grid_dims[1]), kmax_(grid_dims[2])
//wSamp_(imax_*jmax_), uBC_(imax_*jmax_), uBC_(imax_*jmax_), wBC_(imax_*jmax_),
//uCoef_(2*(imax_/2+1)*jmax_), vCoef_(2*(imax_/2+1)*jmax_), wCoef_(2*(imax_/2+1)*jmax_)
{
  wSamp_ = new double[jmax_*imax_]; 
  uBC_   = new double[jmax_*imax_];
  vBC_   = new double[jmax_*imax_];
  wBC_   = new double[jmax_*imax_];
  uCoef_ = new fftw_complex[jmax_*(imax_/2+1)];
  vCoef_ = new fftw_complex[jmax_*(imax_/2+1)];
  wCoef_ = new fftw_complex[jmax_*(imax_/2+1)];
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "cont_velocity_bc");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::initialize_connectivity()
{
  eqSystem_->linsys_->buildDirichletNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  double x0, y0, z0, x1, y1, z1, xL, yL, deltaZ;
  int nx = imax_ - 1;
  int ny = jmax_ - 1;

  const int timeStepCount = realm_.get_time_step_count();

  int printSkip = 1;
  bool dump = false;
  FILE * outFile;
  if( timeStepCount % printSkip == 0 ) {
    dump = true;
    char fileName[8];
    snprintf(fileName, 8, "%4s%03i","sol.",timeStepCount/printSkip);
    outFile = fopen( fileName, "w" );
  }

  //space for LHS/RHS; nodesPerFace*nDim*nodesPerFace*nDim and nodesPerFace*nDim

  // deal with state
  VectorFieldType* coordinates = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  // wFac and uFac are used to blend between a symmetry BC and the potential
  // flow BC near the start of the simulation.  We start with symmetry and
  // gradually switch over to potential flow.  The vertical velocity is
  // switched over first, then the horizontal velocity.

  int startupSteps = 3;
  double wFac = 1.0;
  if( timeStepCount <= startupSteps ) {
    wFac = (double)(timeStepCount-1)/(double)(startupSteps);
  }

  double uFac = 1.0;
  if( timeStepCount <= 2*startupSteps ) {
    uFac = 0.0;
  } else if( timeStepCount <= 3*startupSteps ) {
    uFac = (double)(timeStepCount-2*startupSteps-1)/(double)(startupSteps);
  }
      
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get node
      stk::mesh::Entity nodeBC = b[k];

      stk::mesh::EntityId IdNodeBC = bulk_data.identifier(nodeBC);
      stk::mesh::EntityId IdNodeTmp = IdNodeBC-1;
      int iz = (IdNodeTmp)/(imax_*jmax_);
      IdNodeTmp %= imax_*jmax_;
      int iy  = (IdNodeTmp/imax_);
      int ix  = IdNodeTmp % imax_;
      int ixm = ix % nx;
      int iym = iy % ny;

      stk::mesh::EntityId IdNodeSamp = (iz-12)*imax_*jmax_ + iy*imax_ + ix + 1;
      stk::mesh::Entity nodeSamp = 
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeSamp);

      double *coordBC   = stk::mesh::field_data(*coordinates, nodeBC);
      double *coordSamp = stk::mesh::field_data(*coordinates, nodeSamp);
      double *USamp     = stk::mesh::field_data(velocityNp1,  nodeSamp);

      if( ix == 0 && iy == 0 ) {
        x0 = coordSamp[0];
        y0 = coordSamp[1];
        z0 = coordSamp[2];
      }

      if( ix == nx && iy == ny ) {
        x1 = coordSamp[0];
        y1 = coordSamp[1];
        z1 = coordBC[  2];
      }

      if( ix < imax_ && iy < jmax_ ) {
        wSamp_[iym*nx+ixm] = USamp[2];
      }

    }
  }

  xL = x1 - x0;
  yL = y1 - y0;
  deltaZ  = z1 - z0;

  // Compute the upper boundary velocity field

  AssembleMomentumEdgeABLTopBC::potentialBCPeriodicPeriodic( 
    wSamp_, uCoef_, vCoef_, wCoef_, uBC_, vBC_, wBC_, xL, yL, deltaZ, nx, ny );

//  double uInc = 1.0 - uBC_[0];
  double uInc = 1.0;
  for( int j=0; j<ny; ++j ) {
    for( int i=0; i<nx; ++i ) {
      int ii = j*nx + i;
      uBC_[ii] += uInc;
    }
  }

  // Now set the boundary velocity array values

  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get node
      stk::mesh::Entity nodeBC = b[k];

      stk::mesh::EntityId IdNodeBC = bulk_data.identifier(nodeBC);
      stk::mesh::EntityId IdNodeTmp = IdNodeBC-1;
      int iz = (IdNodeTmp)/(imax_*jmax_);
      IdNodeTmp %= imax_*jmax_;
      int iy  = (IdNodeTmp/imax_);
      int ix  = IdNodeTmp % imax_;
      int ixm = ix % nx;
      int iym = iy % ny;

      stk::mesh::EntityId IdNodem1 = (iz-1)*imax_*jmax_ + iy*imax_ + ix + 1;
      stk::mesh::Entity nodem1 =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodem1);

      double *coord = stk::mesh::field_data(*coordinates, nodeBC);
      double *uTop  = stk::mesh::field_data(*bcVelocity_, nodeBC);
      double *sTop  = stk::mesh::field_data(velocityNp1,  nodeBC);
      double *Um1   = stk::mesh::field_data(velocityNp1,  nodem1);

      int ii = iym*nx   + ixm;
      uTop[0] = uFac*uBC_[ii] + (1.0-uFac)*Um1[0];
//      sTop[0] = uTop[0];
      uTop[1] = wFac*vBC_[ii];
//      sTop[1] = uTop[1];
      uTop[2] = wFac*wBC_[ii];
//      sTop[2] = uTop[2];

      if( dump ) {
//        int i1 = iy*imax_ + ix;
//        fprintf( outFile, "%5i %5i %3i %3i %3i %3i %12.4e %12.4e %12.4e\n",
//        i1, ii, ix, ixm, iy, iym, wSamp_[ii], uBC_[ii], wBC_[ii] );
        fprintf( outFile, "%12.4e%12.4e%12.4e%12.4e%12.4e%12.4e\n",
        coord[0], uTop[0], sTop[0], uTop[2], sTop[2], wSamp_[ii] );
      }

    }
  }

  if( dump ) { fclose(outFile); }

  eqSystem_->linsys_->applyDirichletBCs(
  velocity_,
  bcVelocity_,
  partVec_,
  0,
  3);

}

//--------------------------------------------------------------------------
//-------- potentialBCPeriodicPeriodic -------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::potentialBCPeriodicPeriodic( 
  double *wSamp_,
  fftw_complex *uCoef_,
  fftw_complex *vCoef_,
  fftw_complex *wCoef_,
  double *uBC_,
  double *vBC_,
  double *wBC_,
  double xL,
  double yL,
  double deltaZ,
  int nx,
  int  ny )
{
  double pi, waveX, waveY, norm, kx, ky, ky2, kMag, eFac, scale, xFac, yFac,
         zFac;
  int i, i1, ii, j, jw;

// Symmetrize wSamp.

  for( j=0; j<ny; ++j ) {
    wSamp_[j*nx] = 0.0;
    for( i=1; i<nx/2; ++i ) {
      ii = j*nx + i;
      i1 = j*nx + (nx-i);
      wSamp_[ii] = 0.5*( wSamp_[ii] - wSamp_[i1] );
      wSamp_[i1] = -wSamp_[ii];
    }
  }

// Set up for FFT.

  pi = std::acos(-1.0);
  waveX = 2.0*pi/xL;
  waveY = 2.0*pi/yL;
  norm = 1.0/((double)nx*(double)ny);

  unsigned flags=0;
  fftw_plan plan_wf = fftw_plan_dft_r2c_2d(ny, nx, wSamp_, wCoef_, flags);
  fftw_plan plan_ub = fftw_plan_dft_c2r_2d(ny, nx, uCoef_, uBC_,   flags);
  fftw_plan plan_vb = fftw_plan_dft_c2r_2d(ny, nx, vCoef_, vBC_,   flags);
  fftw_plan plan_wb = fftw_plan_dft_c2r_2d(ny, nx, wCoef_, wBC_,   flags);

// Solve for potential flow at the upper boundary using FFT.

  fftw_execute(plan_wf);

  ii = 0;
  for( j=0; j<ny; ++j ) {
    jw = j;
    if( j > ny/2 ) { jw = jw - ny; }
    ky = waveY*(double)jw;
    ky2 = ky*ky;
    for( i=0; i<=nx/2; ++i ) {
      kx = waveX*(double)i;
      kMag = std::sqrt( kx*kx + ky2 );
      eFac = std::exp(-kMag*deltaZ)*norm;
      scale = 1.0/(kMag+1.0e-15);
      xFac = kx*scale*eFac;
      yFac = ky*scale*eFac;
      zFac =          eFac;
      uCoef_[ii][0] =  xFac*wCoef_[ii][1];
      uCoef_[ii][1] = -xFac*wCoef_[ii][0];
      vCoef_[ii][0] =  yFac*wCoef_[ii][1];
      vCoef_[ii][1] = -yFac*wCoef_[ii][0];
      wCoef_[ii][0] =  zFac*wCoef_[ii][0];
      wCoef_[ii][1] =  zFac*wCoef_[ii][1];
      ii ++;
    }
  }
  wCoef_[0][0] = 0.0;

  fftw_execute(plan_ub);
  fftw_execute(plan_vb);
  fftw_execute(plan_wb);

}

} // namespace nalu
} // namespace Sierra
