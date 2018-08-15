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
  imax_(grid_dims[0]), jmax_(grid_dims[1]), kmax_(grid_dims[2]),
  wSamp_(imax_*jmax_), uBC_(imax_*jmax_), vBC_(imax_*jmax_), wBC_(imax_*jmax_),
  uCoef_((imax_/2+1)*jmax_), vCoef_((imax_/2+1)*jmax_), 
  wCoef_((imax_/2+1)*jmax_)
{
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

  double x0, y0, z0, x1, y1, z1, xL, yL, deltaZ, uAvg, vAvg;
  int ix, iy, iz, ixm, iym, nxny, iOff, count;
  int nx = imax_ - 1;
  int ny = jmax_ - 1;
  int nz = kmax_ - 1;

  const int timeStepCount = realm_.get_time_step_count();
  const unsigned theRank = NaluEnv::self().parallel_rank();

  int printSkip = 1;
  bool dump = false;
  FILE * outFile;
  if( timeStepCount % printSkip == 0 ) {
    dump = true;
    char fileName[12];
    snprintf(fileName, 12, "%4s%03i%1s%03u",
      "sol.",timeStepCount/printSkip,".",theRank);
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
      
//  memset(wSamp_.data(), -1.0e+12, imax_*jmax_); // This does not work, why?
  for(int i=0; i<imax_*jmax_; ++i) { wSamp_[i]=-1.0e+12; }

  double sumU = 0.0;
  double sumV = 0.0;
  nxny = imax_*jmax_;
  iOff = (nz-12)*nxny;

  count=0;
  for( iy=0; iy<ny; ++iy ) {
    for( ix=0; ix<nx; ++ix ) {

      stk::mesh::EntityId IdNodeSamp = iOff + iy*imax_ + ix + 1;
      stk::mesh::Entity nodeSamp = 
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeSamp);

      if ( bulk_data.is_valid(nodeSamp) ) {

        double *USamp = stk::mesh::field_data(velocityNp1,nodeSamp);
        double *coord = stk::mesh::field_data(*coordinates,nodeSamp);

        sumU = sumU + USamp[0];
        sumV = sumV + USamp[1];
        wSamp_[iy*nx+ix] = USamp[2];
        z0 = coord[2];

//      fprintf( outFile, "%5llu %5u %5i %3i %3i %12.4e\n",IdNodeSamp, nodeSamp, ii, ix, iy, wSamp_[ii] );

        count ++;

      }

//      int ii = iy*nx+ix;
//      fprintf( outFile, "%5llu %5u %5i %3i %3i %12.4e\n",IdNodeSamp, nodeSamp, ii, ix, iy, wSamp_[ii] );

    }
  }

  x0 = -5.0;
  x1 =  5.0;
  y0 = -1.0/24.0;
  y1 =  1.0/24.0;
  z1 =  1.0;
  xL = x1 - x0;
  yL = y1 - y0;
  deltaZ  = z1 - z0;
  wSamp_[nx*ny  ] = sumU/((double)nx*(double)ny);
  wSamp_[nx*ny+1] = sumV/((double)nx*(double)ny);

//printf("%u %i\n",theRank,count);

  // Collect the sampling plane data across all processors

  MPI_Allreduce(MPI_IN_PLACE, wSamp_.data(), nx*ny+2,
                MPI_DOUBLE, MPI_MAX, bulk_data.parallel());

  uAvg = wSamp_[nx*ny  ];
  vAvg = wSamp_[nx*ny+1];
  uAvg = 1.0;

/*
  for( iy=0; iy<ny; ++iy ) {
    for( ix=0; ix<nx; ++ix ) {
      int ii = iy*nx + ix;
      fprintf( outFile, "%5i %3i %3i %12.4e\n",ii, ix, iy, wSamp_[ii] );
    }
  }
*/

  // Compute the upper boundary velocity field

  AssembleMomentumEdgeABLTopBC::potentialBCPeriodicPeriodic( &wSamp_[0],
    &uCoef_[0], &vCoef_[0], &wCoef_[0], &uBC_[0], &vBC_[0], &wBC_[0],
    xL, yL, deltaZ, uAvg, nx, ny );

  // Now set the boundary velocity array values

  iOff = nz*nxny;

  count = 0;
  for( iy=0; iy<jmax_; ++iy ) {
    iym = iy % ny;
    for( ix=0; ix<imax_; ++ix ) {
      ixm = ix % nx;

      stk::mesh::EntityId IdNodeBC = iOff + iy*imax_ + ix + 1;
      stk::mesh::Entity nodeBC =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeBC);

      if ( bulk_data.is_valid(nodeBC) ) {

        stk::mesh::EntityId IdNodem1 = IdNodeBC - nxny;
        stk::mesh::Entity nodem1 =
          bulk_data.get_entity(stk::topology::NODE_RANK,IdNodem1);

        double *coord = stk::mesh::field_data(*coordinates, nodeBC);
        double *uTop  = stk::mesh::field_data(*bcVelocity_, nodeBC);
        double *sTop  = stk::mesh::field_data(velocityNp1,  nodeBC);
        double *Um1   = stk::mesh::field_data(velocityNp1,  nodem1);

        int ii = iym*nx   + ixm;
        uTop[0] = uFac*uBC_[ii] + (1.0-uFac)*Um1[0];
        uTop[1] = wFac*vBC_[ii];
        uTop[2] = wFac*wBC_[ii];
//        uTop[0] = Um1[0];
//        uTop[1] = 0.0;
//        uTop[2] = wBC_[ii];

        if( dump ) {
//          int i1 = iy*imax_ + ix;
//          fprintf( outFile, "%5i %5i %3i %3i %3i %3i %12.4e %12.4e %12.4e\n",
//          i1, ii, ix, ixm, iy, iym, wSamp_[ii], uBC_[ii], wBC_[ii] );
          fprintf( outFile, "%12.4e%12.4e%12.4e%12.4e%12.4e%12.4e\n",
          coord[0], uTop[0], sTop[0], uTop[2], sTop[2], wSamp_[ii] );
        }

        count ++;

      }  

    }
  }

//printf("%u %i\n",theRank,count);

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
  std::complex<double> *uCoef_,
  std::complex<double> *vCoef_,
  std::complex<double> *wCoef_,
  double *uBC_,
  double *vBC_,
  double *wBC_,
  double xL,
  double yL,
  double deltaZ,
  double uAvg,
  int nx,
  int  ny )
{
  double waveX, waveY, normFac, kx, ky, ky2, kMag, eFac, scale, xFac, yFac,
         zFac;
  int i, i1, ii, j, jw;

  const double pi = std::acos(-1.0);
  const std::complex<double> iUnit(0.0,1.0);

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

  waveX = 2.0*pi/xL;
  waveY = 2.0*pi/yL;
  normFac = 1.0/((double)nx*(double)ny);

  unsigned flags=0;
  fftw_plan plan_f = fftw_plan_dft_r2c_2d(ny, nx,
    &wSamp_[0], reinterpret_cast<fftw_complex*>(&wCoef_[0]), flags);
  fftw_plan plan_b = fftw_plan_dft_c2r_2d(ny, nx,
    reinterpret_cast<fftw_complex*>(&wCoef_[0]), &wBC_[0],   flags);

// Solve for potential flow at the upper boundary using FFT.

  fftw_execute(plan_f);

  ii = 0;
  for( j=0; j<ny; ++j ) {
    jw = j;
    if( j > ny/2 ) { jw = jw - ny; }
    ky = waveY*(double)jw;
    ky2 = ky*ky;
    for( i=0; i<=nx/2; ++i ) {
      kx = waveX*(double)i;
      kMag = std::sqrt( kx*kx + ky2 );
      eFac = std::exp(-kMag*deltaZ)*normFac;
      scale = 1.0/(kMag+1.0e-15);
      xFac = kx*scale*eFac;
      yFac = ky*scale*eFac;
      zFac =          eFac;
      uCoef_[ii] = -iUnit*xFac*wCoef_[ii];
      vCoef_[ii] = -iUnit*yFac*wCoef_[ii];
      wCoef_[ii] =        zFac*wCoef_[ii];
      ii ++;
    }
  }
  wCoef_[0] = 0.0;
  uCoef_[0] = uAvg;

  fftw_execute(plan_b);
  fftw_execute_dft_c2r(plan_b,
    reinterpret_cast<fftw_complex*>(&uCoef_[0]), &uBC_[0]);
  fftw_execute_dft_c2r(plan_b,
    reinterpret_cast<fftw_complex*>(&vCoef_[0]), &vBC_[0]);

}

} // namespace nalu
} // namespace Sierra
