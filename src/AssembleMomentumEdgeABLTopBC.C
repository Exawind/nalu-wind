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
  EquationSystem *eqSystem, std::vector<int>& grid_dims, 
  std::vector<int>& horiz_bcs, double z_sample)
  : SolverAlgorithm(realm, part, eqSystem),
  imax_(grid_dims[0]), jmax_(grid_dims[1]), kmax_(grid_dims[2]), weight_(jmax_),
  nodeMapSamp_(imax_*jmax_), nodeMapBC_(imax_*jmax_), nodeMapM1_(imax_*jmax_),
  nodeMapX0_(jmax_), indexMapSampGlobal_(imax_*jmax_), indexMapBC_(imax_*jmax_),
  sampleDistrib_(1000), displ_(1000+1), 
  horizBC_(horiz_bcs.begin(),horiz_bcs.end()), zSample_(z_sample),
  needToInitialize_(true)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "cont_velocity_bc");
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

  std::vector<double> wSamp(imax_*jmax_), uBC(imax_*jmax_), vBC(imax_*jmax_),
                      wBC(imax_*jmax_), work(imax_*jmax_), UAvg(4,0.0);
  int i, ii;
  int nx = imax_ - 1;
  int ny = jmax_ - 1;
  double nxnyInv = 1.0/((double)nx*(double)ny);

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // Determine geometrical parameters and generate a list of sample plane
  // and bc plane nodes that exist on this process.

  if( needToInitialize_ ) {
    initialize();
    needToInitialize_ = false;
  }

  // wFac and uFac are used to blend between a symmetry BC and the potential
  // flow BC near the start of the simulation.  We start with symmetry and
  // gradually switch over to potential flow.  The vertical velocity is
  // switched over first, then the horizontal velocity.

  const int timeStepCount = realm_.get_time_step_count();

  int startupSteps = 3;
  double wFac = 1.0;
  if (timeStepCount <= startupSteps) {
    wFac = (double)(timeStepCount-1)/(double)(startupSteps);
  }

  double uFac = 1.0;
  if (timeStepCount <= 2*startupSteps) {
    uFac = 0.0;
  } else if (timeStepCount <= 3*startupSteps) {
    uFac = (double)(timeStepCount-2*startupSteps-1)/(double)(startupSteps);
  }
      
//  wFac = 1.0;
//  uFac = 1.0;

// Set up for diagnostic output.

  const int myrank = bulk_data.parallel_rank();

  int printSkip = 100000;
  bool dump = false;
  FILE * outFile;
  if (timeStepCount % printSkip == 0) {
    dump = true;
    char fileName[12];
    snprintf(fileName, 12, "%4s%03i%1s%03u",
      "sol.",timeStepCount/printSkip,".",myrank);
    outFile = fopen(fileName, "w");
  }

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType* coordinates = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  // Collect the sample plane data that is held on this process.

  int nSamp = sampleDistrib_[myrank];

  for (i=0; i<nSamp; ++i) {
    double *USamp = stk::mesh::field_data(velocityNp1,nodeMapSamp_[i]);
    wSamp[i] = USamp[2];
    UAvg[0] += USamp[0]*nxnyInv;
    UAvg[1] += USamp[1]*nxnyInv;
  }

  // Find contributions to the average velocity at the x=x_min line.

  for (i=0; i<nX0_; ++i) {
    double *USamp = stk::mesh::field_data(velocityNp1,nodeMapX0_[i]);
    UAvg[2] += weight_[i]*USamp[0];
    UAvg[3] += weight_[i]*USamp[1];
  }

  // Gather the sampling plane data across all processes.

  MPI_Allgatherv(wSamp.data(), nSamp, MPI_DOUBLE, work.data(), 
                 sampleDistrib_.data(), displ_.data(), MPI_DOUBLE,
                 bulk_data.parallel());

  // Reorder the sample plane data.

  for (i=0; i<nx*ny; ++i) {
    wSamp[indexMapSampGlobal_[i]] = work[i];
  }

  // Compute the average velocity over the sampling plane.

  MPI_Allreduce(MPI_IN_PLACE, UAvg.data(), 4, MPI_DOUBLE, MPI_SUM,
                bulk_data.parallel());

  // Compute the upper boundary velocity field

  switch (horizBCType_) {
    case 0:
      potentialBCPeriodicPeriodic( wSamp, UAvg, uBC, vBC, wBC );
    break;
    case 1:
      potentialBCInflowPeriodic( wSamp, UAvg, uBC, vBC, wBC );
  }

/*
  if (dump && myrank == 0) {
    for (i=0; i<imax_*jmax_; ++i) {
      fprintf( outFile, "%5i %12.4e %12.4e %12.4e\n",i,uBC[i],vBC[i],wBC[i] );
    }
  }
*/

  // Set the boundary velocity array values.

  for (i=0; i<nBC_; ++i) {
    ii = indexMapBC_[i];
    double *uTop  = stk::mesh::field_data(*bcVelocity_, nodeMapBC_[i]);
    double *sTop  = stk::mesh::field_data(velocityNp1,  nodeMapBC_[i]);
    double *Um1   = stk::mesh::field_data(velocityNp1,  nodeMapM1_[i]);
    double *coord = stk::mesh::field_data(*coordinates, nodeMapBC_[i]);
    uTop[0] = uFac*uBC[ii] + (1.0-uFac)*Um1[0];
    uTop[1] = uFac*vBC[ii] + (1.0-uFac)*Um1[1];
    uTop[2] = wFac*wBC[ii];
    if (dump) {
      fprintf( outFile, "%12.4e%12.4e%12.4e%12.4e%12.4e%12.4e\n",
      coord[0], uTop[0], sTop[0], Um1[0], uTop[2], sTop[2] );
    }
  }

  if (dump) { fclose(outFile); }

  // Apply the boundary values as a Dirichlet condition.

  eqSystem_->linsys_->applyDirichletBCs(velocity_, bcVelocity_, partVec_, 0, 3);

}

//--------------------------------------------------------------------------
//------------------------- initialize -------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::initialize()
{

  std::vector<double> work(imax_*jmax_), zGrid(kmax_), xMin(2), xMax(2);
  std::vector< std::complex<double> > workC(imax_*(jmax_/2+1));
  std::vector<int> indexMapSamp(imax_*jmax_), indexMapX0(jmax_);

  double z0, z1, zL, nyInv;
  int i, ii, ix, iy, iz, izSample, imaxjmax, j, n, nx, ny, nz, iOff, count,
      countX0, nSamp;
  bool unique;

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nprocs = bulk_data.parallel_size();
  const int myrank = bulk_data.parallel_rank();

  VectorFieldType* coordinates = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  nx = imax_-1;
  ny = jmax_-1;
  nz = kmax_-1;
  imaxjmax = imax_*jmax_;

  // Set horizontal BC flag

  if (horizBC_[0]<0 || horizBC_[0]>1 || horizBC_[1]<0 || horizBC_[1]>1) {
    throw std::runtime_error(
      "AssembleMomentumEdgeABLTopBC: Bad user input for horizontal_bcs");
  }

  if (horizBC_[0]==0 && horizBC_[1]==0) horizBCType_ = 0;  // periodic-periodic
  if (horizBC_[0]==1 && horizBC_[1]==0) horizBCType_ = 1;  // inflow  -periodic
  if (horizBC_[0]==0 && horizBC_[1]==1) horizBCType_ = 2;  // periodic-inflow  
  if (horizBC_[0]==1 && horizBC_[1]==1) horizBCType_ = 3;  // inflow  -inflow

  // Define fft plans.

  unsigned flags=0;

  switch (horizBCType_) {
    case 0:
      planFourier2dF_ = 
      fftw_plan_dft_r2c_2d(ny, nx, work.data(),
                           reinterpret_cast<fftw_complex*>(workC.data()),flags);
      planFourier2dB_ = 
      fftw_plan_dft_c2r_2d(ny,nx,reinterpret_cast<fftw_complex*>(workC.data()),
                           work.data(), flags);
    break;
    case 1:
      planSinx_ = 
      fftw_plan_r2r_1d(nx-1, work.data(), work.data(), FFTW_RODFT00, flags);
      planCosx_ = 
      fftw_plan_r2r_1d(nx+1, work.data(), work.data(), FFTW_REDFT00, flags);
      planFourieryF_ = 
      fftw_plan_dft_r2c_1d(ny, work.data(),
                         reinterpret_cast<fftw_complex*>(workC.data()), flags);
      planFourieryB_ = 
      fftw_plan_dft_c2r_1d(ny, reinterpret_cast<fftw_complex*>(workC.data()),
                           work.data(), flags);
    break;
    default:
      printf("%s\n","BC not yet implemented");
      exit(0);
  }
/*
  planSiny_ = 
    fftw_plan_r2r_1d(ny-1, work.data(), work.data(), FFTW_RODFT00, flags);
  planCosy_ = 
    fftw_plan_r2r_1d(ny+1, work.data(), work.data(), FFTW_REDFT00, flags);
  planFourierxF_ = 
    fftw_plan_dft_r2c_1d(nx, work.data(),
                         reinterpret_cast<fftw_complex*>(workC.data()), flags);
  planFourierxB_ = 
    fftw_plan_dft_c2r_1d(nx, reinterpret_cast<fftw_complex*>(workC.data()),
                         work.data(), flags);

  planSinxSiny_ =
    fftw_plan_r2r_2d(ny-1, nx-1, work.data(), work.data(), FFTW_RODFT00,
                     FFTW_RODFT00, flags);
  planCosxSiny_ =
    fftw_plan_r2r_2d(ny-1, nx+1, work.data(), work.data(), FFTW_RODFT00,
                     FFTW_REDFT00, flags);
  planSinxCosy_ =
    fftw_plan_r2r_2d(ny+1, nx-1, work.data(), work.data(), FFTW_REDFT00,
                     FFTW_RODFT00, flags);
*/

  // Determine the vertical mesh distribution by sampling at the middle
  // of the ix=0 face.

  for (iz=0; iz<kmax_; ++iz) { zGrid[iz] = -1.0e+10; }

  ix=0;  iy=jmax_/2;
  for (iz=0; iz<kmax_; ++iz) {
    stk::mesh::EntityId IdNode = iz*imaxjmax + iy*imax_ + ix + 1;
    stk::mesh::Entity node =
      bulk_data.get_entity(stk::topology::NODE_RANK,IdNode);
    if (bulk_data.is_valid(node)) {
      double *coord = stk::mesh::field_data(*coordinates,node);
      zGrid[iz] = coord[2];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, zGrid.data(), kmax_, MPI_DOUBLE, MPI_MAX, 
                bulk_data.parallel());

  z0 = zGrid[0];
  z1 = zGrid[nz];
  zL = z1 - z0;

  // Set the default zSample_ to 90% of the domain height.

  if (zSample_ == -999.0) { zSample_ = z0 + 0.90*zL; }

  // Trap bad values for zSample_.

  if (zSample_ < z0 || zSample_ > z1) {
    throw std::runtime_error(
      "AssembleMomentumEdgeABLTopBC: zSample is not contained in the mesh");
  }
  if ((zSample_-z0) > 0.95*zL) {
    throw std::runtime_error(
    "AssembleMomentumEdgeABLTopBC: zSample is too close to the upper boundary");
  }
  if ((zSample_-z0) < 0.5*zL) {
    throw std::runtime_error(
   "AssembleMomentumEdgeABLTopBC: zSample is too far away from the upper boundary");
  }

  // Determine the grid index for the sampling plane.

  for (iz=0; iz<kmax_; ++iz) {
    if (zGrid[iz] <= zSample_ && zGrid[iz+1] > zSample_) { break; }
  }
  if ((zSample_-zGrid[iz]) > 0.5*(zGrid[iz+1]-zGrid[iz])) { iz ++; }
  izSample = iz;
  deltaZ_ = z1 - zGrid[izSample];

  // Determine the horizontal extent of the sampling plane by looking 
  // at its corner points.

  xMin[0] =  1.0e+12;  xMin[1] =  1.0e+12;
  xMax[0] = -1.0e+12;  xMax[1] = -1.0e+12;

  ix=0;  iy=0;  iz=izSample;
  stk::mesh::EntityId IdNode1 = iz*imaxjmax + iy*imax_ + ix + 1;
  stk::mesh::Entity node1 =
    bulk_data.get_entity(stk::topology::NODE_RANK,IdNode1);
  if (bulk_data.is_valid(node1)) {
    double *coord = stk::mesh::field_data(*coordinates,node1);
    xMin[0] = coord[0];
    xMin[1] = coord[1];
  }

  ix=nx;  iy=ny;  iz=izSample;
  stk::mesh::EntityId IdNode2 = iz*imaxjmax + iy*imax_ + ix + 1;
  stk::mesh::Entity node2 =
    bulk_data.get_entity(stk::topology::NODE_RANK,IdNode2);
  if (bulk_data.is_valid(node2)) {
    double *coord = stk::mesh::field_data(*coordinates,node2);
    xMax[0] = coord[0];
    xMax[1] = coord[1];
  }

  MPI_Allreduce(MPI_IN_PLACE, xMin.data(), 2, MPI_DOUBLE, MPI_MIN, 
                bulk_data.parallel());
  MPI_Allreduce(MPI_IN_PLACE, xMax.data(), 2, MPI_DOUBLE, MPI_MAX, 
                bulk_data.parallel());

  xL_ = xMax[0] - xMin[0];
  yL_ = xMax[1] - xMin[1];

  // Generate a map for the sampling plane points contained on this process.

  iOff = izSample*imaxjmax;

  count = 0;
  for (iy=0; iy<ny; ++iy) {
    for (ix=0; ix<nx; ++ix) {

      stk::mesh::EntityId IdNodeSamp = iOff + iy*imax_ + ix + 1;
      stk::mesh::Entity nodeSamp =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeSamp);

      if (bulk_data.is_valid(nodeSamp)) {
        nodeMapSamp_[count] = nodeSamp;
        indexMapSamp[count] = iy*nx + ix;
        count ++;
      }

    }
  }
  nSamp = count;

  // Generate a map for the boundary points contained on this process.

  iOff = nz*imaxjmax;

  count   = 0;
  countX0 = 0;
  for (iy=0; iy<jmax_; ++iy) {
    for (ix=0; ix<imax_; ++ix) {

      stk::mesh::EntityId IdNodeBC = iOff + iy*imax_ + ix + 1;
      stk::mesh::EntityId IdNodeM1 = IdNodeBC - imaxjmax;
      stk::mesh::Entity nodeBC =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeBC);
      stk::mesh::Entity nodeM1 =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeM1);

      if (bulk_data.is_valid(nodeBC)) {
        nodeMapBC_[ count] = nodeBC;
        nodeMapM1_[ count] = nodeM1;
        indexMapBC_[count] = iy*imax_ + ix;
        count ++;
        if (ix == 0) {
          nodeMapX0_[countX0] = nodeBC;
          indexMapX0[countX0] = iy;
          countX0 ++;
        }
      }

    }
  }
  nBC_ = count;
  nX0_ = countX0;

  // Form a global list of the x=x_min index maps.

  MPI_Allgather(&nX0_, 1, MPI_INT, sampleDistrib_.data(), 1, MPI_INT,
                bulk_data.parallel());

  displ_[0] = 0;
  for (i=1; i<nprocs+1; ++i) { 
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
  }

  MPI_Allgatherv(indexMapX0.data(), nX0_, MPI_INT, 
                 indexMapSampGlobal_.data(), sampleDistrib_.data(), 
                 displ_.data(), MPI_INT, bulk_data.parallel());

  // Eliminate redundant elements from the global x-x_min lists.

  count = nX0_;
  n = myrank;
  for (i=displ_[n]; i<displ_[n+1]; i++) {
    for (j=displ_[n+1]; j<displ_[nprocs]; ++j) {
      if (indexMapSampGlobal_[i] == indexMapSampGlobal_[j]) {
        count --;
        for (ii=i-displ_[n]; ii<count; ++ii) {
          nodeMapX0_[ii] = nodeMapX0_[ii+1];
          indexMapX0[ii] = indexMapX0[ii+1];
        }
      }
    }
  }
  nX0_ = count;

  nyInv = 1.0/(double)ny;
  for (i=0; i<nX0_; ++i) {
    if (indexMapX0[i] == 0 || indexMapX0[i] == ny) {
      weight_[i] = 0.5*nyInv;
    } else {
      weight_[i] = nyInv;
    }
  }

  // Form a global list of the sample plane index maps.

  MPI_Allgather(&nSamp, 1, MPI_INT, sampleDistrib_.data(), 1,
                MPI_INT, bulk_data.parallel());

  displ_[0] = 0;
  for (i=1; i<nprocs+1; ++i) { 
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
  }

  MPI_Allgatherv(indexMapSamp.data(), nSamp, MPI_INT, 
                 indexMapSampGlobal_.data(), sampleDistrib_.data(), 
                 displ_.data(), MPI_INT, bulk_data.parallel());

  // Eliminate redundant elements from the global index list.

  ii = 0;
  for (n=0; n<nprocs; ++n) {
    count = 0;
    for (i=displ_[n]; i<displ_[n+1]; i++) {
      unique = true;
      for (j=displ_[n+1]; j<displ_[nprocs]; ++j) {
        if (indexMapSampGlobal_[i] == indexMapSampGlobal_[j]) {
          unique = false;
          break;
        }
      }
      if (unique) {
        indexMapSampGlobal_[ii] = indexMapSampGlobal_[i];
        if (myrank == n) {
          nodeMapSamp_[count] = nodeMapSamp_[i-displ_[n]];
        }
        ii ++;
        count ++;
      }
    }
    sampleDistrib_[n] = count;
  }

  for (i=1; i<nprocs+1; ++i) {
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
  }

}


//--------------------------------------------------------------------------
//-------- potentialBCPeriodicPeriodic -------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::potentialBCPeriodicPeriodic( 
  std::vector<double>& wSamp,
  std::vector<double>& UAvg,
  std::vector<double>& uBC,
  std::vector<double>& vBC,
  std::vector<double>& wBC )
{

  double waveX, waveY, normFac, kx, ky, ky2, kMag, eFac, scale, xFac, yFac,
         zFac;
  int i, i1, i2, iOff1, iOff2, ii, j, jw, nx, ny;

  std::vector< std::complex<double> > uCoef((imax_/2+1)*jmax_),
    vCoef((imax_/2+1)*jmax_), wCoef((imax_/2+1)*jmax_);
  const double pi = std::acos(-1.0);
  const std::complex<double> iUnit(0.0,1.0);

  nx = imax_-1;
  ny = jmax_-1;

// Symmetrize wSamp.
/*
  for (j=0; j<ny; ++j) {
    wSamp[j*nx] = 0.0;
    for (i=1; i<nx/2; ++i) {
      ii = j*nx + i;
      i1 = j*nx + (nx-i);
      wSamp[ii] = 0.5*( wSamp[ii] - wSamp[i1] );
      wSamp[i1] = -wSamp[ii];
    }
  }
*/

// Forward transform of wSamp.

  fftw_execute_dft_r2c(planFourier2dF_, wSamp.data(),
                       reinterpret_cast<fftw_complex*>(wCoef.data()));

// Solve the potential flow problem.

  waveX = 2.0*pi/xL_;
  waveY = 2.0*pi/yL_;
  normFac = 1.0/((double)nx*(double)ny);

  ii = 0;
  for (j=0; j<ny; ++j) {
    jw = j;
    if (j > ny/2) { jw = jw - ny; }
    ky = waveY*(double)jw;
    ky2 = ky*ky;
    for (i=0; i<=nx/2; ++i) {
      kx = waveX*(double)i;
      kMag = std::sqrt( kx*kx + ky2 );
      eFac = std::exp(-kMag*deltaZ_)*normFac;
      scale = 1.0/(kMag+1.0e-15);
      xFac = kx*scale*eFac;
      yFac = ky*scale*eFac;
      zFac =          eFac;
      uCoef[ii] = -iUnit*xFac*wCoef[ii];
      vCoef[ii] = -iUnit*yFac*wCoef[ii];
      wCoef[ii] =        zFac*wCoef[ii];
      ii ++;
    }
  }
  wCoef[0] = 0.0;
  uCoef[0] = UAvg[0];

  // Reverse transform the solution at the upper boundary.

  fftw_execute_dft_c2r(planFourier2dB_,
                     reinterpret_cast<fftw_complex*>(uCoef.data()), uBC.data());
  fftw_execute_dft_c2r(planFourier2dB_,
                     reinterpret_cast<fftw_complex*>(vCoef.data()), vBC.data());
  fftw_execute_dft_c2r(planFourier2dB_,
                     reinterpret_cast<fftw_complex*>(wCoef.data()), wBC.data());

  // Reorganize the output arrays so they contain the periodic points
  // around the edges.

  iOff1 = 0;
  iOff2 = ny*imax_;
  uBC[iOff2+nx] = uBC[iOff1];
  vBC[iOff2+nx] = vBC[iOff1];
  wBC[iOff2+nx] = wBC[iOff1];
  for (i=nx-1; i>=0; --i) {
    i1 = iOff1 + i;
    i2 = iOff2 + i;
    uBC[i2] = uBC[i1];
    vBC[i2] = vBC[i1];
    wBC[i2] = wBC[i1];
  }

  for (j=ny-1; j>=0; --j) {
    iOff1 = j*nx;
    iOff2 = j*imax_;
    uBC[iOff2+nx] = uBC[iOff1];
    vBC[iOff2+nx] = vBC[iOff1];
    wBC[iOff2+nx] = wBC[iOff1];
    for (i=nx-1; i>=0; --i) {
      i1 = iOff1 + i;
      i2 = iOff2 + i;
      uBC[i2] = uBC[i1];
      vBC[i2] = vBC[i1];
      wBC[i2] = wBC[i1];
    }
  }

}

//--------------------------------------------------------------------------
//-------- potentialBCInflowPeriodic -------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::potentialBCInflowPeriodic( 
  std::vector<double>& wSamp,
  std::vector<double>& UAvg,
  std::vector<double>& uBC,
  std::vector<double>& vBC,
  std::vector<double>& wBC )
{

  std::vector<double> work(imax_*jmax_);
  std::vector< std::complex<double> > uCoef(imax_*(jmax_/2+1)),
    vCoef(imax_*(jmax_/2+1)), wCoef(imax_*(jmax_/2+1));

  double waveX, waveY, normFac, kx, kx2, ky, kMag, eFac, scale, xFac, yFac,
         zFac, wt, u0, v0, uInc, vInc;
  int i, i0, i1, i2, ii, iOff1, iOff2, j, j0, j1, jj, nx, ny, nxny;

  const double pi = std::acos(-1.0);
  const std::complex<double> iUnit(0.0,1.0);

  nx = imax_-1;
  ny = jmax_-1;
  nxny = nx*ny;

// Symmetrize wSamp.

/*
  for (j=0; j<ny; ++j) {
    wSamp[ j   *imax_  ] = 0.0;
    wSamp[(j+1)*imax_-1] = 0.0;
    for (i=1; i<=nx/2; ++i) {
      ii = j*imax_ + i;
      i1 = j*imax_ + (nx-i);
      wSamp[ii] = 0.5*( wSamp[ii] - wSamp[i1] );
      wSamp[i1] = -wSamp[ii];
    }
  }
*/

  // Forward transform of wSamp.  Sine transform in x, Fourier transform
  // in y.  Note that the data is transposed between the x and y transforms.

  for (j=0; j<ny; ++j) {
    i0 = j*nx;
    fftw_execute_r2r(planSinx_, &wSamp[i0+1], &wSamp[i0+1]);
    work[j] = 0.0;
    for (i=1; i<nx; ++i) {
      ii = i*ny + j;
      work[ii] = wSamp[i0+i];
    }
  }
  for (i=0; i<nx; ++i) {
    j0 = i*ny;
    j1 = i*(ny/2+1);
    fftw_execute_dft_r2c(planFourieryF_, &work[j0],
                         reinterpret_cast<fftw_complex*>(&wCoef[j1]));
  }

  // Solve the potential flow problem.  u0 and v0 are the average velocity
  // components at the x=x_min edge.

  waveX =     pi/xL_;
  waveY = 2.0*pi/yL_;
  normFac = 1.0/((double)(2*nx)*(double)ny);

  u0 = 0;
  v0 = 0;
  jj = 0;
  for (i=0; i<nx; ++i) {
    kx = waveX*(double)i;
    kx2 = kx*kx;
    wt = 2.0;
    if (i==0 || i==nx) { wt = 1.0; }
    for (j=0; j<=ny/2; ++j) {
      ky = waveY*(double)j;
      kMag = std::sqrt( kx2 + ky*ky );
      eFac = std::exp(-kMag*deltaZ_)*normFac;
      scale = 1.0/(kMag+1.0e-15);
      xFac = kx*scale*eFac;
      yFac = ky*scale*eFac;
      zFac =          eFac;
      uCoef[jj] =       -xFac*wCoef[jj];
      vCoef[jj] = -iUnit*yFac*wCoef[jj];
      wCoef[jj] =        zFac*wCoef[jj];
      if (j == 0) {
        u0 += wt*real(uCoef[jj]);
        v0 += wt*real(vCoef[jj]);
      }
      jj ++;
    }
  }

  // Reverse transform the solution at the upper boundary.  Fourier transform
  // in y, either sine or cosine transform in x.  Note that the data is 
  // transposed between the y and x transforms.

  for (i=0; i<nx; ++i) {
    j0 = i*ny;
    j1 = i*(ny/2+1);
    fftw_execute_dft_c2r(planFourieryB_,
                        reinterpret_cast<fftw_complex*>(&uCoef[j1]), &work[j0]);
  }
  for (j=0; j<ny; ++j) {
    i0 = j*imax_;
    for (i=0; i<nx; ++i) {
      ii = i*ny + j;
      uBC[i0+i] = work[ii];
    }
    fftw_execute_r2r(planCosx_, &uBC[i0], &uBC[i0]);
  }

  for (i=0; i<nx; ++i) {
    j0 = i*ny;
    j1 = i*(ny/2+1);
    fftw_execute_dft_c2r(planFourieryB_,
                        reinterpret_cast<fftw_complex*>(&vCoef[j1]), &work[j0]);
  }
  for (j=0; j<ny; ++j) {
    i0 = j*imax_;
    for (i=0; i<nx; ++i) {
      ii = i*ny + j;
      vBC[i0+i] = work[ii];
    }
    fftw_execute_r2r(planSinx_, &vBC[i0], &vBC[i0]);
  }

  for (i=0; i<nx; ++i) {
    j0 = i*ny;
    j1 = i*(ny/2+1);
    fftw_execute_dft_c2r(planFourieryB_,
                        reinterpret_cast<fftw_complex*>(&wCoef[j1]), &work[j0]);
  }
  for (j=0; j<ny; ++j) {
    i0 = j*imax_;
    for (i=0; i<nx; ++i) {
      ii = i*ny + j;
      wBC[i0+i] = work[ii];
    }
    fftw_execute_r2r(planSinx_, &wBC[i0+1], &wBC[i0+1]);
    wBC[i0   ] = 0.0;
    wBC[i0+nx] = 0.0;
  }

  // Adjust the u and v mean velocity so that the velocity computed at the
  // x=x_min edge matches the inflow velocity.

  uInc = UAvg[2] - u0;
  vInc = UAvg[3] - v0;
  for (i=0; i<imax_*ny; ++i) {
    uBC[i] += uInc;
    vBC[i] += vInc;
  }

  // Enforce periodicity in y.

  iOff1 = 0;
  iOff2 = ny*imax_;
  for (i=0; i<imax_; ++i) {
    i1 = iOff1 + i;
    i2 = iOff2 + i;
    uBC[i2] = uBC[i1];
    vBC[i2] = vBC[i1];
    wBC[i2] = wBC[i1];
  }

}

} // namespace nalu
} // namespace Sierra
