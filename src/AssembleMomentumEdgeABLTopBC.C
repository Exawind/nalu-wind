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
  wCoef_((imax_/2+1)*jmax_),
  nodeMapSamp_(imax_*jmax_), nodeMapBC_(imax_*jmax_), nodeMapM1_(imax_*jmax_),
  indexMapSamp_(imax_*jmax_), indexMapBC_(imax_*jmax_),
  indexMapSampGlobal_(imax_*jmax_), sampleDistrib_(1000), displ_(1000+1),
  zSample_(0.85), needToInitialize_(true)
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

  std::vector<double> work(imax_*jmax_);
  std::vector<double> UAvg(2,0.0);
  int i, ii;
  int nx = imax_ - 1;
  int ny = jmax_ - 1;
  int nxny = nx*ny;

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // Determine geometrical parameters and generate a list of sample plane
  // and bc plane nodes that exist on this process.

  if( needToInitialize_ ) {
    initialize( imax_, jmax_, kmax_, zSample_,
                &xL_, &yL_, &deltaZ_, 
                nodeMapSamp_.data(), nodeMapBC_.data(), nodeMapM1_.data(),
                indexMapSamp_.data(), indexMapBC_.data(), 
                indexMapSampGlobal_.data(), sampleDistrib_.data(),
                displ_.data(), &nSamp_, &nBC_ );
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
      
// Set up for diagnostic output.

  const int nprocs = bulk_data.parallel_size();
  const int myrank = bulk_data.parallel_rank();

  int printSkip = 1;
  bool dump = false;
  FILE * outFile;
  if (timeStepCount % printSkip == 0) {
    dump = true;
    char fileName[12];
    snprintf(fileName, 12, "%4s%03i%1s%03u",
      "sol.",timeStepCount/printSkip,".",myrank);
    outFile = fopen(fileName, "w");
  }

/*
  if(myrank==0) {
    for (i=0; i<nprocs; ++i) {
      printf("%3i %3i %3i\n",i,sampleDistrib_[i],displ_[i]);
      for (int j=displ_[i]; j<displ_[i]+sampleDistrib_[i]; ++j) {
        printf("%3i %3i\n",j,indexMapSampGlobal_[j]);
      }
    }
  }
*/

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  VectorFieldType* coordinates = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  // Collect the sample plane data that is held on this process.

  for (i=0; i<nSamp_; ++i) {
    double *USamp = stk::mesh::field_data(velocityNp1,nodeMapSamp_[i]);
//    wSamp_[indexMapSamp_[i]] = USamp[2];
    wSamp_[i] = USamp[2];
    UAvg[0] += USamp[0];
    UAvg[1] += USamp[1];
  }

  UAvg[0] /= (double)nxny;
  UAvg[1] /= (double)nxny;

  // Gather the sampling plane data across all processes.

  MPI_Allgatherv(wSamp_.data(), nSamp_, MPI_DOUBLE, work.data(), 
                 sampleDistrib_.data(), displ_.data(), MPI_DOUBLE,
                 bulk_data.parallel());

  // Reorder the sample plane data.

  for (i=0; i<nx*ny; ++i) {
    wSamp_[indexMapSampGlobal_[i]] = work[i];
  }

/*
  if (dump) {
    for (i=0; i<nx*ny; ++i) {
      ii = indexMapSampGlobal_[i];
      fprintf( outFile, "%4i%4i%12.4e\n", i, ii, wSamp_[i]);
    }
  }
*/

  // Compute the average velocity over the sampling plane.

  MPI_Allreduce(MPI_IN_PLACE, UAvg.data(), 2, MPI_DOUBLE, MPI_SUM,
                bulk_data.parallel());
  UAvg[0]=1.0;  UAvg[1]=0.0;

  // Compute the upper boundary velocity field

  potentialBCPeriodicPeriodic( &wSamp_[0], xL_, yL_, deltaZ_, &UAvg[0], imax_,
                               jmax_, &uBC_[0], &vBC_[0], &wBC_[0],
                               &uCoef_[0], &vCoef_[0], &wCoef_[0] );

  // Set the boundary velocity array values.

  for (i=0; i<nBC_; ++i) {
    ii = indexMapBC_[i];
    double *uTop  = stk::mesh::field_data(*bcVelocity_, nodeMapBC_[i]);
    double *sTop  = stk::mesh::field_data(velocityNp1,  nodeMapBC_[i]);
    double *Um1   = stk::mesh::field_data(velocityNp1,  nodeMapM1_[i]);
    double *coord = stk::mesh::field_data(*coordinates, nodeMapBC_[i]);
    uTop[0] = uFac*uBC_[ii] + (1.0-uFac)*Um1[0];
    uTop[1] = wFac*vBC_[ii];
    uTop[2] = wFac*wBC_[ii];
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
AssembleMomentumEdgeABLTopBC::initialize(
  int imax_,
  int jmax_,
  int kmax_,
  double zSample_,
  double *xL_,
  double *yL_,
  double *deltaZ_,
  stk::mesh::Entity *nodeMapSamp_,
  stk::mesh::Entity *nodeMapBC_,
  stk::mesh::Entity *nodeMapM1_,
  int *indexMapSamp_,
  int *indexMapBC_,
  int *indexMapGlobal_,
  int *sampleDistrib_,
  int *displ_,
  int *nSamp_,
  int *nBC_)
{

  double z0, z1, zL, zGrid[kmax_], xMin[2], xMax[2];
  int i, ii, ix, iy, iz, izSample, imaxjmax, j, n, nx, ny, nz, iOff, count;
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

  MPI_Allreduce(MPI_IN_PLACE, zGrid, kmax_, MPI_DOUBLE, MPI_MAX, 
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
  *deltaZ_ = z1 - zGrid[izSample];

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

  MPI_Allreduce(MPI_IN_PLACE, xMin, 2, MPI_DOUBLE, MPI_MIN, 
                bulk_data.parallel());
  MPI_Allreduce(MPI_IN_PLACE, xMax, 2, MPI_DOUBLE, MPI_MAX, 
                bulk_data.parallel());

  *xL_ = xMax[0] - xMin[0];
  *yL_ = xMax[1] - xMin[1];

  // Generate a map for the sampling plane points contained on this process.

  iOff = izSample*imaxjmax;

  count = 0;
  for (iy=0; iy<ny; ++iy) {
    for (ix=0; ix<nx; ++ix) {

      stk::mesh::EntityId IdNodeSamp = iOff + iy*imax_ + ix + 1;
      stk::mesh::Entity nodeSamp =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeSamp);

      if (bulk_data.is_valid(nodeSamp)) {
        nodeMapSamp_[ count] = nodeSamp;
        indexMapSamp_[count] = iy*nx + ix;
        count ++;
      }

    }
  }
  *nSamp_ = count;

  // Generate a map for the boundary points contained on this process.

  iOff = nz*imaxjmax;

  count = 0;
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
      }

    }
  }
  *nBC_ = count;

  // Form a master list of index maps.

  MPI_Allgather(nSamp_, 1, MPI_INT, sampleDistrib_, 1,
                MPI_INT, bulk_data.parallel());

  displ_[0] = 0;
  for (i=1; i<nprocs+1; ++i) { 
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
  }

//  printf("%s %i %i\n","before ",myrank,sampleDistrib_[myrank]);

  MPI_Allgatherv(indexMapSamp_, *nSamp_, MPI_INT, 
                 indexMapSampGlobal_.data(), sampleDistrib_, 
                 displ_, MPI_INT, bulk_data.parallel());

  // Eliminate redundant elements from the master list.

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
  *nSamp_ = sampleDistrib_[myrank];

  for (i=1; i<nprocs+1; ++i) {
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
  }

//  printf("%s %i %i\n","after ",myrank,sampleDistrib_[myrank]);

}


//--------------------------------------------------------------------------
//-------- potentialBCPeriodicPeriodic -------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::potentialBCPeriodicPeriodic( 
  double *wSamp_,
  double xL_,
  double yL_,
  double deltaZ_,
  double *UAvg,
  int imax_,
  int jmax_,
  double *uBC_,
  double *vBC_,
  double *wBC_,
  std::complex<double> *uCoef_,
  std::complex<double> *vCoef_,
  std::complex<double> *wCoef_ )
{

  double waveX, waveY, normFac, kx, ky, ky2, kMag, eFac, scale, xFac, yFac,
         zFac;
  int i, i1, i2, iOff1, iOff2, ii, j, jw, nx, ny;

  const double pi = std::acos(-1.0);
  const std::complex<double> iUnit(0.0,1.0);

  nx = imax_-1;
  ny = jmax_-1;

// Symmetrize wSamp.

  for (j=0; j<ny; ++j) {
    wSamp_[j*nx] = 0.0;
    for (i=1; i<nx/2; ++i) {
      ii = j*nx + i;
      i1 = j*nx + (nx-i);
      wSamp_[ii] = 0.5*( wSamp_[ii] - wSamp_[i1] );
      wSamp_[i1] = -wSamp_[ii];
    }
  }

// Set up for FFT.

  waveX = 2.0*pi/xL_;
  waveY = 2.0*pi/yL_;
  normFac = 1.0/((double)nx*(double)ny);

  unsigned flags=0;
  fftw_plan plan_f = fftw_plan_dft_r2c_2d(ny, nx,
    &wSamp_[0], reinterpret_cast<fftw_complex*>(&wCoef_[0]), flags);
  fftw_plan plan_b = fftw_plan_dft_c2r_2d(ny, nx,
    reinterpret_cast<fftw_complex*>(&wCoef_[0]), &wBC_[0],   flags);

// Forward transform of wSamp.

  fftw_execute(plan_f);

// Solve the potential flow problem.

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
      uCoef_[ii] = -iUnit*xFac*wCoef_[ii];
      vCoef_[ii] = -iUnit*yFac*wCoef_[ii];
      wCoef_[ii] =        zFac*wCoef_[ii];
      ii ++;
    }
  }
  wCoef_[0] = 0.0;
  uCoef_[0] = UAvg[0];

  // Reverse transform the solution at the upper boundary.

  fftw_execute(plan_b);
  fftw_execute_dft_c2r(plan_b,
    reinterpret_cast<fftw_complex*>(&uCoef_[0]), &uBC_[0]);
  fftw_execute_dft_c2r(plan_b,
    reinterpret_cast<fftw_complex*>(&vCoef_[0]), &vBC_[0]);

  // Reorganize the output arrays so they contain the periodic points
  // around the edges.

  iOff1 = 0;
  iOff2 = ny*imax_;
  for (i=0; i<nx; ++i) {
    i1 = iOff1 + i;
    i2 = iOff2 + i;
    uBC_[i2] = uBC_[i1];
    vBC_[i2] = vBC_[i1];
    wBC_[i2] = wBC_[i1];
  }
  uBC_[iOff2+nx] = uBC_[iOff1+0];
  vBC_[iOff2+nx] = vBC_[iOff1+0];
  wBC_[iOff2+nx] = wBC_[iOff1+0];

  for (j=ny-1; j>0; --j) {
    iOff1 = j*nx;
    iOff2 = j*imax_;
    for (i=0; i<nx; ++i) {
      i1 = iOff1 + i;
      i2 = iOff2 + i;
      uBC_[i2] = uBC_[i1];
      vBC_[i2] = vBC_[i1];
      wBC_[i2] = wBC_[i1];
    }
    uBC_[iOff2+nx] = uBC_[iOff1+0];
    vBC_[iOff2+nx] = vBC_[iOff1+0];
    wBC_[iOff2+nx] = wBC_[iOff1+0];
  }

}

} // namespace nalu
} // namespace Sierra
