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
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  std::vector<int>& grid_dims,
  std::vector<int>& horiz_bcs,
  double z_sample)
  : SolverAlgorithm(realm, part, eqSystem),
    imax_(grid_dims[0]),
    jmax_(grid_dims[1]),
    kmax_(grid_dims[2]),
    xInflowWeight_(jmax_),
    yInflowWeight_(imax_),
    nodeMapSamp_(imax_ * jmax_),
    nodeMapBC_(imax_ * jmax_),
    nodeMapM1_(imax_ * jmax_),
    nodeMapXInflow_(jmax_),
    nodeMapYInflow_(imax_),
    indexMapSampGlobal_(imax_ * jmax_),
    indexMapBC_(imax_ * jmax_),
    sampleDistrib_(realm.bulk_data().parallel_size()),
    displ_(realm.bulk_data().parallel_size()+1),
    horizBC_(horiz_bcs.begin(), horiz_bcs.end()),
    zSample_(z_sample),
    needToInitialize_(true)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "cont_velocity_bc");
}

AssembleMomentumEdgeABLTopBC::~AssembleMomentumEdgeABLTopBC()
{
  switch (horizBCType_) {
  case 0:
    fftw_destroy_plan(planFourier2dF_);
    fftw_destroy_plan(planFourier2dB_);
    break;

  case 1:
    fftw_destroy_plan(planSinx_);
    fftw_destroy_plan(planCosx_);
    fftw_destroy_plan(planFourieryF_);
    fftw_destroy_plan(planFourieryB_);
    break;

  case 3:
    fftw_destroy_plan(planSinx_);
    fftw_destroy_plan(planCosx_);
    fftw_destroy_plan(planSiny_);
    fftw_destroy_plan(planCosy_);
    break;
  }

  fftw_cleanup();
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
                      wBC(imax_*jmax_), work(imax_*jmax_), UAvg(9,0.0);
  int i, j, ii;
  int nx = imax_ - 1;
  int ny = jmax_ - 1;
  double nxnyInv = 1.0/((double)nx*(double)ny);

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  const int myrank = bulk_data.parallel_rank();

  // Determine geometrical parameters and generate a list of sample plane
  // and bc plane nodes that exist on this process.

  if( needToInitialize_ ) {
    initialize();
    needToInitialize_ = false;
  }

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);

  // Collect the sample plane data that is held on this process.

  int nSamp = sampleDistrib_[myrank];

  for (i=0; i<nSamp; ++i) {
    double *USamp = stk::mesh::field_data(velocityNp1,nodeMapSamp_[i]);
    wSamp[i] = USamp[2];
    UAvg[0] += USamp[0]*nxnyInv;
    UAvg[1] += USamp[1]*nxnyInv;
    UAvg[2] += USamp[2]*nxnyInv;
  }

  // Find contributions to the average velocity at the x inflow boundary

  if (horizBCType_ == 1 || horizBCType_ == 3) {
    for (i=0; i<nXInflow_; ++i) {
      double *USamp = stk::mesh::field_data(velocityNp1,nodeMapXInflow_[i]);
      UAvg[3] += xInflowWeight_[i]*USamp[0];
      UAvg[4] += xInflowWeight_[i]*USamp[1];
      UAvg[5] += xInflowWeight_[i]*USamp[2];
    }
  }

  // Find contributions to the average velocity at the y inflow boundary

  if (horizBCType_ == 2 || horizBCType_ == 3) {
    for (j=0; j<nYInflow_; ++j) {
      double *USamp = stk::mesh::field_data(velocityNp1,nodeMapYInflow_[j]);
      UAvg[6] += yInflowWeight_[j]*USamp[0];
      UAvg[7] += yInflowWeight_[j]*USamp[1];
      UAvg[8] += yInflowWeight_[j]*USamp[2];
    }
  }

  // Gather the sampling plane data across all processes.

  MPI_Allgatherv(wSamp.data(), nSamp, MPI_DOUBLE, work.data(), 
                 sampleDistrib_.data(), displ_.data(), MPI_DOUBLE,
                 bulk_data.parallel());

  // Reorder the sample plane data.

  for (i=0; i<nx*ny; ++i) {
    wSamp[indexMapSampGlobal_[i]] = work[i];
  }

  // Sum the average velocty contributions across all processes.

  MPI_Allreduce(MPI_IN_PLACE, UAvg.data(), 9, MPI_DOUBLE, MPI_SUM,
                bulk_data.parallel());

  // Compute the upper boundary velocity field

  switch (horizBCType_) {
    case 0:
      potentialBCPeriodicPeriodic( wSamp, UAvg, uBC, vBC, wBC );
    break;
    case 1:
      potentialBCInflowPeriodic( wSamp, UAvg, uBC, vBC, wBC );
    break;
    case 3:
      potentialBCInflowInflow( wSamp, UAvg, uBC, vBC, wBC );
  }

  // Set the boundary velocity array values.

  for (i=0; i<nBC_; ++i) {
    ii = indexMapBC_[i];
    double *uTop  = stk::mesh::field_data(*bcVelocity_, nodeMapBC_[i]);
    uTop[0] = uBC[ii];
    uTop[1] = vBC[ii];
    uTop[2] = wBC[ii];
  }

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
  std::vector<int> indexMapSamp(imax_*jmax_), indexMapXInflow(jmax_),
                   indexMapYInflow(imax_);

  double z0, z1, zL, nxInv, nyInv;
  int i, ii, ix, ixInflow, iy, iyInflow, iz, izSample, imaxjmax, j, n, 
      nx, ny, nz, iOff, count, count1, countXInflow, countYInflow, nSamp;

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

  // Trap bad values for the horizontal BC index flags

  if (std::abs(horizBC_[0])>1 || std::abs(horizBC_[1])>1 || 
      std::abs(horizBC_[2])>1 || std::abs(horizBC_[3])>1) {
    throw std::runtime_error(
      "AssembleMomentumEdgeABLTopBC: Bad user input for horizontal_bcs");
  }
  if ((horizBC_[0]+horizBC_[1])!=0 || (horizBC_[2]+horizBC_[3])!=0) {
    throw std::runtime_error(
      "AssembleMomentumEdgeABLTopBC: Bad user input for horizontal_bcs");
  }

  // Set horizontal BC type flag

  if (         horizBC_[0]==0 && 
               horizBC_[2]==0     ) horizBCType_ = 0;  // periodic-periodic
  if (std::abs(horizBC_[0])==1 &&
               horizBC_[2]==0     ) horizBCType_ = 1;  // inflow  -periodic
  if (         horizBC_[0]==0 && 
      std::abs(horizBC_[2])==1    ) horizBCType_ = 2;  // periodic-inflow  
  if (std::abs(horizBC_[0])==1 && 
      std::abs(horizBC_[2])==1    ) horizBCType_ = 3;  // inflow  -inflow

  // Define fft plans.

  unsigned flags=FFTW_ESTIMATE;

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
    case 3:
      planSinx_ = 
      fftw_plan_r2r_1d(nx-1, work.data(), work.data(), FFTW_RODFT00, flags);
      planCosx_ = 
      fftw_plan_r2r_1d(nx+1, work.data(), work.data(), FFTW_REDFT00, flags);
      planSiny_ = 
      fftw_plan_r2r_1d(ny-1, work.data(), work.data(), FFTW_RODFT00, flags);
      planCosy_ = 
      fftw_plan_r2r_1d(ny+1, work.data(), work.data(), FFTW_REDFT00, flags);
    break;
    default:
      throw std::runtime_error(
        "AssembleMomentumEdgeABLTopBC::initialize(): Invalid value for "
        "horizBCType_. Must be 0, 1, or 3.");
    }

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

      if (bulk_data.is_valid(nodeSamp) &&
          bulk_data.bucket(nodeSamp).owned()) {
        nodeMapSamp_[count] = nodeSamp;
        indexMapSamp[count] = iy*nx + ix;
        count ++;
      }

    }
  }
  nSamp = count;

  // Determine where the inflow plane averages are to be computed.

  ixInflow = -1;
  iyInflow = -1;
  if (horizBC_[0] == 1) ixInflow = 0;
  if (horizBC_[1] == 1) ixInflow = nx;
  if (horizBC_[2] == 1) iyInflow = 0;
  if (horizBC_[3] == 1) iyInflow = ny;

  // Generate a map for the boundary points contained on this process.

  iOff = nz*imaxjmax;

  count        = 0;
  countXInflow = 0;
  countYInflow = 0;
  for (iy=0; iy<jmax_; ++iy) {
    for (ix=0; ix<imax_; ++ix) {

      stk::mesh::EntityId IdNodeBC = iOff + iy*imax_ + ix + 1;
      stk::mesh::EntityId IdNodeM1 = IdNodeBC - imaxjmax;
      stk::mesh::Entity nodeBC =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeBC);
      stk::mesh::Entity nodeM1 =
        bulk_data.get_entity(stk::topology::NODE_RANK,IdNodeM1);

      if (bulk_data.is_valid(nodeBC) &&
          bulk_data.bucket(nodeBC).owned()) {
        nodeMapBC_[ count] = nodeBC;
        nodeMapM1_[ count] = nodeM1;
        indexMapBC_[count] = iy*imax_ + ix;
        count ++;
        if (ix == ixInflow) {
//          nodeMapXInflow_[countXInflow] = nodeBC;
          nodeMapXInflow_[countXInflow] = nodeM1;  // one point below the bndry
          indexMapXInflow[countXInflow] = iy;
          countXInflow ++;
        }

        if (iy == iyInflow) {
//          nodeMapYInflow_[countYInflow] = nodeBC;
          nodeMapYInflow_[countYInflow] = nodeM1; // one point below the bndry
          indexMapYInflow[countYInflow] = ix;
          countYInflow ++;
        }
      }

    }
  }
  nBC_      = count;
  nXInflow_ = countXInflow;
  nYInflow_ = countYInflow;

  // Form a global list of the xInflow index maps.

  MPI_Allgather(&nXInflow_, 1, MPI_INT, sampleDistrib_.data(), 1, MPI_INT,
                bulk_data.parallel());

  displ_[0] = 0;
  for (i=1; i<nprocs+1; ++i) { 
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
  }

  MPI_Allgatherv(indexMapXInflow.data(), nXInflow_, MPI_INT, 
                 indexMapSampGlobal_.data(), sampleDistrib_.data(), 
                 displ_.data(), MPI_INT, bulk_data.parallel());

  // Flag redundant elements in the global xInflow list for removal with -1.

  for (n=0; n<nprocs; ++n) {
    for (i=displ_[n]; i<displ_[n+1]; i++) {
      for (j=displ_[n+1]; j<displ_[nprocs]; ++j) {
        if (indexMapSampGlobal_[i] == indexMapSampGlobal_[j]) {
          if (indexMapSampGlobal_[i] == ny) {
            indexMapSampGlobal_[i] = -1;
          } else {
            indexMapSampGlobal_[j] = -1;
          }
        }
      }
    }
  }

  // Remove the redundant elements from the local xInflow list.

  n = myrank;
  count = 0;
  for (i=displ_[n]; i<displ_[n+1]; i++) {
    ii = i-displ_[n];
    if (indexMapSampGlobal_[i]>=0) {
      nodeMapXInflow_[count] = nodeMapXInflow_[ii];
      indexMapXInflow[count] = indexMapXInflow[ii];
      count ++;
    }
  }
  nXInflow_ = count;

  // Compute the weighting factors for the local contribution to the 
  // xInflow average.

  nyInv = 1.0/(double)ny;
  for (i=0; i<nXInflow_; ++i) {
    if (indexMapXInflow[i] == 0 || indexMapXInflow[i] == ny) {
      xInflowWeight_[i] = 0.5*nyInv;
    } else {
      xInflowWeight_[i] = nyInv;
    }
  }

  // Form a global list of the yInflow index maps.

  MPI_Allgather(&nYInflow_, 1, MPI_INT, sampleDistrib_.data(), 1, MPI_INT,
                bulk_data.parallel());

  displ_[0] = 0;
  for (i=1; i<nprocs+1; ++i) { 
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
  }

  MPI_Allgatherv(indexMapYInflow.data(), nYInflow_, MPI_INT, 
                 indexMapSampGlobal_.data(), sampleDistrib_.data(), 
                 displ_.data(), MPI_INT, bulk_data.parallel());

  // Flag redundant elements in the global yInflow list for removal with -1.

  for (n=0; n<nprocs; ++n) {
    for (i=displ_[n]; i<displ_[n+1]; i++) {
      for (j=displ_[n+1]; j<displ_[nprocs]; ++j) {
        if (indexMapSampGlobal_[i] == indexMapSampGlobal_[j]) {
          if (indexMapSampGlobal_[i] == nx) {
            indexMapSampGlobal_[i] = -1;
          } else {
            indexMapSampGlobal_[j] = -1;
          }
        }
      }
    }
  }

  // Remove the redundant elements from the local yInflow list.

  n = myrank;
  count = 0;
  for (i=displ_[n]; i<displ_[n+1]; i++) {
    ii = i-displ_[n];
    if (indexMapSampGlobal_[i]>=0) {
      nodeMapYInflow_[count] = nodeMapYInflow_[ii];
      indexMapYInflow[count] = indexMapYInflow[ii];
      count ++;
    }
  }
  nYInflow_ = count;

  // Compute the weighting factors for the local contribution to the 
  // yInflow average.

  nxInv = 1.0/(double)nx;
  for (i=0; i<nYInflow_; ++i) {
    if (indexMapYInflow[i] == 0 || indexMapYInflow[i] == nx) {
      yInflowWeight_[i] = 0.5*nxInv;
    } else {
      yInflowWeight_[i] = nxInv;
    }
  }

  // Form a global list of the sample plane index maps.

  MPI_Allgather(&nSamp, 1, MPI_INT, sampleDistrib_.data(), 1,
                MPI_INT, bulk_data.parallel());

  displ_[0] = 0;
  size_t globalSize = 0;
  for (i=1; i<nprocs+1; ++i) { 
    displ_[i] = displ_[i-1] + sampleDistrib_[i-1];
    globalSize += sampleDistrib_[i-1];
  }
  indexMapSampGlobal_.resize(globalSize);

  MPI_Allgatherv(indexMapSamp.data(), nSamp, MPI_INT, 
                 indexMapSampGlobal_.data(), sampleDistrib_.data(), 
                 displ_.data(), MPI_INT, bulk_data.parallel());

  // Flag redundant elements in the global wSamp list for removal with -1.

  for (n=0; n<nprocs; ++n) {
    for (i=displ_[n]; i<displ_[n+1]; i++) {
      for (j=displ_[n+1]; j<displ_[nprocs]; ++j) {
        if (indexMapSampGlobal_[i] == indexMapSampGlobal_[j]) {
          if ( indexMapSampGlobal_[i]<nx ||
              (indexMapSampGlobal_[i]%nx)==0) {
            indexMapSampGlobal_[j] = -1;
          } else {
            indexMapSampGlobal_[i] = -1;
          }
        }
      }
    }
  }

  // Remove the redundant elements from both the global indexMap and
  // the local nodeMap.

  n = myrank;
  count = 0;
  for (n=0; n<nprocs; ++n) {
    count1 = 0;
    for (i=displ_[n]; i<displ_[n+1]; i++) {
      ii = i-displ_[n];
      if (indexMapSampGlobal_[i]>=0) {
        indexMapSampGlobal_[count] = indexMapSampGlobal_[i];
        if (n==myrank) {
          nodeMapSamp_[count1] = nodeMapSamp_[ii];
        }
        count  ++;
        count1 ++;
      }
    }
    sampleDistrib_[n] = count1;
  }

  // Rebuild the global displacement vector.

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
  uCoef[0] = UAvg[0];
//  wCoef[1] = UAvg[1];
//  wCoef[0] = UAvg[2];
  vCoef[0] = 0.0;
  wCoef[0] = 0.0;

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
         zFac, wt, u0, v0, uInc, vInc, wInc;
  int i, i0, ii, iOff, j, j0, j1, jj, nx, ny;

  const double pi = std::acos(-1.0);
  const std::complex<double> iUnit(0.0,1.0);

  nx = imax_-1;
  ny = jmax_-1;

  // Forward transform of wSamp.  Sine transform in x, Fourier transform
  // in y.  The Nyquist mode in x is not stored since it is explicitly
  // zero.  The zero mode in x is set to zero explicitly.  Note that the 
  // data is transposed between the x and y transforms.

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

  // Solve the potential flow problem.  The Nyquist modes in x are not
  // considered since they are identically zero.  u0 and v0 are the 
  // average velocity components at the x=x_min edge.

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
  // in y, either sine or cosine transform in x.  Note that the Nyquist mode
  // in x needs to be set to zero prior to a cosine transform.  Also note
  // that the data is transposed between the y and x transforms.

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
    uBC[i0+nx] = 0.0;
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
    fftw_execute_r2r(planSinx_, &vBC[i0+1], &vBC[i0+1]);
    vBC[i0   ] = 0.0;
    vBC[i0+nx] = 0.0;
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

  uInc = UAvg[3] - u0;
  vInc = UAvg[4] - v0;
//  wInc = UAvg[2];
  wInc = 0.0;
  for (i=0; i<imax_*ny; ++i) {
    uBC[i] += uInc;
    vBC[i] += vInc;
    wBC[i] += wInc;
  }

  // Enforce periodicity in y.

  iOff = ny*imax_;
  for (i=0; i<imax_; ++i) {
    uBC[iOff+i] = uBC[i];
    vBC[iOff+i] = vBC[i];
    wBC[iOff+i] = wBC[i];
  }
}

//--------------------------------------------------------------------------
//-------- potentialBCInflowInflow -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::potentialBCInflowInflow( 
  std::vector<double>& wSamp,
  std::vector<double>& UAvg,
  std::vector<double>& uBC,
  std::vector<double>& vBC,
  std::vector<double>& wBC )
{

  std::vector<double> uCoef(imax_*jmax_), vCoef(imax_*jmax_),
                      wCoef(imax_*jmax_);

  double waveX, waveY, normFac, kx, kx2, ky, kMag, eFac, scale, xFac, yFac,
         zFac, wtX, wtY, u0X, u0Y, v0X, v0Y, uInc, vInc, wInc;
  int i, i0, ii, j, j0, jj, nx, ny;

  const double pi = std::acos(-1.0);

  nx = imax_-1;
  ny = jmax_-1;

  // Forward transform of wSamp.  Sine transform in x, sine transform
  // in y.  The Nyquist modes in x are not stored since they are identically
  // zero.  The Nyquist modes in y are stored (as zeros) in order to make
  // the array stride wide enough for an in place cosine transform (for the
  // reverse transfrom process).  The zero modes in both x and y are explicitly
  // set to zero.  Note that the data is transposed between the x and y 
  // transforms.

  for (j=1; j<ny; ++j) {
    i0 = j*nx;
    fftw_execute_r2r(planSinx_, &wSamp[i0+1], &wSamp[i0+1]);
    wCoef[j] = 0.0;            // i=0  (zero  mode in x)
    for (i=1; i<nx; ++i) {
      ii = i*jmax_ + j;
      wCoef[ii] = wSamp[i0+i];
    }
    wCoef[nx*jmax_+j] = 0.0;   // i=nx (Nyquist mode in x)
  }
  for (i=1; i<nx; ++i) {
    j0 = i*jmax_;
    wCoef[j0] = 0.0;           // j=0  (zero mode in y)
    fftw_execute_r2r(planSiny_, &wCoef[j0+1], &wCoef[j0+1]);
    wCoef[j0+ny] = 0.0;        // j=ny (Nyquist mode in y)
  }
  wCoef[0] = 0.0;              // i=0, j=0 (zero mode in x and y)

  // Solve the potential flow problem.  The Nyquist modes in x and y are not
  // considered since they are identically zero.  u0 and v0 are the average 
  // velocity components at the x=x_min edge.

  waveX = pi/xL_;
  waveY = pi/yL_;
  normFac = 1.0/((double)(2*nx)*(double)(2*ny));

  u0X = 0;   v0X = 0;
  u0Y = 0;   v0Y = 0;
  for (i=0; i<nx; ++i) {
    j0 = i*jmax_;
    kx = waveX*(double)i;
    kx2 = kx*kx;
    wtX = 2.0;
    if (i==0 || i==nx) { wtX = 1.0; }
    for (j=0; j<=ny; ++j) {
      jj = j0 + j;
      ky = waveY*(double)j;
      kMag = std::sqrt( kx2 + ky*ky );
      eFac = std::exp(-kMag*deltaZ_)*normFac;
      scale = 1.0/(kMag+1.0e-15);
      xFac = kx*scale*eFac;
      yFac = ky*scale*eFac;
      zFac =          eFac;
      uCoef[jj] = -xFac*wCoef[jj];
      vCoef[jj] = -yFac*wCoef[jj];
      wCoef[jj] =  zFac*wCoef[jj];
      if (j == 0) {
        u0X += wtX*uCoef[jj];
        v0X += wtX*vCoef[jj];
      }
      if (i == 0) {
        wtY = 2.0;
        if (j==0 || j==ny) { wtY = 1.0; }
        u0Y += wtY*uCoef[jj];
        v0Y += wtY*vCoef[jj];
      }
    }
  }

  // Reverse transform the solution at the upper boundary. Either sine or 
  // cosine in both x and y.  Note that the Nyquist modes need to be set
  // to zero prior to a cosine transform.  Also note that the data is 
  // transposed between the y and x transforms.

  for (i=0; i<nx; ++i) {
    j0 = i*jmax_;
    vCoef[j0+ny] = 0.0;       // Nyquist mode in y
    fftw_execute_r2r(planSiny_, &uCoef[j0+1], &uCoef[j0+1]);
    fftw_execute_r2r(planCosy_, &vCoef[j0  ], &vCoef[j0  ]);
    fftw_execute_r2r(planSiny_, &wCoef[j0+1], &wCoef[j0+1]);
    uCoef[j0   ] = 0.0;
    uCoef[j0+ny] = 0.0;
    wCoef[j0   ] = 0.0;
    wCoef[j0+ny] = 0.0;
  }
  for (j=0; j<jmax_; ++j) {
    i0 = j*imax_;
    for (i=0; i<nx; ++i) {
      ii = i*jmax_ + j;
      uBC[i0+i] = uCoef[ii];
      vBC[i0+i] = vCoef[ii];
      wBC[i0+i] = wCoef[ii];
    }
    uBC[i0+nx] = 0.0;      // (Nyquist mode in x)
    fftw_execute_r2r(planCosx_, &uBC[i0  ], &uBC[i0  ]);
    fftw_execute_r2r(planSinx_, &vBC[i0+1], &vBC[i0+1]);
    fftw_execute_r2r(planSinx_, &wBC[i0+1], &wBC[i0+1]);
    vBC[i0   ] = 0.0;
    vBC[i0+nx] = 0.0;
    wBC[i0   ] = 0.0;
    wBC[i0+nx] = 0.0;
  }

  // Adjust the u and v mean velocity so that the velocity computed at the
  // x=x_min edge matches the inflow velocity.

  uInc = UAvg[3] - u0X;
  vInc = UAvg[7] - v0Y;
//  wInc = UAvg[2];
  wInc = 0.0;
  for (i=0; i<imax_*jmax_; ++i) {
    uBC[i] += uInc;
    vBC[i] += vInc;
    wBC[i] += wInc;
  }

}

} // namespace nalu
} // namespace Sierra
