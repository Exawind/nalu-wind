/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/Tet4CVFEM.h>
#include <master_element/Hex8GeometryFunctions.h>

#include <AlgTraits.h>

#include <NaluEnv.h>
#include <FORTRAN_Proto.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <iostream>

#include <cmath>
#include <limits>
#include <array>
#include <map>
#include <memory>

namespace sierra{
namespace nalu{

//-------- tet_deriv -------------------------------------------------------
template <typename DerivType>
void tet_deriv(DerivType& deriv)
{
  for(size_t j=0; j<deriv.extent(0); ++j) {
    deriv(j,0,0) = -1.0;
    deriv(j,0,1) = -1.0;
    deriv(j,0,2) = -1.0;
              
    deriv(j,1,0) = 1.0;
    deriv(j,1,1) = 0.0;
    deriv(j,1,2) = 0.0;
              
    deriv(j,2,0) = 0.0;
    deriv(j,2,1) = 1.0;
    deriv(j,2,2) = 0.0;
              
    deriv(j,3,0) = 0.0;
    deriv(j,3,1) = 0.0;
    deriv(j,3,2) = 1.0;
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TetSCV::TetSCV()
  : MasterElement()
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 4;
  numIntPoints_ = 4;

  // define ip node mappings
  ipNodeMap_.resize(4);
  ipNodeMap_[0] = 0; ipNodeMap_[1] = 1; ipNodeMap_[2] = 2; ipNodeMap_[3] = 3;

  // standard integration location
  intgLoc_.resize(12);
  const double seventeen96ths = 17.0/96.0;
  const double fourfive96ths  = 45.0/96.0;
  intgLoc_[0] = seventeen96ths; intgLoc_[1]  = seventeen96ths; intgLoc_[2]  = seventeen96ths; // vol 1
  intgLoc_[3] = fourfive96ths;  intgLoc_[4]  = seventeen96ths; intgLoc_[5]  = seventeen96ths; // vol 2
  intgLoc_[6] = seventeen96ths; intgLoc_[7]  = fourfive96ths;  intgLoc_[8]  = seventeen96ths; // vol 3
  intgLoc_[9] = seventeen96ths; intgLoc_[10] = seventeen96ths; intgLoc_[11] = fourfive96ths;  // vol 4

  // shifted
  intgLocShift_.resize(12);
  intgLocShift_[0] = 0.0; intgLocShift_[1]  = 0.0;  intgLocShift_[2] = 0.0;
  intgLocShift_[3] = 1.0; intgLocShift_[4]  = 0.0;  intgLocShift_[5] = 0.0;
  intgLocShift_[6] = 0.0; intgLocShift_[7]  = 1.0;  intgLocShift_[8] = 0.0;
  intgLocShift_[9] = 0.0; intgLocShift_[10] = 0.0; intgLocShift_[11] = 1.0;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TetSCV::~TetSCV()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void TetSCV::determinant(
    SharedMemView<DoubleType**>& coordel,
    SharedMemView<DoubleType*>& volume)
{
  const int tetSubcontrolNodeTable[4][8] = {
    {0, 4, 7, 6, 11, 13, 14, 12},
    {1, 5, 7, 4, 9, 10, 14, 13},
    {2, 6, 7, 5, 8, 12, 14, 10},
    {3, 9, 13, 11, 8, 10, 14, 12}
  };

  const double half = 0.5;
  const double one3rd = 1.0/3.0;
  DoubleType coords[15][3];
  DoubleType ehexcoords[8][3];
  const int dim[3] = {0, 1, 2};

  // element vertices
  for(int j=0; j<4; ++j) {
    for(int k : dim) {
      coords[j][k] = coordel(j,k);
    }
  }

  // face 1 (tri)

  // edge midpoints
  for(int k : dim) {
    coords[4][k] = half*(coordel(0,k) + coordel(1,k));
  }
  for(int k : dim) {
    coords[5][k]= half*(coordel(1,k) + coordel(2,k));
  }
  for(int k : dim) {
    coords[6][k] = half*(coordel(2,k) + coordel(0,k));
  }

  // face mipdoint
  for(int k : dim) {
    coords[7][k] = one3rd*(coordel(0,k) + coordel(1,k) + coordel(2,k));
  }

  // face 2 (tri)

  // edge midpoints
  for(int k : dim) {
    coords[8][k] = half*(coordel(2,k) + coordel(3,k));
  }
  for(int k : dim) {
    coords[9][k] = half*(coordel(3,k) + coordel(1,k));
  }

  // face midpoint
  for(int k : dim) {
    coords[10][k] = one3rd*(coordel(1,k) + coordel(2,k) + coordel(3,k));
  }

  // face 3 (tri)

  // edge midpoint
  for(int k : dim) {
    coords[11][k] = half*(coordel(0,k) + coordel(3,k));
  }

  // face midpoint
  for(int k : dim) {
    coords[12][k] = one3rd*(coordel(0,k) + coordel(2,k) + coordel(3,k));
  }

  // face 4 (tri)

  // face midpoint
  for(int k : dim) {
    coords[13][k] = one3rd*(coordel(0,k) + coordel(1,k) + coordel(3,k));
  }

  // element centroid
  for(int k : dim) {
    coords[14][k] = 0.0;
  }
  for(int j=0; j<nodesPerElement_; ++j) {
    for(int k : dim) {
      coords[14][k] = coords[14][k] + 0.25*coordel(j,k);
    }
  }

  // loop over subcontrol volumes
  for(int icv=0; icv<numIntPoints_; ++icv) {
    // loop over nodes of scv
    for(int inode=0; inode<8; ++inode) {
      // define scv coordinates using node table
      for(int k : dim) {
        ehexcoords[inode][k] = coords[tetSubcontrolNodeTable[icv][inode]][k];
      }
    }
    // compute volume using an equivalent polyhedron
    volume(icv) = hex_volume_grandy(ehexcoords);
    // check for negative volume
    //ThrowAssertMsg( volume(icv) < 0.0, "ERROR in TetSCV::determinant, negative volume.");
  }
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void TetSCV::grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv)
{
  tet_deriv(deriv);
  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void TetSCV::shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv)
{
  tet_deriv(deriv);
  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

void TetSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(tet_scv_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords,
      volume, error, &lerr );
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCV::shape_fcn(double *shpfc)
{
  tet_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
TetSCV::shifted_shape_fcn(double *shpfc)
{
  tet_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- tet_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCV::tet_shape_fcn(
  const int  &npts,
  const double *par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int fourj = 4*j;
    const int k = 3*j;
    const double xi = par_coord[k];
    const double eta = par_coord[k+1];
    const double zeta = par_coord[k+2];
    shape_fcn[fourj] = 1.0 - xi - eta - zeta;
    shape_fcn[1 + fourj] = xi;
    shape_fcn[2 + fourj] = eta;
    shape_fcn[3 + fourj] = zeta;
  }
}

//--------------------------------------------------------------------------
//-------- Mij ------------------------------------------------------------
//--------------------------------------------------------------------------
void TetSCV::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsTet4>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void TetSCV::Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_3d<AlgTraitsTet4>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TetSCS::TetSCS()
  : MasterElement()
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 4;
  numIntPoints_ = 6;

  // define L/R mappings
  lrscv_.resize(12);
  lrscv_[0]  = 0; lrscv_[1]  = 1;
  lrscv_[2]  = 1; lrscv_[3]  = 2;
  lrscv_[4]  = 0; lrscv_[5]  = 2;
  lrscv_[6]  = 0; lrscv_[7]  = 3;
  lrscv_[8]  = 1; lrscv_[9]  = 3;
  lrscv_[10] = 2; lrscv_[11] = 3;

  // elem-edge mapping from ip
  scsIpEdgeOrd_.resize(numIntPoints_);
  scsIpEdgeOrd_[0] = 0; scsIpEdgeOrd_[1] = 1; scsIpEdgeOrd_[2] = 2; 
  scsIpEdgeOrd_[3] = 3; scsIpEdgeOrd_[4] = 4; scsIpEdgeOrd_[5] = 5;
 
  // define opposing node
  oppNode_.resize(12);
  // face 0
  oppNode_[0] = 2; oppNode_[1] = 2;  oppNode_[2] = 2;
  // face 1
  oppNode_[3] = 0; oppNode_[4] = 0;  oppNode_[5] = 0;
  // face 2
  oppNode_[6] = 1; oppNode_[7] = 1;  oppNode_[8] = 1;
  // face 3
  oppNode_[9] = 3; oppNode_[10] = 3; oppNode_[11] = 3;

  // define opposing face
  oppFace_.resize(12);
  // face 0
  oppFace_[0]  = 2; oppFace_[1] = 1;  oppFace_[2] = 5;
  // face 1
  oppFace_[3]  = 0; oppFace_[4] = 2;  oppFace_[5] = 3;
  // face 2
  oppFace_[6]  = 0; oppFace_[7] = 4;  oppFace_[8] = 1;
  // face 3
  oppFace_[9]  = 3; oppFace_[10] = 5; oppFace_[11] = 4;

  // standard integration location
  intgLoc_.resize(18);
  const double thirteen36ths = 13.0/36.0;
  const double five36ths = 5.0/36.0;
  intgLoc_[0]  =  thirteen36ths; intgLoc_[1]  = five36ths;     intgLoc_[2]  = five36ths; // surf 1    1->2
  intgLoc_[3]  =  thirteen36ths; intgLoc_[4]  = thirteen36ths; intgLoc_[5]  = five36ths; // surf 2    2->3
  intgLoc_[6]  =  five36ths;     intgLoc_[7]  = thirteen36ths; intgLoc_[8]  = five36ths; // surf 3    1->3
  intgLoc_[9]  =  five36ths ;    intgLoc_[10] = five36ths;     intgLoc_[11] = thirteen36ths; // surf 4    1->4
  intgLoc_[12] =  thirteen36ths; intgLoc_[13] = five36ths;     intgLoc_[14] = thirteen36ths; // surf 5    2->4
  intgLoc_[15] =  five36ths;     intgLoc_[16] = thirteen36ths; intgLoc_[17] = thirteen36ths; // surf 6    3->4

  // shifted
  intgLocShift_.resize(36);
  intgLocShift_[0]  =  0.50; intgLocShift_[1]  =  0.00; intgLocShift_[2]  =  0.00; // surf 1    1->2
  intgLocShift_[3]  =  0.50; intgLocShift_[4]  =  0.50; intgLocShift_[5]  =  0.00; // surf 2    2->3
  intgLocShift_[6]  =  0.00; intgLocShift_[7]  =  0.50; intgLocShift_[8]  =  0.00; // surf 3    1->3
  intgLocShift_[9]  =  0.00; intgLocShift_[10] =  0.00; intgLocShift_[11] =  0.50; // surf 4    1->4
  intgLocShift_[12] =  0.50; intgLocShift_[13] =  0.00; intgLocShift_[14] =  0.50; // surf 5    2->4
  intgLocShift_[15] =  0.00; intgLocShift_[16] =  0.50; intgLocShift_[17] =  0.50; // surf 6    3->4

  // exposed face
  intgExpFace_.resize(36);
  const double seven36ths = 7.0/36.0;
  const double eleven18ths = 11.0/18.0;
  // face 0; nodes 0,1,3: scs 0, 1, 2
  intgExpFace_[0]  = seven36ths;  intgExpFace_[1]  =  0.00; intgExpFace_[2]  = seven36ths;
  intgExpFace_[3]  = eleven18ths; intgExpFace_[4]  =  0.00; intgExpFace_[5]  = seven36ths;
  intgExpFace_[6]  = seven36ths;  intgExpFace_[7]  =  0.00; intgExpFace_[8]  = eleven18ths;
  // face 1; nodes 1,2,3; scs 0, 1, 2
  intgExpFace_[9]  = eleven18ths; intgExpFace_[10] = seven36ths;  intgExpFace_[11] = seven36ths;
  intgExpFace_[12] = seven36ths;  intgExpFace_[13] = eleven18ths; intgExpFace_[14] = seven36ths;
  intgExpFace_[15] = seven36ths;  intgExpFace_[16] = seven36ths;  intgExpFace_[17] = eleven18ths;
  // face 2; nodes 0,3,2; scs 0, 1, 2
  intgExpFace_[18] =  0.00;      intgExpFace_[19] = seven36ths;  intgExpFace_[20] = seven36ths;
  intgExpFace_[21] =  0.00;      intgExpFace_[22] = seven36ths;  intgExpFace_[23] = eleven18ths;
  intgExpFace_[24] =  0.00;      intgExpFace_[25] = eleven18ths; intgExpFace_[26] = seven36ths;
  //face 3; nodes 0, 2, 1; scs 0, 1, 2
  intgExpFace_[27] = seven36ths;  intgExpFace_[28] = seven36ths;  intgExpFace_[29] =  0.00;
  intgExpFace_[30] = eleven18ths; intgExpFace_[31] = seven36ths;  intgExpFace_[32] =  0.00;
  intgExpFace_[33] = seven36ths;  intgExpFace_[34] = eleven18ths; intgExpFace_[35] =  0.00;

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  ipNodeMap_.resize(12); // 3 ips * 4 faces
  // face 0;
  ipNodeMap_[0] = 0;  ipNodeMap_[1] = 1;  ipNodeMap_[2] = 3; 
  // face 1; 
  ipNodeMap_[3] = 1;  ipNodeMap_[4] = 2;  ipNodeMap_[5] = 3; 
  // face 2;
  ipNodeMap_[6] = 0;  ipNodeMap_[7] = 3;  ipNodeMap_[8] = 2;  
  // face 3;
  ipNodeMap_[9] = 0; ipNodeMap_[10] = 2; ipNodeMap_[11] = 1;

  sideNodeOrdinals_ = {
      0, 1, 3, //ordinal 0
      1, 2, 3, //ordinal 1
      0, 3, 2, //ordinal 2
      0, 2, 1  //ordinal 3
  };

  std::vector<std::vector<double>> nodeLocations =
  {
      {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
  };

  intgExpFaceShift_.resize(3*3*4);
  int index = 0;
  stk::topology topo = stk::topology::TET_4;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = side_node_ordinals(k);
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift_[3*index + 0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift_[3*index + 1] = nodeLocations[ordinals[n]][1];
      intgExpFaceShift_[3*index + 2] = nodeLocations[ordinals[n]][2];
      ++index;
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TetSCS::~TetSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal); 
  return &ipNodeMap_[ordinal*3];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCS::side_node_ordinals(
  int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return &sideNodeOrdinals_[ordinal*3];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void TetSCS::determinant(
    SharedMemView<DoubleType**>& coordel,
    SharedMemView<DoubleType**>&areav)
{
  int tetEdgeFacetTable[6][4] = {
    {4, 7, 14, 13},
    {7, 14, 10, 5},
    {6, 12, 14, 7},
    {11, 13, 14, 12},
    {13, 9, 10, 14},
    {10, 8, 12, 14}
  };

  const int npe = nodesPerElement_;
  const int nscs = numIntPoints_;
  const double half = 0.5;
  const double one3rd = 1.0/3.0;
  const double one4th = 1.0/4.0;
  const int dim[] = {0, 1, 2};
  DoubleType coords[15][3];
  DoubleType scscoords[4][3];

  //element vertices
  for(int j=0; j<4; ++j) {
    for(int k : dim) {
      coords[j][k] = coordel(j,k);
    }
  }

  //face 1 (tri)
  //
  //edge midpoints
  for(int k : dim) {
    coords[4][k] = half*(coordel(0,k) + coordel(1,k));
  }
  for(int k : dim) {
    coords[5][k] = half*(coordel(1,k) + coordel(2,k));
  }
  for(int k : dim) {
    coords[6][k] = half*(coordel(2,k) + coordel(0,k));
  }

  //face midpoint
  for(int k : dim) {
    coords[7][k] = one3rd*(coordel(0,k) + coordel(1,k) + coordel(2,k));
  }

  //face 2 (tri)
  //
  //edge midpoints
  for(int k : dim) {
    coords[8][k] = half*(coordel(2,k) + coordel(3,k));
  }
  for(int k : dim) {
    coords[9][k] = half*(coordel(3,k) + coordel(1,k));
  }

  //face midpoint
  for(int k : dim) {
    coords[10][k] = one3rd*(coordel(1,k) + coordel(2,k) + coordel(3,k));
  }

  //face 3 (tri)
  //
  //edge midpoint
  for(int k : dim) {
    coords[11][k] = half*(coordel(0,k) + coordel(3,k));
  }

  //face midpoint
  for(int k : dim) {
    coords[12][k] = one3rd*(coordel(0,k) + coordel(2,k) + coordel(3,k));
  }

  //face 4 (tri)
  //
  //face midpoint
  for(int k : dim) {
    coords[13][k] = one3rd*(coordel(0,k) + coordel(1,k) + coordel(3,k));
  }

  //element centroid
  for(int k : dim) {
    coords[14][k] = 0.0;
  }
  for(int j=0; j<npe; ++j) {
    for(int k : dim) {
      coords[14][k] += one4th*coordel(j,k);
    }
  }

  //loop over subcontrol surface
  for(int ics=0; ics<nscs; ++ics) {
    //loop over nodes of scs
    for(int inode=0; inode<4; ++inode) {
      int itrianglenode = tetEdgeFacetTable[ics][inode];
      for(int k : dim) {
        scscoords[inode][k] = coords[itrianglenode][k];
      }
    }
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}

void TetSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  SIERRA_FORTRAN(tet_scs_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords, areav );

  // all is always well; no error checking
  *error = 0;
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void TetSCS::grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv)
{
  tet_deriv(deriv);

  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

void TetSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(tet_derivative)
    ( &numIntPoints_, deriv );
  
  SIERRA_FORTRAN(tet_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative TetSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void TetSCS::shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv)
{
  tet_deriv(deriv);

  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

void TetSCS::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(tet_derivative)
    ( &numIntPoints_, deriv );

  SIERRA_FORTRAN(tet_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative TetSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void TetSCS::face_grad_op(
  const int nelem,
  const int /*face_ordinal*/,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  int npf = 3;

  const int nface = 1;
  double dpsi[12];

  for ( int n=0; n<nelem; n++ ) {

    for ( int k=0; k<npf; k++ ) {

      // derivatives are constant
      SIERRA_FORTRAN(tet_derivative)
        ( &nface, dpsi );

      SIERRA_FORTRAN(tet_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[12*n], &gradop[k*nelem*12+n*12], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "sorry, issue with face_grad_op.." << std::endl;

    }
  }
}

void TetSCS::face_grad_op(
  int /*face_ordinal*/,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using traits = AlgTraitsTri3Tet4;

  // one ip at a time
  constexpr int derivSize = traits::numFaceIp_ *  traits::nodesPerElement_ * traits::nDim_;

  DoubleType wderiv[derivSize];
  SharedMemView<DoubleType***> deriv(wderiv,traits::numFaceIp_, traits::nodesPerElement_,  traits::nDim_);
  tet_deriv(deriv);

  generic_grad_op<AlgTraitsTet4>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void TetSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  // no difference for regular face_grad_op
  face_grad_op(face_ordinal, coords, gradop);
}

void TetSCS::shifted_face_grad_op(
  const int nelem,
  const int /*face_ordinal*/,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  // no difference for regular face_grad_op

  int lerr = 0;
  int npf = 3;

  const int nface = 1;
  double dpsi[12];

  for ( int n=0; n<nelem; n++ ) {

    for ( int k=0; k<npf; k++ ) {

      // derivatives are constant
      SIERRA_FORTRAN(tet_derivative)
        ( &nface, dpsi );

      SIERRA_FORTRAN(tet_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[12*n], &gradop[k*nelem*12+n*12], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "sorry, issue with shifted_face_grad_op.." << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- gij ------------------------------------------------------------
//--------------------------------------------------------------------------
void TetSCS::gij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv)
{
  generic_gij_3d<AlgTraitsTet4>(deriv, coords, gupper, glower);
}

void TetSCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(threed_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}
 
//--------------------------------------------------------------------------
//-------- Mij ------------------------------------------------------------
//--------------------------------------------------------------------------
void TetSCS::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsTet4>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void TetSCS::Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_3d<AlgTraitsTet4>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}

//--------------------------------------------------------------------------
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCS::scsIpEdgeOrd()
{
  return &scsIpEdgeOrd_[0];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::shape_fcn(double *shpfc)
{
  tet_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::shifted_shape_fcn(double *shpfc)
{
  tet_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- tet_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::tet_shape_fcn(
  const int  &npts,
  const double *par_coord, 
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    const int fourj = 4*j;
    const int k = 3*j;
    const double xi = par_coord[k];
    const double eta = par_coord[k+1];
    const double zeta = par_coord[k+2];
    shape_fcn[fourj] = 1.0 - xi - eta - zeta;
    shape_fcn[1 + fourj] = xi;
    shape_fcn[2 + fourj] = eta;
    shape_fcn[3 + fourj] = zeta;
  }
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
int
TetSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*3+node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
TetSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*3+node];
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
TetSCS::isInElement(
    const double * elem_nodal_coor,
    const double * point_coor,
	  double * par_coor ) 
{
  // load up the element coordinates
  const double x1 = elem_nodal_coor[0];
  const double x2 = elem_nodal_coor[1];
  const double x3 = elem_nodal_coor[2];
  const double x4 = elem_nodal_coor[3];

  const double y1 = elem_nodal_coor[4];
  const double y2 = elem_nodal_coor[5];
  const double y3 = elem_nodal_coor[6];
  const double y4 = elem_nodal_coor[7];

  const double z1 = elem_nodal_coor[8];
  const double z2 = elem_nodal_coor[9];
  const double z3 = elem_nodal_coor[10];
  const double z4 = elem_nodal_coor[11];

  // determinant of matrix M in eqn x-x1 = M*xi
  const double det =
    (x2 - x1)*( (y3 - y1)*(z4 - z1) - (y4 - y1)*(z3 - z1) ) -
    (x3 - x1)*( (y2 - y1)*(z4 - z1) - (y4 - y1)*(z2 - z1) ) +
    (x4 - x1)*( (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1) );

  const double invDet = 1.0/det;

  // matrix entries in inverse of M

  const double m11 = y3*z4 - y1*z4 - y4*z3+y1*z3+y4*z1 - y3*z1;
  const double m12 =  - (x3*z4 - x1*z4 - x4*z3+x1*z3+x4*z1 - x3*z1);
  const double m13 = x3*y4 - x1*y4 - x4*y3+x1*y3+x4*y1 - x3*y1;
  const double m21 =  - (y2*z4 - y1*z4 - y4*z2+y1*z2+y4*z1 - y2*z1);
  const double m22 = x2*z4 - x1*z4 - x4*z2+x1*z2+x4*z1 - x2*z1;
  const double m23 =  - (x2*y4 - x1*y4 - x4*y2+x1*y2+x4*y1 - x2*y1);
  const double m31 = y2*z3 - y1*z3 - y3*z2+y1*z2+y3*z1 - y2*z1;
  const double m32 =  - (x2*z3 - x1*z3 - x3*z2+x1*z2+x3*z1 - x2*z1);
  const double m33 = x2*y3 - x1*y3 - x3*y2+x1*y2+x3*y1 - x2*y1;

  const double xx1 = point_coor[0] - x1;
  const double yy1 = point_coor[1] - y1;
  const double zz1 = point_coor[2] - z1;

  // solve for parametric coordinates

  const double xi   = invDet*(m11*xx1 + m12*yy1 + m13*zz1);
  const double eta  = invDet*(m21*xx1 + m22*yy1 + m23*zz1);
  const double zeta = invDet*(m31*xx1 + m32*yy1 + m33*zz1);

  // if volume coordinates are negative, point is outside the tet

  par_coor[0] = xi;
  par_coor[1] = eta;
  par_coor[2] = zeta;

  const double dist = parametric_distance(par_coor);

  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::interpolatePoint(
    const int  & ncomp_field,
    const double * par_coord,           // (3)
    const double * field,               // (4,ncomp_field)
	  double * result ) // (ncomp_field)
{
  const double xi   = par_coord[0];
  const double eta  = par_coord[1];
  const double zeta = par_coord[2];

  const double psi1 = 1.0 - xi - eta - zeta;

  for(int i = 0; i < ncomp_field; ++i) {
    const int fourI = 4*i;

    const double f1 = field[fourI];
    const double f2 = field[1 + fourI];
    const double f3 = field[2 + fourI];
    const double f4 = field[3 + fourI];

    result[i] = f1*psi1 + f2*xi + f3*eta + f4*zeta;
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
TetSCS::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  tet_shape_fcn(numIp, &isoParCoord[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void 
TetSCS::general_face_grad_op(
  const int /*face_ordinal*/,
  const double */*isoParCoord*/,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;

  const int nface = 1;
  double dpsi[12];

  // derivatives are constant
  SIERRA_FORTRAN(tet_derivative)
    ( &nface, dpsi );

  SIERRA_FORTRAN(tet_gradient_operator)
    ( &nface,
      &nodesPerElement_,
      &nface,
      dpsi,
      &coords[0], &gradop[0], &det_j[0], error, &lerr );
  
  if ( lerr )
    throw std::runtime_error("TetSCS::general_face_grad_op issue");
 
}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void 
TetSCS::sidePcoords_to_elemPcoords(
  const int & side_ordinal,
  const int & npoints,
  const double *side_pcoords,
  double *elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = 0.0;
      elem_pcoords[i*3+2] = side_pcoords[2*i+1];
    }
    break;
  case 1:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 1.0 - side_pcoords[2*i+0] - side_pcoords[2*i+1];
      elem_pcoords[i*3+1] = side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = side_pcoords[2*i+1];
    }
    break;
  case 2:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.0;
      elem_pcoords[i*3+1] = side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = side_pcoords[2*i+0];
    }
    break;
  case 3:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = side_pcoords[2*i+1];
      elem_pcoords[i*3+1] = side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = 0.0;
    }
    break;
  default:
    throw std::runtime_error("TetSCS::sideMap invalid ordinal");
  }
  return;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
TetSCS::parametric_distance(const double* x)
{
  const double X=x[0] - 1./4.;
  const double Y=x[1] - 1./4.;
  const double Z=x[2] - 1./4.;
  const double dist0 = -4*X;
  const double dist1 = -4*Y;
  const double dist2 = -4*Z;
  const double dist3 =  4*(X+Y+Z);
  const double dist  = std::max(std::max(dist0,dist1),std::max(dist2,dist3));
  return dist;
}

} // namespace nalu
} // namespace sierra

