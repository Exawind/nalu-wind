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
KOKKOS_FUNCTION
TetSCV::TetSCV()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCV::ipNodeMap(
  int /*ordinal*/) const
{
  // define scv->node mappings
  return ipNodeMap_;
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

  const int npe  = nodesPerElement_;
  const int nint = numIntPoints_;
  SIERRA_FORTRAN(tet_scv_det)
    ( &nelem, &npe, &nint, coords,
      volume, error, &lerr );
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCV::shape_fcn(double *shpfc)
{
  tet_shape_fcn(numIntPoints_, intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
TetSCV::shifted_shape_fcn(double *shpfc)
{
  tet_shape_fcn(numIntPoints_, intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- tet_shape_fcn ---------------------------------------------------
//--------------------------------------------------------------------------
void
TetSCV::tet_shape_fcn(
  const int  npts,
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
KOKKOS_FUNCTION
TetSCS::TetSCS()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

  MasterElement::scsIpEdgeOrd_.assign(scsIpEdgeOrd_,  numIntPoints_+scsIpEdgeOrd_);

  MasterElement::intgExpFace_.assign(&intgExpFace_[0][0][0], 36+&intgExpFace_[0][0][0]);

  const double nodeLocations[4][3] = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}};

  stk::topology topo = stk::topology::TET_4;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = side_node_ordinals(k);
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift_[k][n][0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift_[k][n][1] = nodeLocations[ordinals[n]][1];
      intgExpFaceShift_[k][n][2] = nodeLocations[ordinals[n]][2];
    }
  }
  MasterElement::intgExpFaceShift_.assign(&intgExpFaceShift_[0][0][0], 36+&intgExpFaceShift_[0][0][0]);
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCS::ipNodeMap(
  int ordinal) const
{
  // define ip->node mappings for each face (ordinal); 
  return ipNodeMap_[ordinal];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
TetSCS::side_node_ordinals(
  int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return sideNodeOrdinals_[ordinal];
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
  const int npe  = nodesPerElement_;
  const int nint = numIntPoints_;
  SIERRA_FORTRAN(tet_scs_det)
    ( &nelem, &npe, &nint, coords, areav );

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

  const int npe  = nodesPerElement_;
  const int nint = numIntPoints_;
  SIERRA_FORTRAN(tet_derivative)
    ( &nint, deriv );
  
  SIERRA_FORTRAN(tet_gradient_operator)
    ( &nelem,
      &npe,
      &nint,
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

  const int npe  = nodesPerElement_;
  const int nint = numIntPoints_;
  SIERRA_FORTRAN(tet_derivative)
    ( &nint, deriv );

  SIERRA_FORTRAN(tet_gradient_operator)
    ( &nelem,
      &npe,
      &nint,
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

      const int npe  = nodesPerElement_;
      SIERRA_FORTRAN(tet_gradient_operator)
        ( &nface,
          &npe,
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

      const int npe  = nodesPerElement_;
      SIERRA_FORTRAN(tet_gradient_operator)
        ( &nface,
          &npe,
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
  const int npe  = nodesPerElement_;
  const int nint = numIntPoints_;
  SIERRA_FORTRAN(threed_gij)
    ( &npe,
      &nint,
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
  return lrscv_;
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
  const int  npts,
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
  return oppNode_[ordinal][node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
TetSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal][node];
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

  const int npe  = nodesPerElement_;
  SIERRA_FORTRAN(tet_gradient_operator)
    ( &nface,
      &npe,
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

