/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <master_element/Hex8CVFEM.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/TensorOps.h>
#include <master_element/Hex8GeometryFunctions.h>

#include <FORTRAN_Proto.h>
#include <NaluEnv.h>

#include <cmath>
#include <iostream>
#include <array>

namespace sierra{
namespace nalu{

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
HexSCV::HexSCV()
  : MasterElement()
{
  MasterElement::ipNodeMap_       .assign(ipNodeMap_,        8+ipNodeMap_);
  MasterElement::intgLoc_         .assign(intgLoc_,         24+intgLoc_);
  MasterElement::intgLocShift_    .assign(intgLocShift_,    24+intgLocShift_);
  MasterElement::nDim_                  = nDim_;
  MasterElement::nodesPerElement_       = nodesPerElement_;
  MasterElement::numIntPoints_          = numIntPoints_;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
HexSCV::~HexSCV()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_scv_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords,
      volume, error, &lerr );

}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::determinant(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType*>& volume)
{
  constexpr int subDivisionTable[8][8] = {
      {  0,  8, 12, 11, 19, 20, 26, 25},
      {  8,  1,  9, 12, 20, 18, 24, 26},
      { 12,  9,  2, 10, 26, 24, 22, 23},
      { 11, 12, 10,  3, 25, 26, 23, 21},
      { 19, 20, 26, 25,  4, 13, 17, 16},
      { 20, 18, 24, 26, 13,  5, 14, 17},
      { 26, 24, 22, 23, 17, 14,  6, 15},
      { 25, 26, 23, 21, 16, 17, 15,  7}
  };

  DoubleType coordv[27][3];
  subdivide_hex_8(coords, coordv);

  constexpr int numSCV = 8;
  for (int ip = 0; ip < numSCV; ++ip) {
    DoubleType scvHex[8][3];
    for (int n = 0; n < 8; ++n) {
      const int subIndex = subDivisionTable[ip][n];
      for (int d = 0; d < 3; ++d) {
        scvHex[n][d] = coordv[subIndex][d];
      }
    }
    volume(ip) = hex_volume_grandy(scvHex);
  }
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_derivative)
    ( &numIntPoints_,
      &intgLoc_[0], deriv );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative HexSCV volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::shifted_grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCV::shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLoc_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCV::shifted_shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLocShift_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCV::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsHex8>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void HexSCV::Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_Mij_3d<AlgTraitsHex8>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
HexSCS::HexSCS() : MasterElement() { 
  MasterElement::nDim_                  = nDim_;
  MasterElement::nodesPerElement_       = nodesPerElement_;
  MasterElement::numIntPoints_          = numIntPoints_;
  MasterElement::scaleToStandardIsoFac_ = scaleToStandardIsoFac_;
  MasterElement::lrscv_           .assign(lrscv_,       24+lrscv_);
  MasterElement::ipNodeMap_       .assign(ipNodeMap_,   24+ipNodeMap_);
  MasterElement::oppNode_         .assign(oppNode_,     24+oppNode_);
  MasterElement::nodeLoc_         .assign(&nodeLoc_[0][0],  24+&nodeLoc_[0][0]);
  MasterElement::oppFace_         .assign(oppFace_,     24+oppFace_);
  MasterElement::intgLoc_         .assign(intgLoc_,     36+intgLoc_);
  MasterElement::intgLocShift_    .assign(intgLocShift_,36+intgLocShift_);
  MasterElement::scsIpEdgeOrd_    .assign(scsIpEdgeOrd_,12+scsIpEdgeOrd_);
  MasterElement::intgExpFace_     .assign(&intgExpFace_[0][0][0],  72+&intgExpFace_[0][0][0]);
  MasterElement::intgExpFaceShift_.assign(&intgExpFaceShift_[0][0][0],72+&intgExpFaceShift_[0][0][0]);
  MasterElement::nDim_                  = nDim_;
  MasterElement::nodesPerElement_       = nodesPerElement_;
  MasterElement::numIntPoints_          = numIntPoints_;
  MasterElement::scaleToStandardIsoFac_ = scaleToStandardIsoFac_;
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
HexSCS::~HexSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*4];
}


//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
 }

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------c------------------------------------
void HexSCS::shifted_grad_op(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType***>&gradop,
  SharedMemView<DoubleType***>&deriv)
{
  hex8_derivative(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::determinant(
  SharedMemView<DoubleType**>&coords,
  SharedMemView<DoubleType**>&areav)
{
  constexpr int hex_edge_facet_table[12][4] = {
      { 20,  8, 12, 26 },
      { 24,  9, 12, 26 },
      { 10, 12, 26, 23 },
      { 11, 25, 26, 12 },
      { 13, 20, 26, 17 },
      { 17, 14, 24, 26 },
      { 17, 15, 23, 26 },
      { 16, 17, 26, 25 },
      { 19, 20, 26, 25 },
      { 20, 18, 24, 26 },
      { 22, 23, 26, 24 },
      { 21, 25, 26, 23 }
  };

  DoubleType coordv[27][3];
  subdivide_hex_8(coords, coordv);

  constexpr int npf = 4;
  constexpr int nscs = 12;
  for (int ics=0; ics < nscs; ++ics) {
    DoubleType scscoords[4][3];
    for (int inode = 0; inode < npf; ++inode) {
      const int itrianglenode = hex_edge_facet_table[ics][inode];
      for (int d=0; d < 3; ++d) {
        scscoords[inode][d] = coordv[itrianglenode][d];
      }
    }
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCS::side_node_ordinals(int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return sideNodeOrdinals_[ordinal];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  SIERRA_FORTRAN(hex_scs_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords, areav );

  // all is always well; no error checking
  *error = 0;
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_derivative)
    ( &numIntPoints_,
      &intgLoc_[0], deriv );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative HexSCS volume.." << std::endl;
 }

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(hex_derivative)
    ( &numIntPoints_,
      &intgLocShift_[0], deriv );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative HexSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::face_grad_op(
  const int face_ordinal,
  const bool shifted,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using traits = AlgTraitsQuad4Hex8;
  const double *exp_face = shifted ? &intgExpFaceShift_[0][0][0] : &intgExpFace_[0][0][0];

  constexpr int derivSize = traits::numFaceIp_ * traits::nodesPerElement_ * traits::nDim_;
  DoubleType psi[derivSize];
  SharedMemView<DoubleType***> deriv(psi, traits::numFaceIp_, traits::nodesPerElement_, traits::nDim_);

  const int offset = traits::numFaceIp_ * traits::nDim_ * face_ordinal;
  hex8_derivative(traits::numFaceIp_, &exp_face[offset], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

void HexSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  constexpr bool shifted = false;
  face_grad_op(face_ordinal, shifted, coords, gradop);
}

void HexSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  int npf = 4;

  const int nface = 1;
  double dpsi[24];

  for ( int n=0; n<nelem; n++ ) {

    for ( int k=0; k<npf; k++ ) {

      SIERRA_FORTRAN(hex_derivative)
        ( &nface,
          intgExpFace_[face_ordinal][k], dpsi );

      SIERRA_FORTRAN(hex_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[24*n], &gradop[k*nelem*24+n*24], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "sorry, issue with face_grad_op.." << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  constexpr bool shifted = true;
  face_grad_op(face_ordinal, shifted, coords, gradop);
}

void HexSCS::shifted_face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  int npf = 4;

  const int nface = 1;
  double dpsi[24];

  for ( int n=0; n<nelem; n++ ) {

    for ( int k=0; k<npf; k++ ) {

      SIERRA_FORTRAN(hex_derivative)
        ( &nface,
          intgExpFaceShift_[face_ordinal][k], dpsi );

      SIERRA_FORTRAN(hex_gradient_operator)
        ( &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[24*n], &gradop[k*nelem*24+n*24], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "sorry, issue with face_grad_op.." << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::gij(
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
//-------- gij -------------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::gij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_gij_3d<AlgTraitsHex8>(deriv, coords, gupper, glower);
}

//--------------------------------------------------------------------------
//-------- Mij -------------------------------------------------------------
//--------------------------------------------------------------------------
void HexSCS::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsHex8>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void HexSCS::Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_Mij_3d<AlgTraitsHex8>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}

//--------------------------------------------------------------------------
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
const int *
HexSCS::scsIpEdgeOrd()
{
  return &scsIpEdgeOrd_[0];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLoc_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::shifted_shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(hex_shape_fcn)
    (&numIntPoints_,&intgLocShift_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
int
HexSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*4+node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
HexSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*4+node];
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
HexSCS::isInElement(
    const double * elem_nodal_coor,     // (8,3)
    const double * point_coor,          // (3)
    double * par_coor )
{
  const int maxNonlinearIter = 20;
  const double isInElemConverged = 1.0e-16;
  // Translate element so that (x,y,z) coordinates of the first node are (0,0,0)

  double x[] = {0.,
        0.125*(elem_nodal_coor[1] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[2] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[3] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[4] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[5] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[6] - elem_nodal_coor[0]),
        0.125*(elem_nodal_coor[7] - elem_nodal_coor[0]) };
  double y[] = {0.,
        0.125*(elem_nodal_coor[9 ] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[10] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[11] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[12] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[13] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[14] - elem_nodal_coor[8]),
        0.125*(elem_nodal_coor[15] - elem_nodal_coor[8]) };
  double z[] = {0.,
        0.125*(elem_nodal_coor[17] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[18] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[19] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[20] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[21] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[22] - elem_nodal_coor[16]),
        0.125*(elem_nodal_coor[23] - elem_nodal_coor[16]) };

  // (xp,yp,zp) is the point at which we're searching for (xi,eta,zeta)
  // (must translate this also)

  double xp = point_coor[0] - elem_nodal_coor[0];
  double yp = point_coor[1] - elem_nodal_coor[8];
  double zp = point_coor[2] - elem_nodal_coor[16];

  // Newton-Raphson iteration for (xi,eta,zeta)
  double j[9];
  double f[3];
  double shapefct[8];
  double xinew = 0.5;     // initial guess
  double etanew = 0.5;
  double zetanew = 0.5;
  double xicur = 0.5;
  double etacur = 0.5;
  double zetacur = 0.5;
  double xidiff[] = { 1.0, 1.0, 1.0 };
  int i = 0;

  do
  {
    j[0]=
      -(1.0-etacur)*(1.0-zetacur)*x[1]
      -(1.0+etacur)*(1.0-zetacur)*x[2]
      +(1.0+etacur)*(1.0-zetacur)*x[3]
      +(1.0-etacur)*(1.0+zetacur)*x[4]
      -(1.0-etacur)*(1.0+zetacur)*x[5]
      -(1.0+etacur)*(1.0+zetacur)*x[6]
      +(1.0+etacur)*(1.0+zetacur)*x[7];

    j[1]=
       (1.0+xicur)*(1.0-zetacur)*x[1]
      -(1.0+xicur)*(1.0-zetacur)*x[2]
      -(1.0-xicur)*(1.0-zetacur)*x[3]
      +(1.0-xicur)*(1.0+zetacur)*x[4]
      +(1.0+xicur)*(1.0+zetacur)*x[5]
      -(1.0+xicur)*(1.0+zetacur)*x[6]
      -(1.0-xicur)*(1.0+zetacur)*x[7];

    j[2]=
       (1.0-etacur)*(1.0+xicur)*x[1]
      +(1.0+etacur)*(1.0+xicur)*x[2]
      +(1.0+etacur)*(1.0-xicur)*x[3]
      -(1.0-etacur)*(1.0-xicur)*x[4]
      -(1.0-etacur)*(1.0+xicur)*x[5]
      -(1.0+etacur)*(1.0+xicur)*x[6]
      -(1.0+etacur)*(1.0-xicur)*x[7];

    j[3]=
      -(1.0-etacur)*(1.0-zetacur)*y[1]
      -(1.0+etacur)*(1.0-zetacur)*y[2]
      +(1.0+etacur)*(1.0-zetacur)*y[3]
      +(1.0-etacur)*(1.0+zetacur)*y[4]
      -(1.0-etacur)*(1.0+zetacur)*y[5]
      -(1.0+etacur)*(1.0+zetacur)*y[6]
      +(1.0+etacur)*(1.0+zetacur)*y[7];

    j[4]=
       (1.0+xicur)*(1.0-zetacur)*y[1]
      -(1.0+xicur)*(1.0-zetacur)*y[2]
      -(1.0-xicur)*(1.0-zetacur)*y[3]
      +(1.0-xicur)*(1.0+zetacur)*y[4]
      +(1.0+xicur)*(1.0+zetacur)*y[5]
      -(1.0+xicur)*(1.0+zetacur)*y[6]
      -(1.0-xicur)*(1.0+zetacur)*y[7];

    j[5]=
       (1.0-etacur)*(1.0+xicur)*y[1]
      +(1.0+etacur)*(1.0+xicur)*y[2]
      +(1.0+etacur)*(1.0-xicur)*y[3]
      -(1.0-etacur)*(1.0-xicur)*y[4]
      -(1.0-etacur)*(1.0+xicur)*y[5]
      -(1.0+etacur)*(1.0+xicur)*y[6]
      -(1.0+etacur)*(1.0-xicur)*y[7];

    j[6]=
      -(1.0-etacur)*(1.0-zetacur)*z[1]
      -(1.0+etacur)*(1.0-zetacur)*z[2]
      +(1.0+etacur)*(1.0-zetacur)*z[3]
      +(1.0-etacur)*(1.0+zetacur)*z[4]
      -(1.0-etacur)*(1.0+zetacur)*z[5]
      -(1.0+etacur)*(1.0+zetacur)*z[6]
      +(1.0+etacur)*(1.0+zetacur)*z[7];

    j[7]=
       (1.0+xicur)*(1.0-zetacur)*z[1]
      -(1.0+xicur)*(1.0-zetacur)*z[2]
      -(1.0-xicur)*(1.0-zetacur)*z[3]
      +(1.0-xicur)*(1.0+zetacur)*z[4]
      +(1.0+xicur)*(1.0+zetacur)*z[5]
      -(1.0+xicur)*(1.0+zetacur)*z[6]
      -(1.0-xicur)*(1.0+zetacur)*z[7];

    j[8]=
       (1.0-etacur)*(1.0+xicur)*z[1]
      +(1.0+etacur)*(1.0+xicur)*z[2]
      +(1.0+etacur)*(1.0-xicur)*z[3]
      -(1.0-etacur)*(1.0-xicur)*z[4]
      -(1.0-etacur)*(1.0+xicur)*z[5]
      -(1.0+etacur)*(1.0+xicur)*z[6]
      -(1.0+etacur)*(1.0-xicur)*z[7];

    double jdet=-(j[2]*j[4]*j[6])+j[1]*j[5]*j[6]+j[2]*j[3]*j[7]-
      j[0]*j[5]*j[7]-j[1]*j[3]*j[8]+j[0]*j[4]*j[8];

    if (!jdet) {
      i = maxNonlinearIter;
      break;
    }
    shapefct[0]=(1.0-etacur)*(1.0-xicur)*(1.0-zetacur);

    shapefct[1]=(1.0-etacur)*(1.0+xicur)*(1.0-zetacur);

    shapefct[2]=(1.0+etacur)*(1.0+xicur)*(1.0-zetacur);

    shapefct[3]=(1.0+etacur)*(1.0-xicur)*(1.0-zetacur);

    shapefct[4]=(1.0-etacur)*(1.0-xicur)*(1.0+zetacur);

    shapefct[5]=(1.0-etacur)*(1.0+xicur)*(1.0+zetacur);

    shapefct[6]=(1.0+etacur)*(1.0+xicur)*(1.0+zetacur);

    shapefct[7]=(1.0+etacur)*(1.0-xicur)*(1.0+zetacur);

    f[0]=xp-shapefct[1]*x[1]-shapefct[2]*x[2]-shapefct[3]*x[3]-shapefct[4]*x[4]-\
      shapefct[5]*x[5]-shapefct[6]*x[6]-shapefct[7]*x[7];

    f[1]=yp-shapefct[1]*y[1]-shapefct[2]*y[2]-shapefct[3]*y[3]-shapefct[4]*y[4]-\
      shapefct[5]*y[5]-shapefct[6]*y[6]-shapefct[7]*y[7];

    f[2]=zp-shapefct[1]*z[1]-shapefct[2]*z[2]-shapefct[3]*z[3]-shapefct[4]*z[4]-\
      shapefct[5]*z[5]-shapefct[6]*z[6]-shapefct[7]*z[7];

    xinew = (jdet*xicur+f[2]*(j[2]*j[4]-j[1]*j[5])-f[1]*j[2]*j[7]+f[0]*j[5]*j[7]+
       f[1]*j[1]*j[8]-f[0]*j[4]*j[8])/jdet;

    etanew = (etacur*jdet+f[2]*(-(j[2]*j[3])+j[0]*j[5])+f[1]*j[2]*j[6]-f[0]*j[5]*j[6]-
        f[1]*j[0]*j[8]+f[0]*j[3]*j[8])/jdet;

    zetanew = (jdet*zetacur+f[2]*(j[1]*j[3]-j[0]*j[4])-f[1]*j[1]*j[6]+
         f[0]*j[4]*j[6]+f[1]*j[0]*j[7]-f[0]*j[3]*j[7])/jdet;

    xidiff[0] = xinew - xicur;
    xidiff[1] = etanew - etacur;
    xidiff[2] = zetanew - zetacur;
    xicur = xinew;
    etacur = etanew;
    zetacur = zetanew;

  }
  while ( !within_tolerance( vector_norm_sq(xidiff,3), isInElemConverged) && ++i < maxNonlinearIter);

  par_coor[0] = par_coor[1] = par_coor[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (i <maxNonlinearIter) {
    par_coor[0] = xinew;
    par_coor[1] = etanew;
    par_coor[2] = zetanew;

    std::array<double,3> xtmp;
    xtmp[0] = par_coor[0];
    xtmp[1] = par_coor[1];
    xtmp[2] = par_coor[2];
    dist = parametric_distance(xtmp);
  }
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::interpolatePoint(
    const int  & ncomp_field,
    const double * par_coord,           // (3)
    const double * field,               // (8,ncomp_field)
    double * result ) // (ncomp_field)
{
  // 'field' is a flat array of dimension (8,ncomp_field) (Fortran ordering);
  double xi   = par_coord[0];
  double eta  = par_coord[1];
  double zeta = par_coord[2];

  // NOTE: this uses a [-1,1] definition of the reference element,
  // contrary to the rest of the code

  for ( int i = 0; i < ncomp_field; i++ )
  {
    // Base 'field array' index for ith component
    int b = 8*i;

    result[i] = 0.125 * (
        (1 - xi) * (1 - eta) * (1 - zeta) * field[b + 0]
      + (1 + xi) * (1 - eta) * (1 - zeta) * field[b + 1]
      + (1 + xi) * (1 + eta) * (1 - zeta) * field[b + 2]
      + (1 - xi) * (1 + eta) * (1 - zeta) * field[b + 3]
      + (1 - xi) * (1 - eta) * (1 + zeta) * field[b + 4]
      + (1 + xi) * (1 - eta) * (1 + zeta) * field[b + 5]
      + (1 + xi) * (1 + eta) * (1 + zeta) * field[b + 6]
      + (1 - xi) * (1 + eta) * (1 + zeta) * field[b + 7]
    );
  }

}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  // -1:1 isoparametric range
  const double npe = nodesPerElement_;
  for ( int ip = 0; ip < numIp; ++ip ) {

    const int rowIpc = 3*ip;
    const int rowSfc = npe*ip;

    const double s1 = isoParCoord[rowIpc];
    const double s2 = isoParCoord[rowIpc+1];
    const double s3 = isoParCoord[rowIpc+2];
    shpfc[rowSfc  ] = 0.125*(1.0-s1)*(1.0-s2)*(1.0-s3);
    shpfc[rowSfc+1] = 0.125*(1.0+s1)*(1.0-s2)*(1.0-s3);
    shpfc[rowSfc+2] = 0.125*(1.0+s1)*(1.0+s2)*(1.0-s3);
    shpfc[rowSfc+3] = 0.125*(1.0-s1)*(1.0+s2)*(1.0-s3);
    shpfc[rowSfc+4] = 0.125*(1.0-s1)*(1.0-s2)*(1.0+s3);
    shpfc[rowSfc+5] = 0.125*(1.0+s1)*(1.0-s2)*(1.0+s3);
    shpfc[rowSfc+6] = 0.125*(1.0+s1)*(1.0+s2)*(1.0+s3);
    shpfc[rowSfc+7] = 0.125*(1.0-s1)*(1.0+s2)*(1.0+s3);
  }
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::general_face_grad_op(
  const int  /* face_ordinal */,
  const double *isoParCoord,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int nface = 1;

  double dpsi[24];

  SIERRA_FORTRAN(hex_derivative)
    ( &nface, &isoParCoord[0], dpsi );

  SIERRA_FORTRAN(hex_gradient_operator)
    ( &nface,
      &nodesPerElement_,
      &nface,
      dpsi,
      &coords[0], &gradop[0], &det_j[0], error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "HexSCS::general_face_grad_op: issue.." << std::endl;

}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
HexSCS::sidePcoords_to_elemPcoords(
  const int & side_ordinal,
  const int & npoints,
  const double *side_pcoords,
  double *elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = -0.5;
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+1];
    }
    break;
  case 1:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5;
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+1];
    }
    break;
  case 2:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = -0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = 0.5;
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+1];
    }
    break;
  case 3:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = -0.5;
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = 0.5*side_pcoords[2*i+0];
    }
    break;
  case 4:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5*side_pcoords[2*i+1];
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = -0.5;
    }
    break;
  case 5:
    for (int i=0; i<npoints; i++) {
      elem_pcoords[i*3+0] = 0.5*side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = 0.5*side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = 0.5;
    }
    break;
  default:
    throw std::runtime_error("HexSCS::sideMap invalid ordinal");
  }
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double HexSCS::parametric_distance(const std::array<double,3> &x)
{
  std::array<double,3> y;
  for (int i=0; i<3; ++i) {
    y[i] = std::fabs(x[i]);
  }

  double d = 0;
  for (int i=0; i<3; ++i) {
    if (d < y[i]) {
      d = y[i];
    }
  }
  return d;
}

}
}
