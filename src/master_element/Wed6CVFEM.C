/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "master_element/Wed6CVFEM.h"
#include "master_element/MasterElementFunctions.h"
#include "master_element/Hex8GeometryFunctions.h"
#include "FORTRAN_Proto.h"
#include "NaluEnv.h"

#include <array>

namespace sierra {
namespace nalu {

//-------- wed_deriv -------------------------------------------------------
template <typename DerivType>
void wed_deriv(
  const int npts,
  const double* intgLoc,
  DerivType& deriv)
{
  for (int  j = 0; j < npts; ++j) {
    int k  = j*3;

    const DoubleType r  = intgLoc[k];
    const DoubleType s  = intgLoc[k+1];
    const DoubleType t  = 1.0 - r - s;
    const DoubleType xi = intgLoc[k + 2];

    deriv(j,0,0) = -0.5 * (1.0 - xi);  // d(N_1)/ d(r)  = deriv[0]
    deriv(j,0,1) = -0.5 * (1.0 - xi);  // d(N_1)/ d(s)  = deriv[1]
    deriv(j,0,2) = -0.5 * t;           // d(N_1)/ d(xi) = deriv[2]

    deriv(j,1,0) =  0.5 * (1.0 - xi);  // d(N_2)/ d(r)  = deriv[0 + 3]
    deriv(j,1,1) =  0.0;               // d(N_2)/ d(s)  = deriv[1 + 3]
    deriv(j,1,2) = -0.5 * r;           // d(N_2)/ d(xi) = deriv[2 + 3]

    deriv(j,2,0) =  0.0;               // d(N_3)/ d(r)  = deriv[0 + 6]
    deriv(j,2,1) =  0.5 * (1.0 - xi);  // d(N_3)/ d(s)  = deriv[1 + 6]
    deriv(j,2,2) = -0.5 * s;           // d(N_3)/ d(xi) = deriv[2 + 6]

    deriv(j,3,0) = -0.5 * (1.0 + xi);  // d(N_4)/ d(r)  = deriv[0 + 9]
    deriv(j,3,1) = -0.5 * (1.0 + xi);  // d(N_4)/ d(s)  = deriv[1 + 9]
    deriv(j,3,2) =  0.5 * t;           // d(N_4)/ d(xi) = deriv[2 + 9]

    deriv(j,4,0) =  0.5 * (1.0 + xi);  // d(N_5)/ d(r)  = deriv[0 + 12]
    deriv(j,4,1) =  0.0;               // d(N_5)/ d(s)  = deriv[1 + 12]
    deriv(j,4,2) =  0.5 * r;           // d(N_5)/ d(xi) = deriv[2 + 12]

    deriv(j,5,0) =  0.0;               // d(N_6)/ d(r)  = deriv[0 + 15]
    deriv(j,5,1) =  0.5 * (1.0 + xi);  // d(N_6)/ d(s)  = deriv[1 + 15]
    deriv(j,5,2) =  0.5 * s;           // d(N_6)/ d(xi) = deriv[2 + 15]
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WedSCV::WedSCV()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

  // define ip node mappings
  MasterElement::ipNodeMap_.assign(ipNodeMap_, 6+ipNodeMap_);
  // standard integration location
  MasterElement::intgLoc_.assign(intgLoc_, 18+intgLoc_);
  // shifted
  MasterElement::intgLocShift_.assign(intgLocShift_, 18+intgLocShift_);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WedSCV::~WedSCV()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
WedSCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

void WedSCV::determinant(
  SharedMemView<DoubleType**>& coordel,
  SharedMemView<DoubleType*>& volume)
{
  const int wedSubControlNodeTable[6][8] = {
    { 0, 15, 16, 6, 8, 19, 20, 9    },
    { 9, 6, 1, 7, 20, 16, 14, 18    },
    { 8, 9, 7, 2, 19, 20, 18, 17    },
    { 19, 15, 16, 20, 12, 3, 10, 13 },
    { 20, 16, 14, 18, 13, 10, 4, 11 },
    { 19, 20, 18, 17, 12, 13, 11, 5 },
  };

  const double half = 0.5;
  const double one3rd = 1.0/3.0;
  const double one6th = 1.0/6.0;
  DoubleType coords[21][3];
  DoubleType ehexcoords[8][3];
  const int dim[3] = {0, 1, 2};

  // element vertices
  for (int j=0; j < 6; j++)
    for (int k: dim)
      coords[j][k] = coordel(j, k);

  // face 1 (tri)

  // edge midpoints
  for (int k: dim)
    coords[6][k] = half * (coordel(0, k) + coordel(1, k));

  for (int k: dim)
    coords[7][k] = half * (coordel(1, k) + coordel(2, k));

  for (int k: dim)
    coords[8][k] = half * (coordel(2, k) + coordel(0, k));

  // face midpoint
  for (int k: dim)
    coords[9][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(2, k));

  // face 2 (tri)

  // edge midpoints
  for (int k: dim)
    coords[10][k] = half * (coordel(3, k) + coordel(4, k));

  for (int k: dim)
    coords[11][k] = half * (coordel(4, k) + coordel(5, k));

  for (int k: dim)
    coords[12][k] = half * (coordel(5, k) + coordel(3, k));

  // face midpoint
  for (int k: dim)
    coords[13][k] = one3rd * (coordel(3, k) + coordel(4, k) + coordel(5, k));

  // face 3 (quad)

  // edge midpoints
  for (int k: dim)
    coords[14][k] = half * (coordel(1, k) + coordel(4, k));

  for (int k: dim)
    coords[15][k] = half * (coordel(0, k) + coordel(3, k));

  // face midpoint
  for (int k: dim)
    coords[16][k] = 0.25 * (coordel(0, k) + coordel(1, k)
                            + coordel(4, k) + coordel(3, k));

  // face 4 (quad)

  // edge midpoint
  for (int k: dim)
    coords[17][k] = half * (coordel(2, k) + coordel(5, k));

  // face midpoint
  for (int k: dim)
    coords[18][k] = 0.25 * (coordel(1, k) + coordel(4, k)
                            + coordel(5, k) + coordel(2, k));

  // face 5 (quad)

  // face midpoint
  for (int k: dim)
    coords[19][k] = 0.25 * (coordel(5, k) + coordel(3, k)
                            + coordel(0, k) + coordel(2, k));

  // element centroid
  for (int k: dim)
    coords[20][k] = 0.0;
  for (int j=0; j < nodesPerElement_; j++)
    for (int k: dim)
      coords[20][k] += one6th * coordel(j, k);

  // loop over SCVs
  for (int icv=0; icv < numIntPoints_; icv++) {
    for (int inode=0; inode < 8; inode++)
      for (int k: dim)
        ehexcoords[inode][k] = coords[wedSubControlNodeTable[icv][inode]][k];

    // compute volume using an equivalent polyhedron
    volume(icv) = hex_volume_grandy(ehexcoords);
  }
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void WedSCV::grad_op(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop,
  SharedMemView<DoubleType***>& deriv)
{
  wed_deriv(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void WedSCV::shifted_grad_op(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop,
  SharedMemView<DoubleType***>& deriv)
{
  wed_deriv(numIntPoints_, &intgLocShift_[0], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void WedSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(wed_scv_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords,
      volume, error, &lerr );
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCV::shape_fcn(double *shpfc)
{
  wedge_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
WedSCV::shifted_shape_fcn(double *shpfc)
{
  wedge_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}


//--------------------------------------------------------------------------
//-------- wedge_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCV::wedge_shape_fcn(
  const int  &npts,
  const double *isoParCoord,
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    int sixj = 6 * j;
    int k    = 3 * j;
    double r    = isoParCoord[k];
    double s    = isoParCoord[k + 1];
    double t    = 1.0 - r - s;
    double xi   = isoParCoord[k + 2];
    shape_fcn[    sixj] = 0.5 * t * (1.0 - xi);
    shape_fcn[1 + sixj] = 0.5 * r * (1.0 - xi);
    shape_fcn[2 + sixj] = 0.5 * s * (1.0 - xi);
    shape_fcn[3 + sixj] = 0.5 * t * (1.0 + xi);
    shape_fcn[4 + sixj] = 0.5 * r * (1.0 + xi);
    shape_fcn[5 + sixj] = 0.5 * s * (1.0 + xi);
  }
}

//--------------------------------------------------------------------------
//-------- Mij ------------------------------------------------------------
//--------------------------------------------------------------------------
void WedSCV::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsWed6>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void WedSCV::Mij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& metric,
  SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_3d<AlgTraitsWed6>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
WedSCS::WedSCS()
  : MasterElement()
{
  MasterElement::nDim_ = nDim_;
  MasterElement::nodesPerElement_ = nodesPerElement_;
  MasterElement::numIntPoints_ = numIntPoints_;

  // define L/R mappings
  MasterElement::lrscv_.assign(lrscv_, 18+lrscv_);
  // elem-edge mapping from ip
  MasterElement::scsIpEdgeOrd_.assign(scsIpEdgeOrd_, numIntPoints_+scsIpEdgeOrd_); 
  // define opposing node
  MasterElement::oppNode_.assign(oppNode_, 20+oppNode_);
  // define opposing face
  MasterElement::oppFace_.assign(oppFace_, 20+oppFace_);
  MasterElement::intgLoc_.assign(intgLoc_, 27+intgLoc_);
  // shifted
  MasterElement::intgLocShift_.assign(intgLocShift_, 27+intgLocShift_);
  // exposed face
  MasterElement::intgExpFace_.assign(intgExpFace_, 60+intgExpFace_);

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  MasterElement::ipNodeMap_.assign(ipNodeMap_, 20+ipNodeMap_); // 4 ips (pick quad) * 5 faces

  MasterElement::sideOffset_.assign(sideOffset_, 5+sideOffset_);
  const double nodeLocations[6][3] =
  {
      {0.0,0.0, -1.0}, {+1.0, 0.0, -1.0}, {0.0, +1.0, -1.0},
      {0.0,0.0, +1.0}, {+1.0, 0.0, +1.0}, {0.0, +1.0, +1.0}
  };
  int index = 0;
  stk::topology topo = stk::topology::WEDGE_6;
  for (unsigned k = 0; k < topo.num_sides(); ++k) {
    stk::topology side_topo = topo.side_topology(k);
    const int* ordinals = side_node_ordinals(k);
    for (unsigned n = 0; n < side_topo.num_nodes(); ++n) {
      intgExpFaceShift_[3 * index + 0] = nodeLocations[ordinals[n]][0];
      intgExpFaceShift_[3 * index + 1] = nodeLocations[ordinals[n]][1];
      intgExpFaceShift_[3 * index + 2] = nodeLocations[ordinals[n]][2];
      ++index;
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
WedSCS::~WedSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
WedSCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*4];
}

//--------------------------------------------------------------------------
//-------- side_node_ordinals ----------------------------------------------
//--------------------------------------------------------------------------
const int *
WedSCS::side_node_ordinals(
  int ordinal)
{
  // define face_ordinal->node_ordinal mappings for each face (ordinal);
  return &sideNodeOrdinals_[sideOffset_[ordinal]];
}

void WedSCS::determinant(
  SharedMemView<DoubleType**>& coordel,
  SharedMemView<DoubleType**>& areav)
{
  const int wedEdgeFacetTable[9][4] = {
    { 6 ,  9 ,  20 ,  16   }, // sc face 1 -- points from 1 -> 2
    { 7 ,  9 ,  20 ,  18   }, // sc face 2 -- points from 2 -> 3
    { 9 ,  8 ,  19 ,  20   }, // sc face 3 -- points from 1 -> 3
    { 10 ,  16 ,  20 ,  13 }, // sc face 4 -- points from 4 -> 5
    { 13 ,  11 ,  18 ,  20 }, // sc face 5 -- points from 5 -> 6
    { 12 ,  13 ,  20 ,  19 }, // sc face 6 -- points from 4 -> 6
    { 15 ,  16 ,  20 ,  19 }, // sc face 7 -- points from 1 -> 4
    { 16 ,  14 ,  18 ,  20 }, // sc face 8 -- points from 2 -> 5
    { 19 ,  20 ,  18 , 17  }  // sc face 9 -- points from 3 -> 6
  };

  const double one3rd = 1.0/3.0;
  const double one6th = 1.0/6.0;
  const double half = 0.5;
  const int dim[3] = {0, 1, 2};
  DoubleType coords[21][3];
  DoubleType scscoords[4][3];

  // element vertices
  for (int j=0; j < 6; j++)
    for (int k: dim)
      coords[j][k] = coordel(j, k);

  // face 1 (tri)

  // edge midpoints
  for (int k: dim)
    coords[6][k] = half * (coordel(0, k) + coordel(1, k));

  for (int k: dim)
    coords[7][k] = half * (coordel(1, k) + coordel(2, k));

  for (int k: dim)
    coords[8][k] = half * (coordel(2, k) + coordel(0, k));

  // face midpoint
  for (int k: dim)
    coords[9][k] = one3rd * (coordel(0, k) + coordel(1, k) + coordel(2, k));

  // face 2 (tri)

  // edge midpoints
  for (int k: dim)
    coords[10][k] = half * (coordel(3, k) + coordel(4, k));

  for (int k: dim)
    coords[11][k] = half * (coordel(4, k) + coordel(5, k));

  for (int k: dim)
    coords[12][k] = half * (coordel(5, k) + coordel(3, k));

  // face midpoint
  for (int k: dim)
    coords[13][k] = one3rd * (coordel(3, k) + coordel(4, k) + coordel(5, k));

  // face 3 (quad)

  // edge midpoints
  for (int k: dim)
    coords[14][k] = half * (coordel(1, k) + coordel(4, k));

  for (int k: dim)
    coords[15][k] = half * (coordel(0, k) + coordel(3, k));

  // face midpoint
  for (int k: dim)
    coords[16][k] = 0.25 * (coordel(0, k) + coordel(1, k)
                            + coordel(4, k) + coordel(3, k));

  // face 4 (quad)

  // edge midpoint
  for (int k: dim)
    coords[17][k] = half * (coordel(2, k) + coordel(5, k));

  // face midpoint
  for (int k: dim)
    coords[18][k] = 0.25 * (coordel(1, k) + coordel(4, k)
                            + coordel(5, k) + coordel(2, k));

  // face 5 (quad)

  // face midpoint
  for (int k: dim)
    coords[19][k] = 0.25 * (coordel(5, k) + coordel(3, k)
                            + coordel(0, k) + coordel(2, k));

  // element centroid
  for (int k: dim)
    coords[20][k] = 0.0;
  for (int j=0; j < nodesPerElement_; j++)
    for (int k: dim)
      coords[20][k] += one6th * coordel(j, k);

  // loop over SCSs
  for (int ics=0; ics < numIntPoints_; ics++) {
    for (int inode=0; inode < 4; inode++)
      for (int k: dim)
           scscoords[inode][k] = coords[wedEdgeFacetTable[ics][inode]][k];
    quad_area_by_triangulation(ics, scscoords, areav);
  }
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void WedSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  SIERRA_FORTRAN(wed_scs_det)
    ( &nelem, &nodesPerElement_, &numIntPoints_, coords, areav );

  // all is always well; no error checking
  *error = 0;
}

void WedSCS::grad_op(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop,
  SharedMemView<DoubleType***>& deriv)
{
  wed_deriv(numIntPoints_, &intgLoc_[0], deriv);

  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

void WedSCS::shifted_grad_op(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop,
  SharedMemView<DoubleType***>& deriv)
{
  wed_deriv(numIntPoints_, &intgLocShift_[0], deriv);

  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
  //wed_grad_op(deriv, coords, gradop);
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void WedSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  wedge_derivative(numIntPoints_, &intgLoc_[0], deriv);

  SIERRA_FORTRAN(wed_gradient_operator) (
      &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative WedSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- shifted_grad_op -------------------------------------------------
//--------------------------------------------------------------------------
void WedSCS::shifted_grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  int lerr = 0;

  wedge_derivative(numIntPoints_, &intgLocShift_[0], deriv);

  SIERRA_FORTRAN(wed_gradient_operator) (
      &nelem,
      &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gradop, det_j, error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "sorry, negative WedSCS volume.." << std::endl;
}

//--------------------------------------------------------------------------
//-------- wedge_derivative --------------------------------------------------
//--------------------------------------------------------------------------
void WedSCS::wedge_derivative(
  const int npts,
  const double *intgLoc,
  double *deriv)
{
  // d3d(c,s,j) = deriv[c + 3*(s + 6*j)] = deriv[c+3s+18j]

  for (int  j = 0; j < npts; ++j) {

    int k  = j*3;
    const int p = 18*j;

    const double r  = intgLoc[k];
    const double s  = intgLoc[k+1];
    const double t  = 1.0 - r - s;
    const double xi = intgLoc[k + 2];

    deriv[0+3*0+p] = -0.5 * (1.0 - xi);  // d(N_1)/ d(r)  = deriv[0]
    deriv[1+3*0+p] = -0.5 * (1.0 - xi);  // d(N_1)/ d(s)  = deriv[1]
    deriv[2+3*0+p] = -0.5 * t;           // d(N_1)/ d(xi) = deriv[2]

    deriv[0+3*1+p] =  0.5 * (1.0 - xi);  // d(N_2)/ d(r)  = deriv[0 + 3]
    deriv[1+3*1+p] =  0.0;               // d(N_2)/ d(s)  = deriv[1 + 3]
    deriv[2+3*1+p] = -0.5 * r;           // d(N_2)/ d(xi) = deriv[2 + 3]

    deriv[0+3*2+p] =  0.0;               // d(N_3)/ d(r)  = deriv[0 + 6]
    deriv[1+3*2+p] =  0.5 * (1.0 - xi);  // d(N_3)/ d(s)  = deriv[1 + 6]
    deriv[2+3*2+p] = -0.5 * s;           // d(N_3)/ d(xi) = deriv[2 + 6]

    deriv[0+3*3+p] = -0.5 * (1.0 + xi);  // d(N_4)/ d(r)  = deriv[0 + 9]
    deriv[1+3*3+p] = -0.5 * (1.0 + xi);  // d(N_4)/ d(s)  = deriv[1 + 9]
    deriv[2+3*3+p] =  0.5 * t;           // d(N_4)/ d(xi) = deriv[2 + 9]

    deriv[0+3*4+p] =  0.5 * (1.0 + xi);  // d(N_5)/ d(r)  = deriv[0 + 12]
    deriv[1+3*4+p] =  0.0;               // d(N_5)/ d(s)  = deriv[1 + 12]
    deriv[2+3*4+p] =  0.5 * r;           // d(N_5)/ d(xi) = deriv[2 + 12]

    deriv[0+3*5+p] =  0.0;               // d(N_6)/ d(r)  = deriv[0 + 15]
    deriv[1+3*5+p] =  0.5 * (1.0 + xi);  // d(N_6)/ d(s)  = deriv[1 + 15]
    deriv[2+3*5+p] =  0.5 * s;           // d(N_6)/ d(xi) = deriv[2 + 15]
  }
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int ndim = 3;
  const int nface = 1;
  double dpsi[18];

  // nodes per face... ordinal 0, 1, 2 are quad faces, 3 and 4 are tri
  const int npf = (face_ordinal < 3 ) ? 4 : 3;

  for ( int n = 0; n < nelem; ++n ) {

    for ( int k=0; k<npf; ++k ) {

      const int row = 12*face_ordinal + k*ndim;
      wedge_derivative(nface, &intgExpFace_[row], dpsi);

      SIERRA_FORTRAN(wed_gradient_operator) (
          &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[18*n], &gradop[k*nelem*18+n*18], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "problem with EwedSCS::face_grad" << std::endl;

    }
  }
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void WedSCS::face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using tri_traits = AlgTraitsTri3Wed6;
  using quad_traits = AlgTraitsQuad4Wed6;
  constexpr int dim = 3;

  constexpr int maxDerivSize = quad_traits::numFaceIp_ *  quad_traits::nodesPerElement_ * dim;
  NALU_ALIGNED DoubleType psi[maxDerivSize];
  const int numFaceIps = (face_ordinal < 3) ? quad_traits::numFaceIp_ : tri_traits::numFaceIp_;
  SharedMemView<DoubleType***> deriv(psi, numFaceIps, AlgTraitsWed6::nodesPerElement_, dim);

  const int offset = quad_traits::numFaceIp_ * face_ordinal;
  wed_deriv(numFaceIps, &intgExpFace_[dim * offset], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}
//--------------------------------------------------------------------------
//-------- shifted_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void WedSCS::shifted_face_grad_op(
  int face_ordinal,
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gradop)
{
  using tri_traits = AlgTraitsTri3Wed6;
  using quad_traits = AlgTraitsQuad4Wed6;
  constexpr int dim = 3;

  constexpr int maxDerivSize = quad_traits::numFaceIp_ *  quad_traits::nodesPerElement_ * dim;
  NALU_ALIGNED DoubleType psi[maxDerivSize];
  const int numFaceIps = (face_ordinal < 3) ? quad_traits::numFaceIp_ : tri_traits::numFaceIp_;
  SharedMemView<DoubleType***> deriv(psi, numFaceIps, AlgTraitsWed6::nodesPerElement_, dim);

  const int offset = sideOffset_[face_ordinal];
  wed_deriv(numFaceIps, &intgExpFaceShift_[dim * offset], deriv);
  generic_grad_op<AlgTraitsWed6>(deriv, coords, gradop);
}

void
WedSCS::shifted_face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int ndim = 3;
  const int nface = 1;
  double dpsi[18];

  // nodes per face... ordinal 0, 1, 2 are quad faces, 3 and 4 are tri
  const int npf = (face_ordinal < 3 ) ? 4 : 3;

  for ( int n = 0; n < nelem; ++n ) {

    for ( int k=0; k<npf; ++k ) {
      // no blank entries for shifted_face_grad_op . . . have to use offset
      const int row = (sideOffset_[face_ordinal]+k)*ndim;
      wedge_derivative(nface, &intgExpFaceShift_[row], dpsi);

      SIERRA_FORTRAN(wed_gradient_operator) (
          &nface,
          &nodesPerElement_,
          &nface,
          dpsi,
          &coords[18*n], &gradop[k*nelem*18+n*18], &det_j[npf*n+k], error, &lerr );

      if ( lerr )
        NaluEnv::self().naluOutput() << "problem with EwedSCS::face_grad" << std::endl;

    }
  }
}


void WedSCS::gij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& gupper,
  SharedMemView<DoubleType***>& glower,
  SharedMemView<DoubleType***>& deriv)
{
  generic_gij_3d<AlgTraitsWed6>(deriv, coords, gupper, glower);
}

//--------------------------------------------------------------------------
//-------- gij ------------------------------------------------------------
//--------------------------------------------------------------------------
void WedSCS::gij(
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
void WedSCS::Mij(
  const double *coords,
  double *metric,
  double *deriv)
{
  generic_Mij_3d<AlgTraitsWed6>(numIntPoints_, deriv, coords, metric);
}
//-------------------------------------------------------------------------
void WedSCS::Mij(
  SharedMemView<DoubleType**>& coords,
  SharedMemView<DoubleType***>& metric,
  SharedMemView<DoubleType***>& deriv)
{
  generic_Mij_3d<AlgTraitsWed6>(deriv, coords, metric);
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
WedSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}

//--------------------------------------------------------------------------
//-------- scsIpEdgeOrd ----------------------------------------------------
//--------------------------------------------------------------------------
const int *
WedSCS::scsIpEdgeOrd()
{
  return &scsIpEdgeOrd_[0];
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
int
WedSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*4+node];
}


//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
WedSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*4+node];
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::shape_fcn(double *shpfc)
{
  wedge_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::shifted_shape_fcn(double *shpfc)
{
  wedge_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
WedSCS::isInElement(
  const double *elemNodalCoord,
  const double *pointCoord,
  double *isoParCoord )
{
  const double isInElemConverged = 1.0e-16;

  // ------------------------------------------------------------------
  // Pentahedron master element space is (r,s,xi):
  // r=([0,1]), s=([0,1]), xi=([-1,+1])
  // Use natural coordinates to determine if point is in pentahedron.
  // ------------------------------------------------------------------

  // Translate element so that (x,y,z) coordinates of first node are (0,0,0)

  double x[] = {0.0,
                elemNodalCoord[ 1] - elemNodalCoord[ 0],
                elemNodalCoord[ 2] - elemNodalCoord[ 0],
                elemNodalCoord[ 3] - elemNodalCoord[ 0],
                elemNodalCoord[ 4] - elemNodalCoord[ 0],
                elemNodalCoord[ 5] - elemNodalCoord[ 0] };
  double y[] = {0.0,
                elemNodalCoord[ 7] - elemNodalCoord[ 6],
                elemNodalCoord[ 8] - elemNodalCoord[ 6],
                elemNodalCoord[ 9] - elemNodalCoord[ 6],
                elemNodalCoord[10] - elemNodalCoord[ 6],
                elemNodalCoord[11] - elemNodalCoord[ 6] };
  double z[] = {0.0,
                elemNodalCoord[13] - elemNodalCoord[12],
                elemNodalCoord[14] - elemNodalCoord[12],
                elemNodalCoord[15] - elemNodalCoord[12],
                elemNodalCoord[16] - elemNodalCoord[12],
                elemNodalCoord[17] - elemNodalCoord[12] };

  // (xp,yp,zp) is the point to be mapped into (r,s,xi) coordinate system.
  // This point must also be translated as above.

  double xp = pointCoord[0] - elemNodalCoord[ 0];
  double yp = pointCoord[1] - elemNodalCoord[ 6];
  double zp = pointCoord[2] - elemNodalCoord[12];

  // Newton-Raphson iteration for (r,s,xi)
  double j[3][3];
  double jinv[3][3];
  double f[3];
  double shapefct[6];
  double rnew   = 1.0 / 3.0; // initial guess (centroid)
  double snew   = 1.0 / 3.0;
  double xinew  = 0.0;
  double rcur   = rnew;
  double scur   = snew;
  double xicur  = xinew;
  double xidiff[] = { 1.0, 1.0, 1.0 };

  double shp_func_deriv[18];
  double current_pc[3];

  const int MAX_NR_ITER = 20;
  int i = 0;
  do
  {
    current_pc[0] = rcur  = rnew;
    current_pc[1] = scur  = snew;
    current_pc[2] = xicur = xinew;

    // Build Jacobian and Invert

    //aj(1,1)=( dN/dr  ) * x[]
    //aj(1,2)=( dN/ds  ) * x[]
    //aj(1,3)=( dN/dxi ) * x[]
    //aj(2,1)=( dN/dr  ) * y[]
    //aj(2,2)=( dN/ds  ) * y[]
    //aj(2,3)=( dN/dxi ) * y[]
    //aj(3,1)=( dN/dr  ) * z[]
    //aj(3,2)=( dN/ds  ) * z[]
    //aj(3,3)=( dN/dxi ) * z[]

    wedge_derivative(1, current_pc, shp_func_deriv);

    for (int row = 0; row != 3; ++row)
      for (int col = 0; col != 3; ++col)
	j[row][col] = 0.0;

    for (int k = 1; k != 6; ++k)
    {
      j[0][0] -= shp_func_deriv[k * 3 + 0] * x[k];
      j[0][1] -= shp_func_deriv[k * 3 + 1] * x[k];
      j[0][2] -= shp_func_deriv[k * 3 + 2] * x[k];

      j[1][0] -= shp_func_deriv[k * 3 + 0] * y[k];
      j[1][1] -= shp_func_deriv[k * 3 + 1] * y[k];
      j[1][2] -= shp_func_deriv[k * 3 + 2] * y[k];

      j[2][0] -= shp_func_deriv[k * 3 + 0] * z[k];
      j[2][1] -= shp_func_deriv[k * 3 + 1] * z[k];
      j[2][2] -= shp_func_deriv[k * 3 + 2] * z[k];
    }

    const double jdet =   j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
		      - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
		      + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);

    jinv[0][0] =  (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / jdet;
    jinv[0][1] = -(j[0][1] * j[2][2] - j[2][1] * j[0][2]) / jdet;
    jinv[0][2] =  (j[1][2] * j[0][1] - j[0][2] * j[1][1]) / jdet;
    jinv[1][0] = -(j[1][0] * j[2][2] - j[2][0] * j[1][2]) / jdet;
    jinv[1][1] =  (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / jdet;
    jinv[1][2] = -(j[0][0] * j[1][2] - j[1][0] * j[0][2]) / jdet;
    jinv[2][0] =  (j[1][0] * j[2][1] - j[2][0] * j[1][1]) / jdet;
    jinv[2][1] = -(j[0][0] * j[2][1] - j[2][0] * j[0][1]) / jdet;
    jinv[2][2] =  (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / jdet;

    wedge_shape_fcn(1, current_pc, shapefct);

    // x[0] = y[0] = z[0] = 0 by construction
    f[0] = xp - (shapefct[1] * x[1] +
		 shapefct[2] * x[2] +
		 shapefct[3] * x[3] +
		 shapefct[4] * x[4] +
		 shapefct[5] * x[5]);
    f[1] = yp - (shapefct[1] * y[1] +
		 shapefct[2] * y[2] +
		 shapefct[3] * y[3] +
		 shapefct[4] * y[4] +
		 shapefct[5] * y[5]);
    f[2] = zp - (shapefct[1] * z[1] +
		 shapefct[2] * z[2] +
		 shapefct[3] * z[3] +
		 shapefct[4] * z[4] +
		 shapefct[5] * z[5]);

    rnew  = rcur  - (f[0] * jinv[0][0] + f[1] * jinv[0][1] + f[2] * jinv[0][2]);
    snew  = scur  - (f[0] * jinv[1][0] + f[1] * jinv[1][1] + f[2] * jinv[1][2]);
    xinew = xicur - (f[0] * jinv[2][0] + f[1] * jinv[2][1] + f[2] * jinv[2][2]);

    xidiff[0] = rnew  - rcur;
    xidiff[1] = snew  - scur;
    xidiff[2] = xinew - xicur;
  }
  while (!within_tolerance(vector_norm_sq(xidiff,3), isInElemConverged) && ++i != MAX_NR_ITER);

  isoParCoord[0] = isoParCoord[1] = isoParCoord[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (i < MAX_NR_ITER)
  {
    isoParCoord[0] = rnew;
    isoParCoord[1] = snew;
    isoParCoord[2] = xinew;
    std::array<double,3> xx = {{isoParCoord[0], isoParCoord[1], isoParCoord[2]}};

    dist = parametric_distance(xx);
  }
  return dist;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result )
{
  double shapefct[6];

  wedge_shape_fcn(1, isoParCoord, shapefct);

  for ( int i = 0; i < nComp; i++)
  {
    // Base 'field array' index for i_th component
    int b = 6 * i;

    result[i] = 0.0;

    for (int j = 0; j != 6; ++j)
      result[i] += shapefct[j] * field[b + j];
  }
}

//--------------------------------------------------------------------------
//-------- wedge_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::wedge_shape_fcn(
  const int  &npts,
  const double *isoParCoord,
  double *shape_fcn)
{
  for (int j = 0; j < npts; ++j ) {
    int sixj = 6 * j;
    int k    = 3 * j;
    double r    = isoParCoord[k];
    double s    = isoParCoord[k + 1];
    double t    = 1.0 - r - s;
    double xi   = isoParCoord[k + 2];
    shape_fcn[    sixj] = 0.5 * t * (1.0 - xi);
    shape_fcn[1 + sixj] = 0.5 * r * (1.0 - xi);
    shape_fcn[2 + sixj] = 0.5 * s * (1.0 - xi);
    shape_fcn[3 + sixj] = 0.5 * t * (1.0 + xi);
    shape_fcn[4 + sixj] = 0.5 * r * (1.0 + xi);
    shape_fcn[5 + sixj] = 0.5 * s * (1.0 + xi);
  }
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
WedSCS::parametric_distance(const double X, const double Y)
{
  const double dist0 = -3*X;
  const double dist1 = -3*Y;
  const double dist2 =  3*(X+Y);
  const double dist = std::max(std::max(dist0,dist1),dist2);
  return dist;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double
WedSCS::parametric_distance(const std::array<double,3> &x)
{
  const double X = x[0] - 1./3.;
  const double Y = x[1] - 1./3.;
  const double Z = x[2] ;
  const double dist_t = parametric_distance(X,Y);
  const double dist_z = std::fabs(Z);
  const double dist = std::max(dist_z, dist_t);
  return dist;
}

//--------------------------------------------------------------------------
//-------- general_face_grad_op --------------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::general_face_grad_op(
  const int  /* face_ordinal */,
  const double *isoParCoord,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  int lerr = 0;
  const int nface = 1;
  double dpsi[18];

  wedge_derivative(nface, &isoParCoord[0], dpsi);

  SIERRA_FORTRAN(wed_gradient_operator)
    ( &nface,
      &nodesPerElement_,
      &nface,
      dpsi,
      &coords[0], &gradop[0], &det_j[0], error, &lerr );

  if ( lerr )
    NaluEnv::self().naluOutput() << "problem with EwedSCS::general_face_grad" << std::endl;

}

//--------------------------------------------------------------------------
//-------- sidePcoords_to_elemPcoords --------------------------------------
//--------------------------------------------------------------------------
void
WedSCS::sidePcoords_to_elemPcoords(
  const int & side_ordinal,
  const int & npoints,
  const double *side_pcoords,
  double *elem_pcoords)
{
  switch (side_ordinal) {
  case 0:
    for (int i=0; i<npoints; i++) {//face0:quad: (x,y) -> (0.5*(1 + x),0,y)
      elem_pcoords[i*3+0] = 0.5*(1.0+side_pcoords[2*i+0]);
      elem_pcoords[i*3+1] = 0.0;
      elem_pcoords[i*3+2] = side_pcoords[2*i+1];
    }
    break;
  case 1:
    for (int i=0; i<npoints; i++) {//face1:quad: (x,y) -> (0.5*(1-y),0.5*(1 + y),x)
      elem_pcoords[i*3+0] = 0.5*(1.0-side_pcoords[2*i+0]);
      elem_pcoords[i*3+1] = 0.5*(1.0+side_pcoords[2*i+0]);
      elem_pcoords[i*3+2] = side_pcoords[2*i+1];
    }
    break;
  case 2:
    for (int i=0; i<npoints; i++) {//face2:quad: (x,y) -> (0,0.5*(1 + x),y)
      elem_pcoords[i*3+0] = 0.0;
      elem_pcoords[i*3+1] = 0.5*(1.0+side_pcoords[2*i+1]);
      elem_pcoords[i*3+2] = side_pcoords[2*i+0];
    }
    break;
  case 3:
    for (int i=0; i<npoints; i++) {//face3:tri: (x,y) -> (x,y,-1)
      elem_pcoords[i*3+0] = side_pcoords[2*i+1];
      elem_pcoords[i*3+1] = side_pcoords[2*i+0];
      elem_pcoords[i*3+2] = -1.0;
    }
    break;
  case 4:
    for (int i=0; i<npoints; i++) {//face4:tri: (x,y) -> (x,y,+1 )
      elem_pcoords[i*3+0] = side_pcoords[2*i+0];
      elem_pcoords[i*3+1] = side_pcoords[2*i+1];
      elem_pcoords[i*3+2] = 1.0;
    }
    break;
  default:
    throw std::runtime_error("WedSCS::sidePcoords_to_elemPcoords invalid ordinal");
  }
}

}  // nalu
}  // sierra
