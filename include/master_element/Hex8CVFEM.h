/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Hex8CVFEM_h
#define Hex8CVFEM_h

#include <array>

#include<master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>

namespace sierra{
namespace nalu{

template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void hex8_shape_fcn(
  const int    & npts,
  const double * isoParCoord,
  ViewType &shape_fcn)
{
  const DoubleType half   = 0.50;
  const DoubleType one4th = 0.25;
  const DoubleType one8th = 0.125;
  for ( int j = 0; j < npts; ++j ) {

    const DoubleType s1 = isoParCoord[j*3];
    const DoubleType s2 = isoParCoord[j*3+1];
    const DoubleType s3 = isoParCoord[j*3+2];

    shape_fcn(j,0) = one8th + one4th*(-s1 - s2 - s3) + half*( s2*s3 + s3*s1 + s1*s2 ) - s1*s2*s3;
    shape_fcn(j,1) = one8th + one4th*( s1 - s2 - s3) + half*( s2*s3 - s3*s1 - s1*s2 ) + s1*s2*s3;
    shape_fcn(j,2) = one8th + one4th*( s1 + s2 - s3) + half*(-s2*s3 - s3*s1 + s1*s2 ) - s1*s2*s3;
    shape_fcn(j,3) = one8th + one4th*(-s1 + s2 - s3) + half*(-s2*s3 + s3*s1 - s1*s2 ) + s1*s2*s3;
    shape_fcn(j,4) = one8th + one4th*(-s1 - s2 + s3) + half*(-s2*s3 - s3*s1 + s1*s2 ) + s1*s2*s3;
    shape_fcn(j,5) = one8th + one4th*( s1 - s2 + s3) + half*(-s2*s3 + s3*s1 - s1*s2 ) - s1*s2*s3;
    shape_fcn(j,6) = one8th + one4th*( s1 + s2 + s3) + half*( s2*s3 + s3*s1 + s1*s2 ) + s1*s2*s3;
    shape_fcn(j,7) = one8th + one4th*(-s1 + s2 + s3) + half*( s2*s3 - s3*s1 - s1*s2 ) - s1*s2*s3;
  }
}

// Hex 8 subcontrol volume
class HexSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex8;

  HexSCV();
  virtual ~HexSCV();

  const int * ipNodeMap(int ordinal = 0);

  using MasterElement::determinant;
  using MasterElement::shifted_grad_op;
  using MasterElement::shifted_shape_fcn;
  using MasterElement::shape_fcn;

  // NGP-ready methods first
  void determinant(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType*>& volume);

  void grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv);

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error );

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void Mij(
    const double *coords,
    double *metric,
    double *deriv);


  template<typename ViewType>
  KOKKOS_FUNCTION
  void shape_fcn(ViewType &shpfc);

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);


  const int nDim_ = 3;
  const int nodesPerElement_ = 8;
  const int numIntPoints_ = 8;
 
   // define ip node mappings
  const int ipNodeMap_[8] = {0, 1, 2, 3, 4, 5, 6, 7};
 
   // standard integration location
  const double intgLoc_[24] = {
   -0.25,  -0.25,  -0.25,
   +0.25,  -0.25,  -0.25,
   +0.25,  +0.25,  -0.25,
   -0.25,  +0.25,  -0.25,
   -0.25,  -0.25,  +0.25,
   +0.25,  -0.25,  +0.25,
   +0.25,  +0.25,  +0.25,
   -0.25,  +0.25,  +0.25};
 
  // shifted integration location
  const double intgLocShift_[24] = {
   -0.5,  -0.5,  -0.5,
   +0.5,  -0.5,  -0.5,
   +0.5,  +0.5,  -0.5,
   -0.5,  +0.5,  -0.5,
   -0.5,  -0.5,  +0.5,
   +0.5,  -0.5,  +0.5,
   +0.5,  +0.5,  +0.5,
   -0.5,  +0.5,  +0.5};

};

// Hex 8 subcontrol surface
class HexSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex8;
  using AlgTraitsFace = AlgTraitsQuad4;
  using MasterElement::adjacentNodes;


  HexSCS();
  virtual ~HexSCS();

  const int * ipNodeMap(int ordinal = 0);

  using MasterElement::determinant;

  // NGP-ready methods first
  void shape_fcn(
    SharedMemView<DoubleType**> &shpfc);

  template<typename ViewType>
  KOKKOS_FUNCTION
  void shape_fcn(ViewType &shpfc);

  void shifted_shape_fcn(
    SharedMemView<DoubleType**> &shpfc);

  void hex8_gradient_operator(
    const int nodesPerElem,
    const int numIntgPts,
    SharedMemView<DoubleType***> &deriv,
    SharedMemView<DoubleType**> &cordel,
    SharedMemView<DoubleType***> &gradop,
    SharedMemView<DoubleType*> &det_j,
    DoubleType &error,
    int &lerr);

  template<typename ViewTypeCoord, typename ViewTypeGrad>
  KOKKOS_FUNCTION
  void grad_op(
    ViewTypeCoord& coords,
    ViewTypeGrad&  gradop,
    ViewTypeGrad&  deriv);

  void grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void shifted_face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void determinant(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType**>&areav);

  void gij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv);

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv);

  // non NGP-ready methods second
  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void shifted_grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void shifted_face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv);

  void Mij(
    const double *coords,
    double *metric,
    double *deriv);

  virtual const int * adjacentNodes() final;

  const int * scsIpEdgeOrd();

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  int opposingNodes(
    const int ordinal, const int node);

  int opposingFace(
    const int ordinal, const int node);

  double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord);

  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc);

  void general_face_grad_op(
    const int face_ordinal,
    const double *isoParCoord,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void sidePcoords_to_elemPcoords(
    const int & side_ordinal,
    const int & npoints,
    const double *side_pcoords,
    double *elem_pcoords);

  const int* side_node_ordinals(int sideOrdinal) final;
  using MasterElement::side_node_ordinals;

  double parametric_distance(const std::array<double,3> &x);

  const int nDim_            = 3;
  const int nodesPerElement_ = 8;
  const int numIntPoints_   = 12;
  const double scaleToStandardIsoFac_ = 2.0;

  // standard integration location
  const double  intgLoc_  [36] = {
    0.00,  -0.25,  -0.25, // surf 1    1->2
    0.25,   0.00,  -0.25, // surf 2    2->3
    0.00,   0.25,  -0.25, // surf 3    3->4
   -0.25,   0.00,  -0.25, // surf 4    1->4
    0.00,  -0.25,   0.25, // surf 5    5->6
    0.25,   0.00,   0.25, // surf 6    6->7
    0.00,   0.25,   0.25, // surf 7    7->8
   -0.25,   0.00,   0.25, // surf 8    5->8
   -0.25,  -0.25,   0.00, // surf 9    1->5
    0.25,  -0.25,   0.00, // surf 10   2->6
    0.25,   0.25,   0.00, // surf 11   3->7
   -0.25,   0.25,   0.00};// surf 12   4->8

private:

  // define L/R mappings
  const int     lrscv_    [24]={0,1,1,2,2,3,0,3,4,5,5,6,6,7,4,7,0,4,1,5,2,6,3,7};

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  const int     ipNodeMap_[24]={
  /* face 0 */ 0, 1, 5, 4, 
  /* face 1 */ 1, 2, 6, 5, 
  /* face 2 */ 2, 3, 7, 6, 
  /* face 3 */ 0, 4, 7, 3, 
  /* face 4 */ 0, 3, 2, 1, 
  /* face 5 */ 4, 5, 6, 7};

  // define opposing node
  const int     oppNode_  [24]={
  /* face 0 */ 3, 2, 6, 7,
  /* face 1 */ 0, 3, 7, 4,
  /* face 2 */ 1, 0, 4, 5,
  /* face 3 */ 1, 5, 6, 2,
  /* face 4 */ 4, 7, 6, 5,
  /* face 5 */ 0, 1, 2, 3};

  // define opposing face
  const int     oppFace_  [24]={
  /* face 0 */  3,  1,  5,  7,
  /* face 1 */  0,  2,  6,  4,
  /* face 2 */  1,  3,  7,  5,
  /* face 3 */  0,  4,  6,  2,
  /* face 4 */  8, 11, 10,  9,
  /* face 5 */  8,  9, 10, 11};

  // shifted
  const double  intgLocShift_  [36] = {
    0.00,  -0.50,  -0.50, // surf 1    1->2
    0.50,   0.00,  -0.50, // surf 2    2->3
    0.00,   0.50,  -0.50, // surf 3    3->4
   -0.50,   0.00,  -0.50, // surf 4    1->4
    0.00,  -0.50,   0.50, // surf 5    5->6
    0.50,   0.00,   0.50, // surf 6    6->7
    0.00,   0.50,   0.50, // surf 7    7->8
   -0.50,   0.00,   0.50, // surf 8    5->8
   -0.50,  -0.50,   0.00, // surf 9    1->5
    0.50,  -0.50,   0.00, // surf 10   2->6
    0.50,   0.50,   0.00, // surf 11   3->7
   -0.50,   0.50,   0.00};// surf 12   4->8

  // exposed face
  const double  intgExpFace_[6][4][3] = {
  // face 0; scs 0, 1, 2, 3
 {{-0.25,  -0.50,  -0.25},
  { 0.25,  -0.50,  -0.25},
  { 0.25,  -0.50,   0.25},
  {-0.25,  -0.50,   0.25}},
  // face 1; scs 0, 1, 2, 3
 {{ 0.50,  -0.25,  -0.25},
  { 0.50,   0.25,  -0.25},
  { 0.50,   0.25,   0.25},
  { 0.50,  -0.25,   0.25}},
  // face 2; scs 0, 1, 2, 3
 {{ 0.25,   0.50,  -0.25},
  {-0.25,   0.50,  -0.25},
  {-0.25,   0.50,   0.25},
  { 0.25,   0.50,   0.25}},
  // face 3; scs 0, 1, 2, 3
 {{-0.50,  -0.25,  -0.25},
  {-0.50,  -0.25,   0.25},
  {-0.50,   0.25,   0.25},
  {-0.50,   0.25,  -0.25}},
  // face 4; scs 0, 1, 2, 3
 {{-0.25,  -0.25,  -0.50},
  {-0.25,   0.25,  -0.50},
  { 0.25,   0.25,  -0.50},
  { 0.25,  -0.25,  -0.50}},
  // face 5; scs 0, 1, 2, 3
 {{-0.25,  -0.25,   0.50},
  { 0.25,  -0.25,   0.50},
  { 0.25,   0.25,   0.50},
  {-0.25,   0.25,   0.50}}};

  const double  intgExpFaceShift_[6][4][3] = {
 {{ -0.5,   -0.5,   -0.5},
  {  0.5,   -0.5,   -0.5},
  {  0.5,   -0.5,    0.5},
  { -0.5,   -0.5,    0.5}},
 {{  0.5,   -0.5,   -0.5},
  {  0.5,    0.5,   -0.5},
  {  0.5,    0.5,    0.5},
  {  0.5,   -0.5,    0.5}},
 {{  0.5,    0.5,   -0.5},
  { -0.5,    0.5,   -0.5},
  { -0.5,    0.5,    0.5},
  {  0.5,    0.5,    0.5}},
 {{ -0.5,   -0.5,   -0.5},
  { -0.5,   -0.5,    0.5},
  { -0.5,    0.5,    0.5},
  { -0.5,    0.5,   -0.5}},
 {{ -0.5,   -0.5,   -0.5},
  { -0.5,    0.5,   -0.5},
  {  0.5,    0.5,   -0.5},
  {  0.5,   -0.5,   -0.5}},
 {{ -0.5,   -0.5,    0.5},
  {  0.5,   -0.5,    0.5},
  {  0.5,    0.5,    0.5},
  { -0.5,    0.5,    0.5}}};

  // nodes for collocation calculations
  const double  nodeLoc_ [8][3] = {
  /* node 0 */{-0.5,  -0.5,  -0.5},
  /* node 1 */{ 0.5,  -0.5,  -0.5},
  /* node 2 */{ 0.5,   0.5,  -0.5},
  /* node 3 */{-0.5,   0.5,  -0.5},
  /* node 4 */{-0.5,  -0.5,   0.5},
  /* node 5 */{ 0.5,  -0.5,   0.5},
  /* node 6 */{ 0.5,   0.5,   0.5},
  /* node 7 */{-0.5,   0.5,   0.5}};

  // mapping from a side ordinal to the node ordinals on that side
  const int   sideNodeOrdinals_[6][4] = {
     {0, 1, 5, 4}, // ordinal 0
     {1, 2, 6, 5}, // ordinal 1
     {2, 3, 7, 6}, // ordinal 2
     {0, 4, 7, 3}, // ordinal 3
     {0, 3, 2, 1}, // ordinal 4
     {4, 5, 6, 7}};// ordinal 5

  // elem-edge mapping from ip
  const int  scsIpEdgeOrd_  [12]={0,1,2,3,4,5,6,7,8,9,10,11};

private :

  void face_grad_op(
    const int face_ordinal,
    const bool shifted,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop);
};
    
//-------- hex8_derivative -------------------------------------------------
template <typename DerivType>
KOKKOS_FUNCTION
void hex8_derivative(
  const int npts,
  const double *intgLoc,
  DerivType &deriv)
{
  const DoubleType half = 0.50;
  const DoubleType one4th = 0.25;
  for (int  ip = 0; ip < npts; ++ip) {
    const DoubleType s1 = intgLoc[ip*3];
    const DoubleType s2 = intgLoc[ip*3+1];
    const DoubleType s3 = intgLoc[ip*3+2];
    const DoubleType s1s2 = s1*s2;
    const DoubleType s2s3 = s2*s3;
    const DoubleType s1s3 = s1*s3;

    // shape function derivative in the s1 direction -
    deriv(ip,0,0) = half*( s3 + s2 ) - s2s3 - one4th;
    deriv(ip,1,0) = half*(-s3 - s2 ) + s2s3 + one4th;
    deriv(ip,2,0) = half*(-s3 + s2 ) - s2s3 + one4th;
    deriv(ip,3,0) = half*( s3 - s2 ) + s2s3 - one4th;
    deriv(ip,4,0) = half*(-s3 + s2 ) + s2s3 - one4th;
    deriv(ip,5,0) = half*( s3 - s2 ) - s2s3 + one4th;
    deriv(ip,6,0) = half*( s3 + s2 ) + s2s3 + one4th;
    deriv(ip,7,0) = half*(-s3 - s2 ) - s2s3 - one4th;

    // shape function derivative in the s2 direction -
    deriv(ip,0,1) = half*( s3 + s1 ) - s1s3 - one4th;
    deriv(ip,1,1) = half*( s3 - s1 ) + s1s3 - one4th;
    deriv(ip,2,1) = half*(-s3 + s1 ) - s1s3 + one4th;
    deriv(ip,3,1) = half*(-s3 - s1 ) + s1s3 + one4th;
    deriv(ip,4,1) = half*(-s3 + s1 ) + s1s3 - one4th;
    deriv(ip,5,1) = half*(-s3 - s1 ) - s1s3 - one4th;
    deriv(ip,6,1) = half*( s3 + s1 ) + s1s3 + one4th;
    deriv(ip,7,1) = half*( s3 - s1 ) - s1s3 + one4th;

    // shape function derivative in the s3 direction -
    deriv(ip,0,2) = half*( s2 + s1 ) - s1s2 - one4th;
    deriv(ip,1,2) = half*( s2 - s1 ) + s1s2 - one4th;
    deriv(ip,2,2) = half*(-s2 - s1 ) - s1s2 - one4th;
    deriv(ip,3,2) = half*(-s2 + s1 ) + s1s2 - one4th;
    deriv(ip,4,2) = half*(-s2 - s1 ) + s1s2 + one4th;
    deriv(ip,5,2) = half*(-s2 + s1 ) - s1s2 + one4th;
    deriv(ip,6,2) = half*( s2 + s1 ) + s1s2 + one4th;
    deriv(ip,7,2) = half*( s2 - s1 ) - s1s2 + one4th;
  }
}

template<typename ViewType>
KOKKOS_FUNCTION
void
HexSCV::shape_fcn(ViewType &shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

template<typename ViewTypeCoord, typename ViewTypeGrad>
KOKKOS_FUNCTION
void HexSCS::grad_op(
  ViewTypeCoord& coords,
  ViewTypeGrad&  gradop,
  ViewTypeGrad&  deriv)
{
  hex8_derivative(numIntPoints_, &intgLoc_[0], deriv);
  generic_grad_op<AlgTraitsHex8>(deriv, coords, gradop);
}

template<typename ViewType>
KOKKOS_FUNCTION
void
HexSCS::shape_fcn(ViewType &shpfc)
{
  hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

} // namespace nalu
} // namespace Sierra

#endif
