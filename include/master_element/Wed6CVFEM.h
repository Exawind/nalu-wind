/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WED6CVFEM_H
#define WED6CVFEM_H

#include "master_element/MasterElement.h"

#include <array>

namespace sierra {
namespace nalu {

// Wedge 6 subcontrol volume
class WedSCV : public MasterElement
{
public:
  KOKKOS_FUNCTION
  WedSCV();
  KOKKOS_FUNCTION
  virtual ~WedSCV() = default;

  using AlgTraits = AlgTraitsWed6;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType*>& volume);

  void grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv);

  void shifted_grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv);

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void Mij(
    const double *coords,
    double *metric,
    double *deriv);

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  void wedge_shape_fcn(
    const int &npts,
    const double *par_coord,
    double* shape_fcn);
private:
  const int nDim_ = AlgTraits::nDim_;
  const int nodesPerElement_ = AlgTraits::nodesPerElement_; 
  const int numIntPoints_ = AlgTraits::numScvIp_; 

  // define ip node mappings
  const int ipNodeMap_[6] = {0, 1, 2, 3, 4, 5};

  // standard integration location
  const double eleven18ths = 11.0/18.0;
  const double seven36ths = 7.0/36.0;
  const double intgLoc_[18] = {
   seven36ths,   seven36ths,   -0.5, // vol 0
   eleven18ths,  seven36ths,   -0.5, // vol 1
   seven36ths,   eleven18ths,  -0.5, // vol 2
   seven36ths,   seven36ths,   0.5,  // vol 3
   eleven18ths,  seven36ths,   0.5,  // vol 4
   seven36ths,   eleven18ths,  0.5}; // vol 5

  // shifted
  const double intgLocShift_[18] = {
   0.0,   0.0,  -1.0, // vol 0
   1.0,   0.0,  -1.0, // vol 1
   0.0,   1.0,  -1.0, // vol 2
   0.0,   0.0,   1.0, // vol 3
   1.0,   0.0,  1.0,  // vol 4
   0.0,   1.0,  1.0}; // vol 5
};

// Wedge 6 subcontrol surface
class WedSCS : public MasterElement
{
public:
  KOKKOS_FUNCTION
  WedSCS();
  KOKKOS_FUNCTION
  virtual ~WedSCS() = default;

  using AlgTraits = AlgTraitsWed6;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;
  using MasterElement::adjacentNodes;

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType**>& areav);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv);

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void shifted_grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv);

  void shifted_grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void wedge_derivative(
    const int npts,
    const double *intLoc,
    double *deriv);

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void shifted_face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void shifted_face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void gij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv);

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv);

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv);

  void Mij(
    const double *coords,
    double *metric,
    double *deriv);

  virtual const int * adjacentNodes() final;
  
  const int * scsIpEdgeOrd();

  int opposingNodes(
    const int ordinal, const int node);

  int opposingFace(
    const int ordinal, const int node);

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord);

  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

  void wedge_shape_fcn(
    const int &npts,
    const double *par_coord,
    double* shape_fcn);

  void
  general_shape_fcn(const int numIp, const double* isoParCoord, double* shpfc)
  {
    wedge_shape_fcn(numIp, isoParCoord, shpfc);
  }

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

  // helper functions to isInElement
  double parametric_distance( const double X, const double Y);
  double parametric_distance( const std::array<double,3> &x);

  const int* side_node_ordinals(int sideOrdinal) final;
private:
  const int nDim_ = AlgTraits::nDim_;
  const int nodesPerElement_ = AlgTraits::nodesPerElement_; 
  const int numIntPoints_ = AlgTraits::numScsIp_; 

  const int sideNodeOrdinals_[18] = {
      0, 1, 4, 3, // ordinal 0
      1, 2, 5, 4, // ordinal 1
      0, 3, 5, 2, // ordinal 2
      0, 2, 1,    // ordinal 3
      3, 4, 5     // ordinal 4
  };

  // define L/R mappings
  const int lrscv_[18] = {
   0,  1, 
   1,  2, 
   0,  2, 
   3,  4, 
   4,  5, 
   3,  5, 
   0,  3, 
   1,  4, 
   2,  5};

  // elem-edge mapping from ip
  const int scsIpEdgeOrd_[AlgTraits::numScsIp_] = {
   0,  1,  2, 
   3,  4,  5, 
   6,  7,  8};

  // define opposing node
  const int oppNode_[20] = {
   2, 2, 5, 5,   // face 0: nodes 0,1,4,3
   0, 0, 3, 3,   // face 1: nodes 1,2,5,4
   1, 4, 4, 1,   // face 2: nodes 0,3,5,2
   3, 5, 4, -1,  // face 3: nodes 0,2,1  
   0, 1, 2, -1}; // face 4: nodes 3,4,5  

  // define opposing face
  const int oppFace_[20] = {
       2,  1,  4,   5,    // face 0: nodes 0,1,4,3
       0,  2,  5,   3,    // face 1: nodes 1,2,5,4
       0,  3,  4,   1,    // face 2: nodes 0,3,5,2
       6,  8,  7,  -1,   // face 3: nodes 0,2,1  
       6,  7,  8,  -1};  // face 4: nodes 3,4,5  

  // standard integration location
  const double oneSixth = 1.0/6.0;
  const double five12th = 5.0/12.0;
  const double eleven18th = 11.0/18.0;
  const double seven36th = 7.0/36.0;

  const double intgLoc_[27] = {
    five12th,   oneSixth,  -0.50, // surf 1    1->2
    five12th,   five12th,  -0.50, // surf 2    2->3
    oneSixth,   five12th,  -0.50, // surf 3    1->3
    five12th,   oneSixth,   0.50, // surf 4    4->5
    five12th,   five12th,   0.50, // surf 5    5->6
    oneSixth,   five12th,   0.50, // surf 6    4->6
    seven36th,  seven36th,   0.00, // surf 7    1->4
    eleven18th, seven36th,   0.00, // surf 8    2->5
    seven36th,  eleven18th,  0.00}; // surf 9    3->6

  // shifted
  const double intgLocShift_[27] = {
    0.50,   0.00,  -1.00, // surf 1    1->2
    0.50,   0.50,  -1.00, // surf 2    2->3
    0.00,   0.50,  -1.00, // surf 3    1->3
    0.50,   0.00,   1.00, // surf 4    4->5
    0.50,   0.50,   1.00, // surf 5    5->6
    0.00,   0.50,   1.00, // surf 6    4->6
    0.00,   0.00,   0.00, // surf 7    1->4
    1.00,   0.00,   0.00, // surf 8    2->5
    0.00,   1.00,   0.00};// surf 9    3->6

  // exposed face
  const double intgExpFace_[60] = {
   0.25,        0.0,        -0.5,  // surf 0: nodes 0,1,4,3
   0.75,        0.0,        -0.5,  // face 0, surf 1
   0.75,        0.0,         0.5,  // face 0, surf 2
   0.25,        0.0,         0.5,  // face 0, surf 3
   0.75,        0.25,        -0.5, // surf 1: nodes 1,2,5,4
   0.25,        0.75,        -0.5, // face 1, surf 1
   0.25,        0.75,         0.5, // face 1, surf 2
   0.75,        0.25,         0.5, // face 1, surf 3
   0.0,         0.25,        -0.5, // surf 2: nodes 0,3,5,2
   0.0,         0.25,         0.5, // face 2, surf 1 
   0.0,         0.75,         0.5, // face 2, surf 2
   0.0,         0.75,        -0.5, // face 2, surf 3
   seven36th,   seven36th,   -1.0, // surf 3: nodes 0,2,1
   seven36th,   eleven18th,  -1.0, // face 3, surf 1
   eleven18th,  seven36th,   -1.0, // face 3, surf 2
   0.0,         0.0,          0.0, // (blank)
   seven36th,   seven36th,   1.0,  // surf 4: nodes 3,4,5
   eleven18th,  seven36th,   1.0,  // face 4, surf 1
   seven36th,   eleven18th,  1.0,  // face 4, surf 2
   0.0,         0.0,         0.0}; // (blank)

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  const int ipNodeMap_[20]  = { // 4 ips (pick quad) * 5 faces
   0,  1,  4,  3,  // face 0
   1,  2,  5,  4,  // face 1
   0,  3,   5,  2, // face 2
    0,  2,  1,  0, // face 3 empty
    3,  4,  5,  0};// face 4 empty

  // ordinal to offset map.  Really only convenient for the wedge.
  const int sideOffset_[5] = {0, 4, 8, 12, 15};

  double intgExpFaceShift_[54]; // no blanked entries

};


}  // nalu
}  // sierra


#endif /* WED6CVFEM_H */
