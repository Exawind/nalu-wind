/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Tet4CVFEM_h
#define Tet4CVFEM_h

#include<master_element/MasterElement.h>

namespace sierra{
namespace nalu{

// Tet 4 subcontrol volume
class TetSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsTet4;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  TetSCV();
  virtual ~TetSCV();

  const int * ipNodeMap(int ordinal = 0);

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
  
  void tet_shape_fcn(
    const int &npts,
    const double *par_coord, 
    double* shape_fcn);
private:
  const int nDim_ = AlgTraits::nDim_;
  const int nodesPerElement_ = AlgTraits::nodesPerElement_; 
  const int numIntPoints_ = AlgTraits::numScvIp_; 

  // define ip node mappings
  const int ipNodeMap_[4] = { 0, 1, 2, 3};

  // standard integration location
  const double seventeen96ths = 17.0/96.0;
  const double fourfive96ths  = 45.0/96.0;
  const double intgLoc_[4][3] = {
   {seventeen96ths,  seventeen96ths,  seventeen96ths}, // vol 1
   {fourfive96ths,   seventeen96ths,  seventeen96ths}, // vol 2
   {seventeen96ths,  fourfive96ths,   seventeen96ths}, // vol 3
   {seventeen96ths,  seventeen96ths,  fourfive96ths}}; // vol 4

  // shifted
  const double intgLocShift_[4][3] = {
   {0.0,  0.0,  0.0}, 
   {1.0,  0.0,  0.0}, 
   {0.0,  1.0,  0.0}, 
   {0.0,  0.0,  1.0}}; 

};

// Tet 4 subcontrol surface
class TetSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsTet4;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  TetSCS();
  virtual ~TetSCS();

  const int * ipNodeMap(int ordinal = 0);

  virtual void determinant(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType**>&areav);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv);

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

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  void tet_shape_fcn(
    const int &npts,
    const double *par_coord,
    double* shape_fcn);

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

  double parametric_distance(const double* x);

  const int* side_node_ordinals(int sideOrdinal) final;
private:
  const int nDim_ = AlgTraits::nDim_;
  const int nodesPerElement_ = AlgTraits::nodesPerElement_; 
  const int numIntPoints_ = AlgTraits::numScsIp_; 

  const int sideNodeOrdinals_[4][3] = {
     {0, 1, 3}, //ordinal 0
     {1, 2, 3}, //ordinal 1
     {0, 3, 2}, //ordinal 2
     {0, 2, 1}  //ordinal 3
  };

  // define L/R mappings
  const int lrscv_[12] = {0, 1, 1, 2, 0, 2, 0, 3, 1, 3, 2, 3};

  // elem-edge mapping from ip
  const int scsIpEdgeOrd_[6] = {0, 1, 2, 3, 4, 5};

  // define opposing node
  const int oppNode_[4][3] {
    {2, 2, 2},  // face 0
    {0, 0, 0},  // face 1
    {1, 1, 1},  // face 2
    {3, 3, 3}}; // face 3

  // define opposing face
  const int oppFace_[4][3] = {  
   {2,  1,  5},  // face 0 
   {0,  2,  3},  // face 1 
   {0,  4,  1},  // face 2 
   {3,  5,  4}}; // face 3 

  // standard integration location
  const double thirteen36ths = 13.0/36.0;
  const double five36ths = 5.0/36.0;
  const double intgLoc_[18] = {
    thirteen36ths,  five36ths,      five36ths, // surf 1    1->2
    thirteen36ths,  thirteen36ths,  five36ths, // surf 2    2->3
    five36ths,      thirteen36ths,  five36ths, // surf 3    1->3
    five36ths ,     five36ths,      thirteen36ths, // surf 4    1->4
    thirteen36ths,  five36ths,      thirteen36ths, // surf 5    2->4
    five36ths,      thirteen36ths,  thirteen36ths};// surf 6    3->4

  // shifted
  const double intgLocShift_[18] = {
    0.50,  0.00,  0.00, // surf 1    1->2
    0.50,  0.50,  0.00, // surf 2    2->3
    0.00,  0.50,  0.00, // surf 3    1->3
    0.00,  0.00,  0.50, // surf 4    1->4
    0.50,  0.00,  0.50, // surf 5    2->4
    0.00,  0.50,  0.50};// surf 6    3->4

  // exposed face
  const double seven36ths = 7.0/36.0;
  const double eleven18ths = 11.0/18.0;
  const double intgExpFace_[4][3][3] = {
  // face 0: nodes 0,1,3: scs 0, 1, 2
  {{seven36ths,   0.00,  seven36ths},
   {eleven18ths,  0.00,  seven36ths},
   {seven36ths,   0.00,  eleven18ths}},
  // face 1: nodes 1,2,3: scs 0, 1, 2
  {{eleven18ths,  seven36ths,   seven36ths},
   {seven36ths,   eleven18ths,  seven36ths},
   {seven36ths,   seven36ths,   eleven18ths}},
  // face 2: nodes 0,3,2: scs 0, 1, 2
  {{0.00,       seven36ths,   seven36ths},
   {0.00,       seven36ths,   eleven18ths},
   {0.00,       eleven18ths,  seven36ths}},
  //face 3: nodes 0, 2, 1: scs 0, 1, 2
  {{seven36ths,   seven36ths,    0.00},
   {eleven18ths,  seven36ths,    0.00},
   {seven36ths,   eleven18ths,   0.00}}};

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  const int ipNodeMap_[4][3] = { // 3 ips * 4 faces
    {0,  1,  3}, // face 0
    {1,  2,  3}, // face 1
    {0,  3,  2}, // face 2
    {0,  2,  1}};// face 3

  int intgExpFaceShift_[4][3][3];

};

} // namespace nalu
} // namespace Sierra

#endif
