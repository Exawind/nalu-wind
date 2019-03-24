/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Quad42DCVFEM_h   
#define Quad42DCVFEM_h   

#include <master_element/MasterElement.h>

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>

namespace sierra{
namespace nalu{

// 2D Quad 4 subcontrol volume
class Quad42DSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad4_2D;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  Quad42DSCV();
  KOKKOS_FUNCTION
  virtual ~Quad42DSCV() = default;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    SharedMemView<DoubleType**> &coords,
    SharedMemView<DoubleType*> &vol) override ;

  void grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void shifted_grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void Mij(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv) override ;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) override ;

  void Mij(
     const double *coords,
     double *metric,
     double *deriv) override ;

  void shape_fcn(
    double *shpfc) override ;

  void shifted_shape_fcn(
    double *shpfc) override ;
  
  virtual const double* integration_locations() const final {
    return intgLoc_;
  }
  virtual const double* integration_location_shift() const final {
    return intgLocShift_;
  }

private:
   static constexpr int nDim_ = AlgTraits::nDim_;
   static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
   static constexpr int numIntPoints_ = AlgTraits::numScvIp_;
   
  // define ip node mappings
  const int ipNodeMap_[4] = {0, 1, 2, 3}; 

  // standard integration location
  const double intgLoc_[8] = { 
   -0.25,  -0.25, 
   +0.25,  -0.25, 
   +0.25,  +0.25, 
   -0.25,  +0.25};

  // shifted integration location
  const double intgLocShift_[8] = {
   -0.50,  -0.50, 
   +0.50,  -0.50, 
   +0.50,  +0.50, 
   -0.50,  +0.50};

  void quad_shape_fcn(
    const double *par_coord, 
    double* shape_fcn) ;
};

// 2D Quad 4 subcontrol surface
class Quad42DSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad4_2D;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  Quad42DSCS();
  KOKKOS_FUNCTION
  virtual ~Quad42DSCS() = default;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType**>& areav) override ;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) override ;

  void grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error ) override ;

  void shifted_grad_op(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void shifted_grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error ) override ;

  void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error ) override ;

  void shifted_face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  void shifted_face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error ) override ;

  void gij( 
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv) override ;

  void gij(
     const double *coords,
     double *gupperij,
     double *gij,
     double *deriv) override ;

  void Mij(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv) override ;

  void Mij(
     const double *coords,
     double *metric,
     double *deriv) override ;

  virtual const int * adjacentNodes() final;

  const int * scsIpEdgeOrd() override;

  int opposingNodes(
    const int ordinal, const int node) override;

  int opposingFace(
    const int ordinal, const int node) override;

  void shape_fcn(
    double *shpfc) override;

  void shifted_shape_fcn(
    double *shpfc) override;
  
  double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord) override;
  
  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result) override;
  
  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc) override;

  void general_face_grad_op(
    const int face_ordinal,
    const double *isoParCoord,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error ) override;

  void sidePcoords_to_elemPcoords(
    const int & side_ordinal,
    const int & npoints,
    const double *side_pcoords,
    double *elem_pcoords) override;

  const int* side_node_ordinals(int sideOrdinal) final;

  virtual const double* integration_locations() const final {
    return intgLoc_;
  }
  virtual const double* integration_location_shift() const final {
    return intgLocShift_;
  }

private :

  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScsIp_;
  static constexpr double scaleToStandardIsoFac_ = 2.0;

  // define L/R mappings
  const int lrscv_[8] = {
   0,  1, 
   1,  2, 
   2,  3, 
   0,  3};
  
  // elem-edge mapping from ip
  const int scsIpEdgeOrd_[numIntPoints_] = {0, 1, 2, 3};

  // define opposing node
  const int oppNode_[4][2] = {
    {3,  2}, // face 0; nodes 0,1
    {0,  3}, // face 1; nodes 1,2
    {1,  0}, // face 2; nodes 2,3
    {2,  1}};// face 3; nodes 3,0

  // define opposing face
  const int oppFace_[4][2] = {
    {3,  1},  // face 0
    {0,  2},  // face 1
    {1,  3},  // face 2 
    {2,  0}}; // face 3

  // standard integration location
  const double intgLoc_[8] = { 
    0.00,  -0.25, // surf 1; 1->2
    0.25,   0.00, // surf 2; 2->3
    0.00,   0.25, // surf 3; 3->4
   -0.25,   0.00};// surf 3; 1->5

  // shifted
  const double intgLocShift_[8] = {
    0.00,  -0.50,
    0.50,   0.00,
    0.00,   0.50,
   -0.50,   0.00};

  // exposed face
  const double intgExpFace_[4][2][2] = {
  {{-0.25,  -0.50}, { 0.25, -0.50}},  // face 0; scs 0, 1; nodes 0,1 
  {{ 0.50,  -0.25}, { 0.50,  0.25}},  // face 1; scs 0, 1; nodes 1,2 
  {{ 0.25,   0.50}, {-0.25,  0.50}},  // face 2, surf 0, 1; nodes 2,3
  {{-0.50,   0.25}, {-0.50, -0.25}}}; // face 3, surf 0, 1; nodes 3,0

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  const int ipNodeMap_[4][2] = { // 2 ips * 4 faces
   {0,   1},   // face 0; 
   {1,   2},   // face 1; 
   {2,   3},   // face 2; 
   {3,   0}};  // face 3; 

  const int sideNodeOrdinals_[4][2] = {
      {0, 1},
      {1, 2},
      {2, 3},
      {3, 0} 
  };

  double intgExpFaceShift_[4][2][2];

  void face_grad_op(
    const int face_ordinal,
    const bool shifted,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop);

  void quad_shape_fcn(
    const double *par_coord, 
    double* shape_fcn) ;

};

} // namespace nalu
} // namespace Sierra

#endif
