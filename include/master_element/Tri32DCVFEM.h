/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Tri32DCVFEM_h   
#define Tri32DCVFEM_h   

#include <master_element/MasterElement.h>

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>

namespace sierra{
namespace nalu{

// 2D Tri 3 subcontrol volume
class Tri32DSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsTri3_2D;
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  KOKKOS_FUNCTION
  Tri32DSCV();
  KOKKOS_FUNCTION
  virtual ~Tri32DSCV() = default;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    SharedMemView<DoubleType**> &coords,
    SharedMemView<DoubleType*> &vol) override ;

  void grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void shifted_grad_op(
    SharedMemView<DoubleType**>&coords,
    SharedMemView<DoubleType***>&gradop,
    SharedMemView<DoubleType***>&deriv) override ;

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv) override;

  void Mij(
    const double *coords,
    double *metric,
    double *deriv) override;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) override;

  void shape_fcn(
    double *shpfc) override;

  void shifted_shape_fcn (
    double *shpfc) override;

  void tri_shape_fcn(
    const int npts,
    const double *par_coord,
    double* shape_fcn);

private :
  const int nDim_ = AlgTraits::nDim_;
  const int nodesPerElement_ = AlgTraits::nodesPerElement_; 
  const int numIntPoints_ = AlgTraits::numScvIp_; 

  // define ip node mappings
  const int ipNodeMap_[3] = {0, 1, 2}; 

  const double intgLoc_[6] = {
      5.0/24.0, 5.0/24.0,
      7.0/12.0, 5.0/24.0,
      5.0/24.0, 7.0/12.0
  };

  const double intgLocShift_[6] = {
      0.0,  0.0, 
      1.0,  0.0, 
      0.0,  1.0  
  };

};

// 2D Tri 3 subcontrol surface
class Tri32DSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsTri3_2D;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  Tri32DSCS();
  KOKKOS_FUNCTION
  virtual ~Tri32DSCS() = default;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType**>& areav) override ;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) override;

  void grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error ) override;

  void shifted_grad_op(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop,
    SharedMemView<DoubleType***>& deriv) override ;

  void shifted_grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error ) override;

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
    double * error ) override;

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
    double * error ) override;

  void gij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv) override ;

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv) override;

  void Mij(
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv) override;

  void Mij(
    const double *coords,
    double *metric,
    double *deriv) override;

  const int * adjacentNodes() final;

  const int * scsIpEdgeOrd() override;

  void shape_fcn(
    double *shpfc) override;

  void shifted_shape_fcn(
    double *shpfc) override;
  
  void tri_shape_fcn(
    const int npts,
    const double *par_coord, 
    double* shape_fcn);

  void
  general_shape_fcn(const int numIp, const double* isoParCoord, double* shpfc) override
  {
    tri_shape_fcn(numIp, isoParCoord, shpfc);
  }

  int opposingNodes(
    const int ordinal, const int node) override;
  
  int opposingFace(
    const int ordinal, const int node) override;

  double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord) override;
  
  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result) override;

  double tri_parametric_distance(
    const std::array<double,2> &x);
  
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

private:
  const int nDim_ = AlgTraits::nDim_;
  const int nodesPerElement_ = AlgTraits::nodesPerElement_; 
  const int numIntPoints_ = AlgTraits::numScsIp_; 

  const int sideNodeOrdinals_[3][2] = {
     {0, 1},  // ordinal 0
     {1, 2},  // ordinal 1
     {2, 0}   // ordinal 2
  };

  // define L/R mappings
  const int lrscv_[6] = {
   0,  1, 
   1,  2, 
   0,  2};

  // elem-edge mapping from ip
  const int scsIpEdgeOrd_[AlgTraits::numScsIp_] = {0, 1, 2}; 

  // define opposing node
  const int oppNode_[3][2] = 
  {{2,  2},  // face 0; nodes 0,1
   {0,  0},  // face 1; nodes 1,2
   {1,  1}}; // face 2; nodes 2,0

  // define opposing face
  const int oppFace_[3][2] = 
  {{2,  1},    // face 0 
   {0,  2},    // face 1 
   {1,  0}};   // face 2 

  // standard integration location
  const double five12ths = 5.0/12.0;
  const double one6th = 1.0/6.0;
  const double intgLoc_[6] =  
  {five12ths,  one6th,     // surf 1: 0->1
   five12ths,  five12ths,  // surf 2: 1->3
   one6th,     five12ths}; // surf 3: 0->2
 
  // shifted
  const double intgLocShift_[6] =
  {0.50,  0.00,  // surf 1: 0->1
   0.50,  0.50,  // surf 1: 1->3
   0.00,  0.50}; // surf 1: 0->2
 
  // exposed face
  const double intgExpFace_[3][2][2] =   
  // face 0: scs 0, 1: nodes 0,1
  {{{0.25,  0.00},
    {0.75,  0.00}},
  // face 1: scs 0, 1: nodes 1,2
   {{0.75,  0.25},
    {0.25,  0.75}},
  // face 2: surf 0, 1: nodes 2,0
   {{0.00,  0.75},
    {0.00,  0.25}}};

  // boundary integration point ip node mapping (ip on an ordinal to local node number)
  const int ipNodeMap_[3][2]= // 2 ips * 3 faces
    {{0,   1},   // face 0
     {1,   2},   // face 1
     {2,   0}};  // face 2

  double intgExpFaceShift_[3][2][2];

};

} // namespace nalu
} // namespace Sierra

#endif
