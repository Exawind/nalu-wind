/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Quad92DCVFEM_h  
#define Quad92DCVFEM_h  

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

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



class QuadrilateralP2Element : public MasterElement
{
public:
  using AlgTraits = AlgTraitsQuad9_2D;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  QuadrilateralP2Element();
  virtual ~QuadrilateralP2Element() {}

  void shape_fcn(double *shpfc);
  void shifted_shape_fcn(double *shpfc);
protected:
  struct ContourData {
    Jacobian::Direction direction;
    double weight;
  };  

  void set_quadrature_rule();
  void GLLGLL_quadrature_weights();

  int tensor_product_node_map(int i, int j) const;

  double gauss_point_location(
    int nodeOrdinal,
    int gaussPointOrdinal) const;

  double shifted_gauss_point_location(
    int nodeOrdinal,
    int gaussPointOrdinal) const;

  double tensor_product_weight(
    int s1Node, int s2Node,
    int s1Ip, int s2Ip) const;

  double tensor_product_weight(int s1Node, int s1Ip) const;

  double parametric_distance(const std::array<double, 2>& x); 

  virtual void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

  virtual double isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord);

  virtual void sidePcoords_to_elemPcoords(
    const int & side_ordinal,
    const int & npoints,
    const double *side_pcoords,
    double *elem_pcoords);

  void eval_shape_functions_at_ips();
  void eval_shape_functions_at_shifted_ips();

  void eval_shape_derivs_at_ips();
  void eval_shape_derivs_at_shifted_ips();

  void eval_shape_derivs_at_face_ips();

  const double scsDist_;
  const int nodes1D_;
  int numQuad_;

  //quadrature info
  std::vector<int> lrscv_;
  std::vector<double> gaussAbscissae_;
  std::vector<double> gaussAbscissaeShift_;
  std::vector<double> gaussWeight_;

  std::vector<int> stkNodeMap_;
  std::vector<double> scsEndLoc_;

  std::vector<double> shapeFunctions_;
  std::vector<double> shapeFunctionsShift_;
  std::vector<double> shapeDerivs_;
  std::vector<double> shapeDerivsShift_;
  std::vector<double> expFaceShapeDerivs_;

  const int sideNodeOrdinals_[12] =  {
      0, 1, 4,
      1, 2, 5,
      2, 3, 6,
      3, 0, 7 
  };  


private:
  void quad9_shape_fcn(
    int npts,
    const double *par_coord,
    double* shape_fcn
  ) const;

  void quad9_shape_deriv(
    int npts,
    const double *par_coord,
    double* shape_fcn
  ) const;
};

// 3D Quad 27 subcontrol volume
class Quad92DSCV : public QuadrilateralP2Element
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shifted_grad_op;

  Quad92DSCV();
  virtual ~Quad92DSCV() {}

  const int * ipNodeMap(int ordinal = 0) override ;

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

private:
  void set_interior_info();

  DoubleType jacobian_determinant(
    const SharedMemView<DoubleType**> &coords,
    const double *POINTER_RESTRICT shapeDerivs ) const;

  double jacobian_determinant(
    const double *POINTER_RESTRICT elemNodalCoords,
    const double *POINTER_RESTRICT shapeDerivs ) const;

  std::vector<double> ipWeight_;
};

// 3D Hex 27 subcontrol surface
class Quad92DSCS : public QuadrilateralP2Element
{
public:
  using MasterElement::determinant;

  Quad92DSCS();
  virtual ~Quad92DSCS() {}

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

  void gij(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& gupper,
    SharedMemView<DoubleType***>& glower,
    SharedMemView<DoubleType***>& deriv) override ;

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv) override ;

  void Mij(
    SharedMemView<DoubleType** >& coords,
    SharedMemView<DoubleType***>& metric,
    SharedMemView<DoubleType***>& deriv) override ;

  void Mij(
    const double *coords,
    double *metric,
    double *deriv) override ;

  virtual const int * adjacentNodes() final ;

  const int * ipNodeMap(int ordinal = 0) override ;

  int opposingNodes(
    const int ordinal, const int node) override ;

  int opposingFace(
    const int ordinal, const int node) override ;

  const int* side_node_ordinals(int sideOrdinal) final;

private:
  std::vector<ContourData> ipInfo_;

  void set_interior_info();
  void set_boundary_info();

  template <Jacobian::Direction direction> void
  area_vector(
    const SharedMemView<DoubleType**>& elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    DoubleType *POINTER_RESTRICT areaVector ) const;
  template <Jacobian::Direction direction> void
  area_vector(
    const double *POINTER_RESTRICT elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT areaVector ) const;

  int ipsPerFace_;
};

} // namespace nalu
} // namespace Sierra

#endif
