/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef MasterElementHO_h
#define MasterElementHO_h

#include <master_element/MasterElement.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/HexNElementDescription.h>
#include <element_promotion/QuadNElementDescription.h>

#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <vector>
#include <array>

namespace sierra{
namespace nalu{

  struct ContourData {
    Jacobian::Direction direction;
    double weight;
  };

struct ElementDescription;
struct HexNElementDescription;

class LagrangeBasis;
class TensorProductQuadratureRule;

class HigherOrderHexSCV final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  HigherOrderHexSCV(
    ElementDescription elem,
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderHexSCV() {}

  void shape_fcn(double *shpfc) final;
  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error ) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  std::vector<double> shape_functions() {
    return shapeFunctionVals_;
  }

  std::vector<double> shape_function_derivatives() {
    return shapeDerivs_;
  }

  std::vector<double> ip_weights() {
    return ipWeights_;
  }


  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();

  double jacobian_determinant(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDerivs ) const;

  const ElementDescription elem_;
  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  std::vector<double> shapeFunctionVals_;
  std::vector<double> shapeDerivs_;
  std::vector<double> ipWeights_;
  std::vector<double> intgLoc_;
  std::vector<int> ipNodeMap_;
};

// 3D Hex 27 subcontrol surface
class HigherOrderHexSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::grad_op;
  using MasterElement::shape_fcn;
  using MasterElement::gij;
  using MasterElement::face_grad_op;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  HigherOrderHexSCS(
    ElementDescription elem,
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderHexSCS() {}

  void shape_fcn(double *shpfc) final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error) final;

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv) final;

  double isInElement(
      const double *elemNodalCoord,
      const double *pointCoord,
      double *isoParCoord) final;

  void interpolatePoint(
      const int &nComp,
      const double *isoParCoord,
      const double *field,
      double *result) final;

  const int * adjacentNodes() final;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  const int * side_node_ordinals(int ordinal = 0) const final;

  int opposingNodes(
    const int ordinal, const int node) final;

  int opposingFace(
    const int ordinal, const int node) final;

  std::vector<double> shape_functions() {
    return shapeFunctionVals_;
  }

  std::vector<double> shape_function_derivatives() {
    return shapeDerivs_;
  }

  void face_grad_op(
    int face_ordinal,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop) final;

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  std::vector<int> lrscv_;
  std::vector<int> oppNode_;

  void set_interior_info();
  void set_boundary_info();

  template <Jacobian::Direction direction> void
  area_vector(
    const double *POINTER_RESTRICT elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT areaVector) const;

  const ElementDescription elem_;
  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  std::vector<int> sideNodeOrdinals_;
  std::vector<double> shapeFunctionVals_;
  std::vector<double> shapeDerivs_;
  std::vector<double> expFaceShapeDerivs_;
  std::vector<double> intgLoc_;
  std::vector<double> intgExpFace_;
  std::vector<ContourData> ipInfo_;
  std::vector<int> ipNodeMap_;
  std::vector<int> oppFace_;
  int ipsPerFace_;

  AlignedViewType<DoubleType**[3]> expRefGradWeights_;
};

// 3D Quad 9
class HigherOrderQuad3DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  HigherOrderQuad3DSCS(
    ElementDescription elem,
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);

  KOKKOS_FUNCTION
  virtual ~HigherOrderQuad3DSCS() {}

  void shape_fcn(double *shpfc) final;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  std::vector<double> shape_functions() {
    return shapeFunctionVals_;
  }

  std::vector<double> shape_function_derivatives() {
    return shapeDerivs_;
  }

  std::vector<double> ip_weights() {
    return ipWeights_;
  }

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();
  void eval_shape_functions_at_ips();
  void eval_shape_derivs_at_ips();

  void area_vector(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    std::array<double,3>& areaVector) const;

  const ElementDescription elem_;
  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  std::vector<double> shapeFunctionVals_;
  std::vector<double> shapeDerivs_;
  std::vector<double> ipWeights_;
  std::vector<double> intgLoc_;
  std::vector<int> ipNodeMap_;
  int surfaceDimension_;
};

class HigherOrderQuad2DSCV final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::grad_op;

  KOKKOS_FUNCTION
  HigherOrderQuad2DSCV(
    ElementDescription elem,
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderQuad2DSCV() {}

  void shape_fcn(double *shpfc) final;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error ) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  std::vector<double> shape_functions() {
    return shapeFunctionVals_;
  }

  std::vector<double> shape_function_derivatives() {
    return shapeDerivs_;
  }

  std::vector<double> ip_weights() {
    return ipWeights_;
  }

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void set_interior_info();

  double jacobian_determinant(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDerivs ) const;

  const ElementDescription elem_;
  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  std::vector<double> shapeFunctionVals_;
  std::vector<double> shapeDerivs_;
  std::vector<double> ipWeights_;
  std::vector<double> intgLoc_;
  std::vector<int> ipNodeMap_;
};
class HigherOrderQuad2DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::grad_op;
  using MasterElement::face_grad_op;
  using MasterElement::gij;
  using MasterElement::adjacentNodes;

  KOKKOS_FUNCTION
  HigherOrderQuad2DSCS(
    ElementDescription elem,
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderQuad2DSCS() {}

  void shape_fcn(double *shpfc) final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error) final;

  void gij(
    const double *coords,
    double *gupperij,
    double *glowerij,
    double *deriv) final;

  double isInElement(
      const double *elemNodalCoord,
      const double *pointCoord,
      double *isoParCoord) final;

  void interpolatePoint(
      const int &nComp,
      const double *isoParCoord,
      const double *field,
      double *result) final;

  const int * adjacentNodes() final;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  int opposingNodes(
    const int ordinal, const int node) final;

  int opposingFace(
    const int ordinal, const int node) final;

  const int * side_node_ordinals(int ordinal = 0) const final;
  virtual const std::vector<int>& side_node_ordinals() const final {return sideNodeOrdinals_;};
  virtual void side_node_ordinals(const std::vector<int>& v) final {sideNodeOrdinals_=v;};

  std::vector<double> shape_functions() {
    return shapeFunctionVals_;
  }

  std::vector<double> shape_function_derivatives() {
    return shapeDerivs_;
  }

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  std::vector<int> lrscv_;

  void set_interior_info();
  void set_boundary_info();

  template <Jacobian::Direction direction> void
  area_vector(
    const double *POINTER_RESTRICT elemNodalCoords,
    double *POINTER_RESTRICT shapeDeriv,
    double *POINTER_RESTRICT normalVec ) const;

  const ElementDescription elem_;
  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  std::vector<int> sideNodeOrdinals_;
  std::vector<double> shapeFunctionVals_;
  std::vector<double> shapeDerivs_;
  std::vector<double> intgLoc_;
  std::vector<ContourData> ipInfo_;
  std::vector<int> ipNodeMap_;
  int ipsPerFace_;
  std::vector<double> expFaceShapeDerivs_;
  std::vector<int> oppNode_;
  std::vector<int> oppFace_;
  std::vector<double> intgExpFace_;
};

class HigherOrderEdge2DSCS final: public MasterElement
{
public:
  using MasterElement::determinant;
  using MasterElement::shape_fcn;

  KOKKOS_FUNCTION
  explicit HigherOrderEdge2DSCS(
    ElementDescription elem,
    LagrangeBasis basis,
    TensorProductQuadratureRule quadrature);
  KOKKOS_FUNCTION
  virtual ~HigherOrderEdge2DSCS() = default;

  virtual const int * ipNodeMap(int ordinal = 0) const final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) final;

  void shape_fcn(
    double *shpfc) final;

  std::vector<double> shape_functions() {
    return shapeFunctionVals_;
  }

  std::vector<double> shape_function_derivatives() {
    return shapeDerivs_;
  }

  std::vector<double> ip_weights() {
    return ipWeights_;
  }

  virtual const double* integration_locations() const final {
    return intgLoc_.data();
  }

private:
  void area_vector(
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    std::array<double,2>& areaVector) const;

  const ElementDescription elem_;
  LagrangeBasis basis_;
  const TensorProductQuadratureRule quadrature_;

  std::vector<double> shapeFunctionVals_;
  std::vector<double> shapeDerivs_;
  std::vector<double> ipWeights_;
  std::vector<double> intgLoc_;
  std::vector<int> ipNodeMap_;
};

} // namespace nalu
} // namespace Sierra

#endif
