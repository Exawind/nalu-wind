/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MasterElement_h
#define MasterElement_h

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

namespace stk {
  struct topology;
}

namespace sierra{
namespace nalu{

namespace Jacobian{
enum Direction
{
  S_DIRECTION = 0,
  T_DIRECTION = 1,
  U_DIRECTION = 2
};
}

struct ElementDescription;
class MasterElement;



class MasterElement
{
public:
  MasterElement();
  virtual ~MasterElement();

  // NGP-ready methods first
  virtual void shape_fcn(
    SharedMemView<DoubleType**> &/* shpfc */) {
    throw std::runtime_error("shape_fcn using SharedMemView is not implemented");}

  virtual void shifted_shape_fcn(
    SharedMemView<DoubleType**> &/* shpfc */) {
    throw std::runtime_error("shifted_shape_fcn using SharedMemView is not implemented");}

  virtual void grad_op(
    SharedMemView<DoubleType**>&/* coords */,
    SharedMemView<DoubleType***>&/* gradop */,
    SharedMemView<DoubleType***>&/* deriv */) {
    throw std::runtime_error("grad_op using SharedMemView is not implemented");}

  virtual void shifted_grad_op(
    SharedMemView<DoubleType**>&/* coords */,
    SharedMemView<DoubleType***>&/* gradop */,
    SharedMemView<DoubleType***>&/* deriv */) {
    throw std::runtime_error("shifted_grad_op using SharedMemView is not implemented");}

  virtual void face_grad_op(
    int /* face_ordinal */,
    SharedMemView<DoubleType**>& /* coords */,
    SharedMemView<DoubleType***>& /* gradop */) {
    throw std::runtime_error("face_grad_op using SharedMemView is not implemented");}

  virtual void shifted_face_grad_op(
    int /* face_ordinal */,
    SharedMemView<DoubleType**>& /* coords */,
    SharedMemView<DoubleType***>& /* gradop */) {
    throw std::runtime_error("shifted_face_grad_op using SharedMemView is not implemented");}

  virtual void grad_op_fem(
    SharedMemView<DoubleType**>&/* coords */,
    SharedMemView<DoubleType***>&/* gradop */,
    SharedMemView<DoubleType***>&/* deriv */,
    SharedMemView<DoubleType*>& /*det_j*/) {
    throw std::runtime_error("grad_op using SharedMemView is not implemented");}

  virtual void shifted_grad_op_fem(
    SharedMemView<DoubleType**>&/* coords */,
    SharedMemView<DoubleType***>&/* gradop */,
    SharedMemView<DoubleType***>&/* deriv */,
    SharedMemView<DoubleType*>& /*det_j*/) {
    throw std::runtime_error("shifted_grad_op using SharedMemView is not implemented");}

  virtual void determinant(
    SharedMemView<DoubleType**>&/* coords */,
    SharedMemView<DoubleType**>&/* areav */) {
    throw std::runtime_error("determinant using SharedMemView is not implemented");}

  virtual void gij(
    SharedMemView<DoubleType**>& /* coords */,
    SharedMemView<DoubleType***>& /* gupper */,
    SharedMemView<DoubleType***>& /* glower */,
    SharedMemView<DoubleType***>& /* deriv */) {
    throw std::runtime_error("gij using SharedMemView is not implemented");
  }

  virtual void Mij(
    SharedMemView<DoubleType**>& /* coords */,
    SharedMemView<DoubleType***>& /* metric */,
    SharedMemView<DoubleType***>& /* deriv */) {
    throw std::runtime_error("Mij using SharedMemView is not implemented");
  }

  virtual void determinant(
    SharedMemView<DoubleType**>& /* coords */,
    SharedMemView<DoubleType*>& /* volume */) {
    throw std::runtime_error("scv determinant using SharedMemView is not implemented");
  }

  // non-NGP-ready methods second
  virtual void determinant(
    const int /* nelem */,
    const double * /* coords */,
    double * /* volume */,
    double * /* error  */) {
    throw std::runtime_error("determinant not implemented");}

  virtual void grad_op(
    const int /* nelem */,
    const double * /* coords */,
    double * /* gradop */,
    double * /* deriv */,
    double * /* det_j */,
    double * /* error  */) {
    throw std::runtime_error("grad_op not implemented");}

  virtual void shifted_grad_op(
    const int /* nelem */,
    const double * /* coords */,
    double * /* gradop */,
    double * /* deriv */,
    double * /* det_j */,
    double * /* error  */) {
    throw std::runtime_error("shifted_grad_op not implemented");}

  virtual void gij(
    const double * /* coords */,
    double * /* gupperij */,
    double * /* glowerij */,
    double * /* deriv */) {
    throw std::runtime_error("gij not implemented");}

  virtual void Mij(
    const double * /* coords */,
    double * /* metric */,
    double * /* deriv */) {
    throw std::runtime_error("Mij not implemented");}

  virtual void nodal_grad_op(
    const int /* nelem */,
    double * /* deriv */,
    double * /* error  */) {
    throw std::runtime_error("nodal_grad_op not implemented");}


  virtual void face_grad_op(
    const int /* nelem */,
    const int /* face_ordinal */,
    const double * /* coords */,
    double * /* gradop */,
    double * /* det_j */,
    double * /* error  */) {
    throw std::runtime_error("face_grad_op not implemented; avoid this element type at open bcs, walls and symms");}


  virtual void shifted_face_grad_op(
     const int /* nelem */,
     const int /* face_ordinal */,
     const double * /* coords */,
     double * /* gradop */,
     double * /* det_j */,
     double * /* error  */) {
     throw std::runtime_error("shifted_face_grad_op not implemented");}

  virtual const int * adjacentNodes() {
    throw std::runtime_error("adjacentNodes not implemented");
    }

  virtual const int * scsIpEdgeOrd() {
    throw std::runtime_error("scsIpEdgeOrd not implemented");
    }

  virtual const int * ipNodeMap(int /* ordinal */ = 0) {
      throw std::runtime_error("ipNodeMap not implemented");
     }

  virtual void shape_fcn(
    double * /* shpfc */) {
    throw std::runtime_error("shape_fcn not implemented"); }

  virtual void shifted_shape_fcn(
    double * /* shpfc */) {
    throw std::runtime_error("shifted_shape_fcn not implemented"); }

  virtual int opposingNodes(
    const int /* ordinal */, const int /* node */) {
    throw std::runtime_error("opposingNodes not implemented"); }

  virtual int opposingFace(
    const int /* ordinal */, const int /* node */) {
    throw std::runtime_error("opposingFace not implemented"); 
    }

  virtual double isInElement(
    const double * /* elemNodalCoord */,
    const double * /* pointCoord */,
    double * /* isoParCoord */) {
    throw std::runtime_error("isInElement not implemented"); 
    }

  virtual void interpolatePoint(
    const int & /* nComp */,
    const double * /* isoParCoord */,
    const double * /* field */,
    double * /* result */) {
    throw std::runtime_error("interpolatePoint not implemented"); }
  
  virtual void general_shape_fcn(
    const int /* numIp */,
    const double * /* isoParCoord */,
    double * /* shpfc */) {
    throw std::runtime_error("general_shape_fcn not implement"); }

  virtual void general_face_grad_op(
    const int /* face_ordinal */,
    const double * /* isoParCoord */,
    const double * /* coords */,
    double * /* gradop */,
    double * /* det_j */,
    double * /* error  */) {
    throw std::runtime_error("general_face_grad_op not implemented");}

  virtual void general_normal(
    const double * /* isoParCoord */,
    const double * /* coords */,
    double * /* normal */) {
    throw std::runtime_error("general_normal not implemented");}

  virtual void sidePcoords_to_elemPcoords(
    const int & /* side_ordinal */,
    const int & /* npoints */,
    const double * /* side_pcoords */,
    double * /* elem_pcoords */) {
    throw std::runtime_error("sidePcoords_to_elemPcoords");}

  virtual const int* side_node_ordinals(int /* sideOrdinal */) {
    throw std::runtime_error("side_node_ordinals not implemented");
  }

  double isoparametric_mapping(const double b, const double a, const double xi) const;
  bool within_tolerance(const double & val, const double & tol);
  double vector_norm_sq(const double * vect, int len);

  int nDim_;
  int nodesPerElement_;
  int numIntPoints_;
  double scaleToStandardIsoFac_;

  std::vector<int> lrscv_;
  std::vector<int> ipNodeMap_;
  std::vector<int> oppNode_;
  std::vector<int> oppFace_;
  std::vector<double> intgLoc_;
  std::vector<double> intgLocShift_;
  std::vector<double> intgExpFace_;
  std::vector<double> intgExpFaceShift_;
  std::vector<double> nodeLoc_;
  std::vector<int> sideNodeOrdinals_;
  std::vector<int> sideOffset_;
  std::vector<int> scsIpEdgeOrd_;

  // FEM
  std::vector<double>weights_;
};

class QuadrilateralP2Element : public MasterElement
{
public:
  using Traits = AlgTraitsQuad9_2D;
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
  std::vector<double> gaussAbscissae1D_;
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

// 2D Tri 3 subcontrol volume
class Tri2DSCV : public MasterElement
{
public:
  Tri2DSCV();
  virtual ~Tri2DSCV();

  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  void tri_shape_fcn(
    const int &npts,
    const double *par_coord,
    double* shape_fcn);
};

// 3D Tri 3
class Tri3DSCS : public MasterElement
{
public:

  Tri3DSCS();
  virtual ~Tri3DSCS();

  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void shape_fcn(
     double *shpfc);

   void shifted_shape_fcn(
     double *shpfc);

   void tri_shape_fcn(
     const int &npts,
     const double *par_coord,
     double* shape_fcn);

   double isInElement(
     const double *elemNodalCoord,
     const double *pointCoord,
     double *isoParCoord);

  double parametric_distance(
    const std::vector<double> &x);

  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc);

  void general_normal(
    const double *isoParCoord,
    const double *coords,
    double *normal);
};

// edge 2d
class Edge2DSCS : public MasterElement
{
public:
  Edge2DSCS();
  virtual ~Edge2DSCS();
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

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

  void general_shape_fcn(
    const int numIp,
    const double *isoParCoord,
    double *shpfc);

  void general_normal(
    const double *isoParCoord,
    const double *coords,
    double *normal);

  double parametric_distance(const std::vector<double> &x);

  const double elemThickness_;  
};

// edge 2d
class Edge32DSCS : public QuadrilateralP2Element
{
public:
  Edge32DSCS();
  virtual ~Edge32DSCS() {}
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);

  void interpolatePoint(
    const int &nComp,
    const double *isoParCoord,
    const double *field,
    double *result);

private:
  void area_vector(
    const double *POINTER_RESTRICT coords,
    const double s,
    double *POINTER_RESTRICT areaVector) const;

  std::vector<double> ipWeight_;
};

} // namespace nalu
} // namespace Sierra

#endif
