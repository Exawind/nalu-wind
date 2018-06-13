/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Hex8CVFEM_h
#define Hex8CVFEM_h

#include<master_element/MasterElement.h>

namespace sierra{
namespace nalu{

// Hex 8 subcontrol volume
class HexSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex8;

  HexSCV();
  virtual ~HexSCV();

  const int * ipNodeMap(int ordinal = 0);

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

  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

  void shape_fcn(
    double *shpfc);

  void shifted_shape_fcn(
    double *shpfc);
};

// Hex 8 subcontrol surface
class HexSCS : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex8;
  using AlgTraitsFace = AlgTraitsQuad4;


  HexSCS();
  virtual ~HexSCS();

  const int * ipNodeMap(int ordinal = 0);

  // NGP-ready methods first
  void shape_fcn(
    SharedMemView<DoubleType**> &shpfc);

  void shifted_shape_fcn(
    SharedMemView<DoubleType**> &shpfc);

  void hex8_shape_fcn(
    const int  &numIp,
    const double *isoParCoord,
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

  const int * adjacentNodes();

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

  double parametric_distance(const std::vector<double> &x);

private :

  void face_grad_op(
    const int face_ordinal,
    const bool shifted,
    SharedMemView<DoubleType**>& coords,
    SharedMemView<DoubleType***>& gradop);
};
    
} // namespace nalu
} // namespace Sierra

#endif
