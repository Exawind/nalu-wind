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

  const int * adjacentNodes();

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
  virtual const std::vector<int>& side_node_ordinals() const final {return sideNodeOrdinals_;};
  virtual void side_node_ordinals(const std::vector<int>& v) final {sideNodeOrdinals_=v;};
};

} // namespace nalu
} // namespace Sierra

#endif
