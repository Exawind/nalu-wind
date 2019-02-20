/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WED6CVFEM_H
#define WED6CVFEM_H

#include "master_element/MasterElement.h"

namespace sierra {
namespace nalu {

// Wedge 6 subcontrol volume
class WedSCV : public MasterElement
{
public:
  WedSCV();
  virtual ~WedSCV();

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
};

// Wedge 6 subcontrol surface
class WedSCS : public MasterElement
{
public:
  WedSCS();
  virtual ~WedSCS();

  using AlgTraits = AlgTraitsWed6;
  using MasterElement::determinant;
  using MasterElement::shape_fcn;
  using MasterElement::shifted_shape_fcn;

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

  const int * adjacentNodes();
  
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
  double parametric_distance( const std::vector<double> &x);

  const int* side_node_ordinals(int sideOrdinal) final;
  virtual const std::vector<int>& side_node_ordinals() const final {return sideNodeOrdinals_;};
  virtual void side_node_ordinals(const std::vector<int>& v) final {sideNodeOrdinals_=v;};
};


}  // nalu
}  // sierra


#endif /* WED6CVFEM_H */
