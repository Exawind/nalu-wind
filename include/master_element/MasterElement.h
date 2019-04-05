/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MasterElement_h
#define MasterElement_h

#include <AlgTraits.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <stdexcept>

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
  KOKKOS_FUNCTION
  MasterElement(const double scaleToStandardIsoFac=1.0);
  KOKKOS_FUNCTION
  virtual ~MasterElement() {} // = default is apparently not allowed for virtual destructors...

  // NGP-ready methods first
  KOKKOS_FUNCTION virtual void shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &/* shpfc */) {
  }

  KOKKOS_FUNCTION virtual void shifted_shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &/* shpfc */) {
  }

  KOKKOS_FUNCTION virtual void grad_op(
    SharedMemView<DoubleType**, DeviceShmem>&/* coords */,
    SharedMemView<DoubleType***, DeviceShmem>&/* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>&/* deriv */) {
  }

  KOKKOS_FUNCTION virtual void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>&/* coords */,
    SharedMemView<DoubleType***, DeviceShmem>&/* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>&/* deriv */) {
  }

  KOKKOS_FUNCTION virtual void face_grad_op(
    int /* face_ordinal */,
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */) {
  }

  KOKKOS_FUNCTION virtual void shifted_face_grad_op(
    int /* face_ordinal */,
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gradop */) {
  }

  KOKKOS_FUNCTION virtual void grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>&/* coords */,
    SharedMemView<DoubleType***, DeviceShmem>&/* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>&/* deriv */,
    SharedMemView<DoubleType*, DeviceShmem>& /*det_j*/) {
  }

  KOKKOS_FUNCTION virtual void shifted_grad_op_fem(
    SharedMemView<DoubleType**, DeviceShmem>&/* coords */,
    SharedMemView<DoubleType***, DeviceShmem>&/* gradop */,
    SharedMemView<DoubleType***, DeviceShmem>&/* deriv */,
    SharedMemView<DoubleType*, DeviceShmem>& /*det_j*/) {
  }

  KOKKOS_FUNCTION virtual void determinant(
    SharedMemView<DoubleType**, DeviceShmem>&/* coords */,
    SharedMemView<DoubleType**, DeviceShmem>&/* areav */) {
  }

  KOKKOS_FUNCTION virtual void gij(
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* gupper */,
    SharedMemView<DoubleType***, DeviceShmem>& /* glower */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */) {
  }

  KOKKOS_FUNCTION virtual void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType***, DeviceShmem>& /* metric */,
    SharedMemView<DoubleType***, DeviceShmem>& /* deriv */) {
  }

  KOKKOS_FUNCTION virtual void determinant(
    SharedMemView<DoubleType**, DeviceShmem>& /* coords */,
    SharedMemView<DoubleType*, DeviceShmem>& /* volume */) {
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

  KOKKOS_FUNCTION virtual const int *  ipNodeMap(int /* ordinal */ = 0) const {
#ifndef KOKKOS_ENABLE_CUDA
      throw std::runtime_error("ipNodeMap not implemented");
#else
      printf("Invalid ipNodeMap call on GPUs");
      return nullptr;
#endif
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

  double isoparametric_mapping(const double b, const double a, const double xi) const;
  bool within_tolerance(const double & val, const double & tol) const;
  double vector_norm_sq(const double * vect, int len) const;

  virtual int ndim()                           const {return nDim_;} 
  virtual int nodes_per_element()              const {return nodesPerElement_;} 
  KOKKOS_FUNCTION int num_integration_points() const {return numIntPoints_;}
          double scal_to_standard_iso_factor() const {return scaleToStandardIsoFac_;} 

  virtual const int   * adjacentNodes()              const {throw std::runtime_error("adjacentNodes not implimented");}
  virtual const double* integration_locations()      const {throw std::runtime_error("integration_locations not implemented");}
  virtual const double* integration_location_shift() const {throw std::runtime_error("adjacentNodes not implimented");}
  virtual const double* integration_exp_face_shift() const {throw std::runtime_error("integration_exp_face_shift not implimented");}
  virtual const int   * side_node_ordinals(int)      const {throw std::runtime_error("side_node_ordinals not implemented");}


  int nDim_;
  int nodesPerElement_;
protected:
  int numIntPoints_;
private:
  const double scaleToStandardIsoFac_;
};

} // namespace nalu
} // namespace Sierra

#endif
