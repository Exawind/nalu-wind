#ifndef BSPLINE_H
#define BSPLINE_H

#include <vector>

namespace sierra {
namespace nalu {

// Forward declarations
class H5IO;

//====================================================================
//====================================================================

/**
 *  @class  BSpline
 *  @author James C. Sutherland
 *  @date   November, 2005
 *  @brief  Base class for B-Splines
 *
 *  \li I/O capabilities for HDF5 format are provided.
 *
 *  \li The approach here is guided heavily by
 *      "The NURBS Book" 2nd ed, Piegl & Tiller
 */
class BSpline
{
public:
  BSpline(
    const int order,
    const int dimension,
    const bool enableValueClipping = true);

  virtual ~BSpline();

  /**
   *  Copy this object and return a BSpline pointer.  This facilitates
   * polymorphic copying given a base-class pointer or reference.
   */
  virtual BSpline* clone() const = 0;

  /**
   *  Query the order of accuracy for this spline.
   *  The spline has (order-1) continuous derivatives
   */
  int get_order() const { return order_; }

  /** Query the dimensionality of this spline (number of independent variables)
   */
  int get_dimension() const { return dim_; }

  /**
   *  Evaluate the dependent variable via interpolation
   *  at the given value of the dependent variable(s).
   */
  virtual double value(const double*) const = 0;

  double value(std::vector<double>& x) const { return value(&x[0]); }

  /**
   *  Read a spline from an HDF5 database.  The file should be opened
   *  and an hdf5 "group" specified.  This spline will be read from the
   *  specified group.
   */
  virtual void read_hdf5(H5IO& io) = 0;

  /**
   *  Write a spline to an HDF5 database.  The file should be opened
   *  and an hdf5 "group" created.  This spline will be written to the
   *  specified group.
   */
  virtual void write_hdf5(H5IO& io) const = 0;

protected:
  int order_;
  const int dim_;
  const bool enableValueClipping_;

private:
  BSpline(const BSpline&);            // no copying
  BSpline& operator=(const BSpline&); // no assignment
};

//====================================================================
//====================================================================

/**
 *  @class  BSpline1D
 *  @author James C. Sutherland
 *  @date   November, 2005
 *
 *  @brief Support for B-Spline of a curve.
 *
 *  See "The NURBS Book" second edition, Chapter 9.
 */
class BSpline1D : public BSpline
{
public:
  /**
   *  Construct a 1-D bspline curve of the desired order by fitting
   *  to a set of data points.
   *
   *  @param order : Order of accuracy for interpolant, gives
   *                 (order-1 continuous derivatives)
   *  @param indepVars1 : std::vector of independent variables
   *  @param depVars    : std::vector of dependent variables
   */
  BSpline1D(
    const int order,
    const std::vector<double>& indepVars,
    const std::vector<double>& depVars,
    const bool allowClipping = true);

  /**
   *  Construct a skeletal 1D spline.
   *  Useful when loading splined data from disk.
   */
  BSpline1D(const bool allowClipping = true);

  /** Copy constructor */
  BSpline1D(const BSpline1D&);

  /** copy this object and return a BSpline pointer */
  BSpline* clone() const { return new BSpline1D(*this); }

  ~BSpline1D();

  /**
   *  Evaluate the dependent variable via interpolation
   *  at the given value of the dependent variable.
   */
  double value(const double* indepVar) const;
  inline double value(const double& x) const { return value(&x); }

  inline const std::vector<double>& get_control_pts() const
  {
    return controlPts_;
  }
  inline std::vector<double>& get_control_pts() { return controlPts_; }
  inline const std::vector<double>& get_knot_vector() const { return knots_; };

  inline int get_npts() const { return npts_; }

  inline double get_maxval() const { return maxIndepVarVal_; }
  inline double get_minval() const { return minIndepVarVal_; }

  inline const std::vector<double>& get_basisfun() const { return basisFun_; }

  void sort_inputs(
    const std::vector<double>& indepVars,
    const std::vector<double>& depVars,
    std::vector<double>& sortedIndepVars,
    std::vector<double>& sortedDepVars) const;

  /**
   *  Dump out information about this bspline.
   */
  void dump();

  void write_hdf5(H5IO& io) const;
  void read_hdf5(H5IO& io);

  inline bool operator==(const BSpline1D& a) const
  {
    return (
      a.npts_ == npts_ && a.maxIndepVarVal_ == maxIndepVarVal_ &&
      a.minIndepVarVal_ == minIndepVarVal_ && a.knots_ == knots_ &&
      a.controlPts_ == controlPts_);
  }

  inline bool operator!=(const BSpline1D& a) const { return !(*this == a); }

private:
  /**
   *  Compute the control points for this B-spline surface.
   *  This requires solution of a linear system of equations.
   */
  void compute_control_pts(
    const std::vector<double>& indepVars, const std::vector<double>& depVars);

  int npts_;
  double maxIndepVarVal_, minIndepVarVal_;
  std::vector<double> knots_, controlPts_;
  mutable std::vector<double> basisFun_;

  BSpline1D& operator=(const BSpline1D&); // no assignment
};

//====================================================================
//====================================================================

/**
 *  @class  BSpline2D
 *  @author James C. Sutherland
 *  @date November, 2005
 *
 *  2-D B-splines for STRUCTURED data
 */
class BSpline2D : public BSpline
{
public:
  /**
   *  Construct a 2-D Bspline of the requested order
   *
   *  @param order : Order of accuracy for interpolant
   *                 (order-1 continuous derivatives)
   *  @param indepVars1 : std::vector of independent variables in the first
   * dimension
   *  @param indepVars2 : std::vector of independent variables in the second
   * dimension
   *  @param depVars    : std::vector of dependent variables at each point on
   * the mesh implied by indepVars1 and indepVars2, with fortran-style ordering
   * (first dimension varies fastest)
   */
  BSpline2D(
    const int order,
    const std::vector<double>& indepVars1,
    const std::vector<double>& indepVars2,
    const std::vector<double>& depVars,
    const bool allowClipping = true);

  /**
   *  Construct a skeletal 2D spline.
   *  Useful when loading splined data from disk.
   */
  BSpline2D(const bool allowClipping = true);

  /** Copy constructor - performs deep copies */
  BSpline2D(const BSpline2D&);

  /** copy this object and return a BSpline pointer */
  BSpline* clone() const { return new BSpline2D(*this); }

  ~BSpline2D();

  /**
   *  Evaluate the dependent variable via interpolation at the
   *  given value of the dependent variable.  Ordering is [x1,x2]
   */
  double value(const double* indepVar) const;

  void write_hdf5(H5IO& io) const;
  void read_hdf5(H5IO& io);

  inline bool operator==(const BSpline2D& a) const
  {
    bool isEqual = true;
    std::vector<const BSpline1D*>::const_iterator isp = dim2Splines_.begin();
    std::vector<const BSpline1D*>::const_iterator ispa = a.dim2Splines_.begin();
    for (; isp != dim2Splines_.end(); isp++, ispa++)
      isEqual = ((**isp) == (**ispa)) ? isEqual : false;
    isEqual = (*a.sp1_ == *sp1_) ? isEqual : false;
    return isEqual;
  }

  inline bool operator!=(const BSpline2D& a) const { return !(*this == a); }

private:
  std::vector<const BSpline1D*> dim2Splines_;
  BSpline1D* sp1_;

  void compute_control_pts(
    const std::vector<double>& indepVars1,
    const std::vector<double>& indepVars2,
    const std::vector<double>& depVars);

  BSpline2D& operator=(const BSpline2D&); // no assignment
};

//====================================================================
//====================================================================

/**
 *  @class  BSpline3D
 *  @author James C. Sutherland
 *  @date December, 2005
 *
 *  3-D B-splines for STRUCTURED data
 */
class BSpline3D : public BSpline
{

public:
  /**
   *  Construct a 3-D BSpline, which is composed of a set of 2-D splines
   *  and a 1-D spline
   *
   * @param order : Order of accuracy.  Note that for higher order,
   *     interpolation times will increase.  All dimensions will be
   *     splined to this order of accuracy, although in principle each
   *     dimension could be splined to a different order...
   * @param x1 : vertices for the independent variable in the first dimension
   * @param x2 : vertices for the independent variable in the second dimension
   * @param x3 : vertices for the independent variable in the thrid dimension
   * @param phi : value of the dependent variable at all points on the mesh
   *    implied by x1,x2,x3.  By convention, phi should be arranged in memory
   *    such that it varies fastest in x1 and slowest in x3.  The size of phi
   *    should be equal to the product of sizes of x1, x2, x3.
   */
  BSpline3D(
    const int order,
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& x3,
    const std::vector<double>& phi,
    const bool allowClipping = true);

  /**
   *  Construct a skeletal 3D spline.
   *  Useful when loading splined data from disk.
   */
  BSpline3D(const bool allowClipping = true);

  /** Copy constructor - performs deep copy */
  BSpline3D(const BSpline3D&);

  /** copy this object and return a BSpline pointer */
  BSpline* clone() const { return new BSpline3D(*this); }

  ~BSpline3D();

  /**
   *  Obtain the approximation to the dependent variable given
   *  the independent variables.  Ordering is [x1,x2,x3].
   */
  double value(const double*) const;

  void write_hdf5(H5IO& io) const;
  void read_hdf5(H5IO& io);

  inline bool operator==(const BSpline3D& a) const
  {
    bool isEqual = true;
    std::vector<const BSpline2D*>::const_iterator isp = sp2d_.begin();
    std::vector<const BSpline2D*>::const_iterator ispa = a.sp2d_.begin();
    for (; isp != sp2d_.end(); isp++, ispa++)
      isEqual = (**isp == **ispa) ? isEqual : false;
    isEqual = (*a.sp1_ == *sp1_) ? isEqual : false;
    return isEqual;
  }

  inline bool operator!=(const BSpline3D& a) const { return !(*this == a); }

private:
  const int n1_, n2_, n3_; // number of control points for each dimension
  std::vector<const BSpline2D*> sp2d_;
  BSpline1D* sp1_;

  void compute_control_pts(
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& x3,
    const std::vector<double>& phi);

  BSpline3D& operator=(const BSpline3D&); // no assignment
};

//====================================================================
//====================================================================

/**
 *  @class  BSpline4D
 *  @author James C. Sutherland
 *  @date December, 2005
 *
 *  4-D B-splines for STRUCTURED data
 */
class BSpline4D : public BSpline
{

public:
  /**
   *  Construct a 4-D BSpline, which is composed of a set of 3-D splines
   *  and a 1-D spline.
   *
   * @param order : Order of accuracy.  Note that for higher order,
   *     interpolation times will increase.  All dimensions will be
   *     splined to this order of accuracy, although in principle each
   *     dimension could be splined to a different order...
   * @param x1 : vertices for the independent variable in the first dimension
   * @param x2 : vertices for the independent variable in the second dimension
   * @param x3 : vertices for the independent variable in the thrid dimension
   * @param x4 : vertices for the independent variable in the fourth dimension
   * @param phi : value of the dependent variable at all points on the mesh
   *    implied by x1,x2,x3,x4.  By convention, phi should be arranged in memory
   *    such that it varies fastest in x1 and slowest in x4.  The size of phi
   *    should be equal to the product of sizes of x1, x2, x3, x4.
   */
  BSpline4D(
    const int order,
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& x3,
    const std::vector<double>& x4,
    const std::vector<double>& phi,
    const bool allowClipping = true);

  /**
   *  Construct a skeletal 4D spline.
   *  Useful when loading splined data from disk.
   */
  BSpline4D(const bool allowClipping = true);

  /** Copy constructor - performs deep copy */
  BSpline4D(const BSpline4D&);

  /** copy this object and return a BSpline pointer */
  BSpline* clone() const { return new BSpline4D(*this); }

  ~BSpline4D();

  /**
   *  Obtain the approximation to the dependent variable given
   *  the independent variables.  Ordering is [x1,x2,x3,x4].
   */
  double value(const double* x) const;

  void write_hdf5(H5IO& io) const;
  void read_hdf5(H5IO& io);

  inline bool operator==(const BSpline4D& a) const
  {
    bool isEqual = true;
    std::vector<const BSpline3D*>::const_iterator isp = sp3d_.begin();
    std::vector<const BSpline3D*>::const_iterator ispa = a.sp3d_.begin();
    for (; isp != sp3d_.end(); isp++, ispa++)
      isEqual = (**isp == **ispa) ? isEqual : false;
    isEqual = (*a.sp1_ == *sp1_) ? isEqual : false;
    return isEqual;
  }

  inline bool operator!=(const BSpline4D& a) const { return !(*this == a); }

private:
  const int n1_, n2_, n3_, n4_;
  std::vector<const BSpline3D*> sp3d_;
  BSpline1D* sp1_;

  void compute_control_pts(
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& x3,
    const std::vector<double>& x4,
    const std::vector<double>& phi);

  BSpline4D& operator=(const BSpline4D&); // no assignment
};

//====================================================================
//====================================================================

/**
 *  @class  BSpline5D
 *  @author James C. Sutherland
 *  @date December, 2005
 *
 *  5-D B-splines for STRUCTURED data
 */
class BSpline5D : public BSpline
{

public:
  /**
   *  Construct a 5-D BSpline, which is composed of a set of 4-D splines
   *  and a 1-D spline.
   *
   * @param order : Order of accuracy.  Note that for higher order,
   *     interpolation times will increase.  All dimensions will be
   *     splined to this order of accuracy, although in principle each
   *     dimension could be splined to a different order...
   * @param x1 : vertices for the independent variable in the first dimension
   * @param x2 : vertices for the independent variable in the second dimension
   * @param x3 : vertices for the independent variable in the thrid dimension
   * @param x4 : vertices for the independent variable in the fourth dimension
   * @param x5 : vertices for the independent variable in the fourth dimension
   * @param phi : value of the dependent variable at all points on the mesh
   *    implied by x1,x2,x3,x4,x5.  By convention, phi should be arranged in
   *    memory such that it varies fastest in x1 and slowest in x4.  The size
   *    of phi should be equal to the product of sizes of x1, x2, x3, x4, x5.
   */
  BSpline5D(
    const int order,
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& x3,
    const std::vector<double>& x4,
    const std::vector<double>& x5,
    const std::vector<double>& phi,
    const bool allowClipping = true);

  /**
   *  Construct a skeletal 4D spline.
   *  Useful when loading splined data from disk.
   */
  BSpline5D(const bool allowClipping = true);

  /** Copy constructor - performs deep copy */
  BSpline5D(const BSpline5D&);

  /** copy this object and return a BSpline pointer */
  BSpline* clone() const { return new BSpline5D(*this); }

  ~BSpline5D();

  /**
   *  Obtain the approximation to the dependent variable given
   *  the independent variables.  Ordering is [x1,x2,x3,x4,x5].
   */
  double value(const double* x) const;

  void write_hdf5(H5IO& io) const;
  void read_hdf5(H5IO& io);

  inline bool operator==(const BSpline5D& a) const
  {
    bool isEqual = true;
    std::vector<const BSpline4D*>::const_iterator isp = sp4d_.begin();
    std::vector<const BSpline4D*>::const_iterator ispa = a.sp4d_.begin();
    for (; isp != sp4d_.end(); isp++, ispa++)
      isEqual = (**isp == **ispa) ? isEqual : false;
    isEqual = (*a.sp1_ == *sp1_) ? isEqual : false;
    return isEqual;
  }

  inline bool operator!=(const BSpline5D& a) const { return !(*this == a); }

private:
  const int n1_, n2_, n3_, n4_, n5_;
  std::vector<const BSpline4D*> sp4d_;
  BSpline1D* sp1_;

  void compute_control_pts(
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& x3,
    const std::vector<double>& x4,
    const std::vector<double>& x5,
    const std::vector<double>& phi);

  BSpline5D& operator=(const BSpline5D&); // no assignment
};

//====================================================================
//====================================================================

} // namespace nalu
} // namespace sierra

#endif
