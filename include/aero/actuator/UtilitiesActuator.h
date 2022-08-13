// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#ifndef UTILITIESACTUATOR_H_
#define UTILITIESACTUATOR_H_

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_search/Point.hpp>
#ifdef NALU_USES_OPENFAST
#include <OpenFAST.H>
#endif

namespace sierra {
namespace nalu {

struct Coordinates;
using Point = stk::search::Point<double>;

namespace actuator_utils {

#ifdef NALU_USES_OPENFAST

Point get_fast_point(
  fast::OpenFAST& fast,
  int turbId,
  fast::ActuatorNodeType type,
  int pointId = 0,
  int bladeId = 0);

int get_fast_point_index(
  const fast::fastInputs& fi,
  int turbId,
  int nBlades,
  fast::ActuatorNodeType type,
  int pointId = 0,
  int bladeId = 0);

#endif

/** Implementation of a periodic Bezier curve (Sanchez-Reyes, 2009) to connect
 * points at a specific radius The advantage of this method is it maps distorted
 * points to an elipsoide with fewer samples than pure B-Splines or Bezier
 * curves. Fewer points are needed to create a perfect circle in the case of
 * equispaced points (min =3) It is a parametric curve over the interval [0,
 * 2pi]
 */
class SweptPointLocator
{
public:
  SweptPointLocator();
  ~SweptPointLocator() = default;
  Point operator()(double t);
  void update_point_location(int i, Point p);
  static int binomial_coefficient(int n, int v);
  std::vector<Point> get_control_points();
  double get_radius(int pntNum);
  Point get_centriod();

private:
  const int order_ = 2; // fix order at 2 for 3 point sampling
  const double delta_ = 2.0 * M_PI / (order_ + 1);
  double periodic_basis(double t);
  void generate_control_points();
  std::vector<Point> bladePoints_;
  std::vector<Point> controlPoints_;
  bool controlPointsCurrent_;
};

template <typename T>
inline void
reduce_view_on_host(T view)
{
  ThrowAssert(view.size() > 0);
  ThrowAssert(view.data());
  MPI_Datatype mpi_type;
  if (std::is_same<typename T::value_type, double>::value) {
    mpi_type = MPI_DOUBLE;
  } else if (std::is_same<typename T::value_type, int>::value) {
    mpi_type = MPI_INT;
  } else if (std::is_same<typename T::value_type, bool>::value) {
    mpi_type = MPI_C_BOOL;
  } else if (std::is_same<typename T::value_type, uint64_t>::value) {
    mpi_type = MPI_LONG;
  } else {
    ThrowErrorMsg("unsupported type to reduce view on host");
  }

  MPI_Allreduce(
    MPI_IN_PLACE, view.data(), view.size(), mpi_type, MPI_SUM,
    NaluEnv::self().parallel_comm());
}

// A Gaussian projection function
double Gaussian_projection(int nDim, double* dis, const Coordinates& epsilon);

// A Gaussian projection function
double Gaussian_projection(int nDim, double* dis, double* epsilon);

void resize_std_vector(
  const int& sizeOfField,
  std::vector<double>& theVector,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData);

void gather_field(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement);

void gather_field_for_interp(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement);

void interpolate_field(
  const int& sizeOfField,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData,
  const double* isoParCoords,
  const double* fieldAtNodes,
  double* pointField);

void compute_distance(
  int nDim,
  const double* elemCentroid,
  const double* pointCentroid,
  double* distance);
} // namespace actuator_utils
} // namespace nalu
} // namespace sierra

#endif
