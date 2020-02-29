// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include <cmath>

#include "actuator/UtilitiesActuator.h"// master elements


// This is to access sierra::nalu::Coordinates
#include "NaluParsing.h"


namespace sierra{
namespace nalu {
namespace actuator_utils {

const double pi = M_PI;
///
/// A Gaussian projection function
///
double Gaussian_projection(
  int nDim,  // The dimension of the Gaussian (2 or 3)
  double *dis,      // The distance from the center of the Gaussian
  const Coordinates &epsilon  // The width of the Gaussian
  )
{
  // Compute the force projection weight at this location using a
  // Gaussian function.
  double g;
  if ( nDim == 2 )
    g = (1.0 / (epsilon.x_ * epsilon.y_ * pi)) *
        exp(-pow((dis[0]/epsilon.x_),2.0)
            -pow((dis[1]/epsilon.y_),2.0)
           );
  else
    g = (1.0 / (epsilon.x_ * epsilon.y_ * epsilon.z_ * std::pow(pi,1.5))) *
        exp(-pow((dis[0]/epsilon.x_),2.0)
            -pow((dis[1]/epsilon.y_),2.0)
            -pow((dis[2]/epsilon.z_),2.0)
           );

  return g;
}
///
/// A Gaussian projection function
///
double Gaussian_projection(
  int nDim,
  double *dis,
  double *epsilon)
{
  // Compute the force projection weight at this location using a
  // Gaussian function.
  double g;
  if ( nDim == 2 )
    g = (1.0 / (epsilon[0] * epsilon[1] * pi)) *
        exp(-pow((dis[0]/epsilon[0]),2.0)
            -pow((dis[1]/epsilon[1]),2.0)
           );
  else
    g = (1.0 / (epsilon[0] * epsilon[1] * epsilon[2] * std::pow(pi,1.5))) *
        exp(-pow((dis[0]/epsilon[0]),2.0)
            -pow((dis[1]/epsilon[1]),2.0)
            -pow((dis[2]/epsilon[2]),2.0)
           );

  return g;
}

void resize_std_vector(
  const int& sizeOfField,
  std::vector<double>& theVector,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData)
{
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);
  const int nodesPerElement = meSCS->nodesPerElement_;
  theVector.resize(nodesPerElement * sizeOfField);
}

//--------------------------------------------------------------------------
//-------- gather_field ----------------------------------------------------
//--------------------------------------------------------------------------
void gather_field(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement)
{
  for (int ni = 0; ni < nodesPerElement; ++ni) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double* theField = (double*)stk::mesh::field_data(stkField, node);
    for (int j = 0; j < sizeOfField; ++j) {
      const int offSet = ni * sizeOfField + j;
      fieldToFill[offSet] = theField[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- gather_field_for_interp -----------------------------------------
//--------------------------------------------------------------------------
void gather_field_for_interp(
  const int& sizeOfField,
  double* fieldToFill,
  const stk::mesh::FieldBase& stkField,
  stk::mesh::Entity const* elem_node_rels,
  const int& nodesPerElement)
{
  for (int ni = 0; ni < nodesPerElement; ++ni) {
    stk::mesh::Entity node = elem_node_rels[ni];
    const double* theField = (double*)stk::mesh::field_data(stkField, node);
    for (int j = 0; j < sizeOfField; ++j) {
      const int offSet = j * nodesPerElement + ni;
      fieldToFill[offSet] = theField[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- interpolate_field -----------------------------------------------
//--------------------------------------------------------------------------
void interpolate_field(
  const int& sizeOfField,
  stk::mesh::Entity elem,
  const stk::mesh::BulkData& bulkData,
  const double* isoParCoords,
  const double* fieldAtNodes,
  double* pointField)
{
  // extract master element from the bucket in which the element resides
  const stk::topology& elemTopo = bulkData.bucket(elem).topology();
  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);

  // interpolate velocity to this best point
  meSCS->interpolatePoint(sizeOfField, isoParCoords, fieldAtNodes, pointField);
}

void
compute_distance(
  int nDim,
  const double *elemCentroid,
  const double *pointCentroid,
  double *distance)
{
  for ( int j = 0; j < nDim; ++j )
    distance[j] = elemCentroid[j] - pointCentroid[j];
  //~ return distance;
}

}
}
}
