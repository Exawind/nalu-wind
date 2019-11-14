/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

namespace sierra {
namespace nalu {

struct Coordinates;

namespace actuator_utils {

// A Gaussian projection function
double Gaussian_projection(
  const int &nDim,
  double *dis,
  const Coordinates &epsilon);

// A Gaussian projection function
double Gaussian_projection(
  const int &nDim,
  double *dis,
  double *epsilon);

}  // namespace actuator_utils
}  // namespace actuator_utils
}  // namespace actuator_utils
