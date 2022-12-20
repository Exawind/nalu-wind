// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SOLIDBODY_H_
#define SOLIDBODY_H_

namespace sierra::nalu {
/** A parent class for implementing a solid body for aero dynamics applications
 * This class should become the owner of surface post processing algorithms
 * and instances will live inside the AeroContainer class
 *
 */
class SolidBody
{
  // virtual class or template? do we think we'd want this to live on device
  virtual void register_nodal_fields() = 0;
  // std::unique_ptr<SurfaceFMPostProcessing> surfacePostProcessor;
};

} // namespace sierra::nalu

#endif
