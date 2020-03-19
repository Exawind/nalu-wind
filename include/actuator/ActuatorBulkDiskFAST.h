// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef INCLUDE_ACTUATOR_ACTUATORBULKDISKFAST_H_
#define INCLUDE_ACTUATOR_ACTUATORBULKDISKFAST_H_

#include <actuator/ActuatorBulkFAST.h>

namespace sierra
{
namespace nalu
{

struct ActuatorBulkDiskFAST: public ActuatorBulkFAST
{
public:
  ActuatorBulkDiskFAST(ActuatorMetaFAST& actMeta, double naluTimeStep);

  ActFixArrayInt numSweptCount_;
  ActFixArrayInt numSweptOffset_;
  void spread_forces_over_disk(const ActuatorMetaFAST& actMeta);
  bool searchExecuted_;

private:
  void compute_swept_point_count(ActuatorMetaFAST& actMeta);
  void resize_arrays(const ActuatorMetaFAST& actMeta);
  void initialize_swept_points(const ActuatorMetaFAST& actMeta);

};

} /* namespace nalu */
} /* namespace sierra */

#endif /* INCLUDE_ACTUATOR_ACTUATORBULKDISKFAST_H_ */
