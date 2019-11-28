// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef SDRWALLFUNCALGDRIVER_H
#define SDRWALLFUNCALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

/** Wrapper class to handle computation of omega wall parameters
 *
 *. The actual kernels are templated on the face/element topology types, so we
 *. need a driver class to handle synchronization after all the topology-specific
 *. algorithms have had a chance to do their work.
 *
 *. \sa SDRWallFuncAlg, SDRLowReWallAlg
 */
class SDRWallFuncAlgDriver : public NgpAlgDriver
{
public:
  SDRWallFuncAlgDriver(Realm&);

  virtual ~SDRWallFuncAlgDriver() = default;

  virtual void pre_work() override;

  virtual void post_work() override;
};

}  // nalu
}  // sierra


#endif /* SDRWALLFUNCALGDRIVER_H */
