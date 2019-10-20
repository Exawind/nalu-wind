/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
