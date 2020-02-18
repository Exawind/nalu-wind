// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SSTMAXLENGTHSCALEDRIVER_H
#define SSTMAXLENGTHSCALEDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
namespace sierra {
namespace nalu {

class Realm;

class SSTMaxLengthScaleDriver : public NgpAlgDriver
{
public:
  SSTMaxLengthScaleDriver(Realm&);

  virtual ~SSTMaxLengthScaleDriver() = default;

  //! Reset fields before calling algorithms
  virtual void pre_work() override;

  //! Synchronize fields after algorithms have done their work
  virtual void post_work() override;
};

}  // nalu
}  // sierra


#endif /* SSTMAXLENGTHSCALEDRIVER_H */
