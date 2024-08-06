// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NODALBUOYANCYALGDRIVER_H
#define NODALBUOYANCYALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class NodalBuoyancyAlgDriver : public NgpAlgDriver
{
public:
  NodalBuoyancyAlgDriver(Realm&, const std::string&, const std::string&);

  virtual ~NodalBuoyancyAlgDriver() = default;

  //! Reset fields before calling algorithms
  virtual void pre_work() override;

  //! Synchronize fields after algorithms have done their work
  virtual void post_work() override;

private:
  //! Field that is synchronized pre/post updates
  const std::string sourceName_;
  const std::string sourceweightName_;
};

} // namespace nalu
} // namespace sierra

#endif /* NODALBUOYANCYALGDRIVER_H */
