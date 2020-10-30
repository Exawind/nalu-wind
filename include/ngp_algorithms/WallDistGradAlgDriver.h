// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef WALLDISTGRADALGDRIVER_H
#define WALLDISTGRADALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

template<typename GradPhiType>
class WallDistGradAlgDriver : public NgpAlgDriver
{
  static_assert(
    std::is_same<GradPhiType, VectorFieldType>::value ||
    std::is_same<GradPhiType, GenericFieldType>::value,
    "Invalid field type provided to WallDistGradAlgDriver");

public:
  WallDistGradAlgDriver(Realm&, const std::string&);

  virtual ~WallDistGradAlgDriver() = default;

  //! Reset fields before calling algorithms
  virtual void pre_work() override;

  //! Synchronize fields after algorithms have done their work
  virtual void post_work() override;

private:
  //! Field that is synchronized pre/post updates
  const std::string gradPhiName_;
};

using ScalarWallDistGradAlgDriver = WallDistGradAlgDriver<VectorFieldType>;
using VectorWallDistGradAlgDriver = WallDistGradAlgDriver<GenericFieldType>;

}  // nalu
}  // sierra


#endif /* WALLDISTGRADALGDRIVER_H */
