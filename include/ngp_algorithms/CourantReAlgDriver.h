// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COURANTREALGDRIVER_H
#define COURANTREALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"

namespace sierra {
namespace nalu {

class CourantReAlgDriver : public NgpAlgDriver
{
public:
  CourantReAlgDriver(Realm&);

  virtual ~CourantReAlgDriver() = default;

  void pre_work() override;

  void post_work() override;

  void update_max_cfl_rey(const double cfl, const double rey);

private:
  double maxCFL_;
  double maxRe_;
};

}  // nalu
}  // sierra



#endif /* COURANTREALGDRIVER_H */
