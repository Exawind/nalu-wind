// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#pragma once

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra::kynema_ugf {

class FluxDivAlgDriver : public NgpAlgDriver
{
public:
  FluxDivAlgDriver(Realm&, const std::string&);
  void pre_work() final;
  void post_work() final;

private:
  const std::string div_flux_name_;
};

} // namespace sierra::kynema_ugf
