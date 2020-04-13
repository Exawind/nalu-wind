// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MAKE_RCP_H
#define MAKE_RCP_H

#include "Teuchos_RCP.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <typename T, typename... Args>
Teuchos::RCP<T>
make_rcp(Args&&... args)
{
  return Teuchos::RCP<T>(new T(std::forward<Args>(args)...));
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
