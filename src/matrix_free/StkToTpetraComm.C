// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkToTpetraComm.h"
#include "stk_util/parallel/Parallel.hpp"
#include "Teuchos_DefaultMpiComm.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

Teuchos::RCP<const Teuchos::Comm<int>>
teuchos_communicator(const stk::ParallelMachine& pm)
{
  auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(pm));
  return Teuchos::RCP<const Teuchos::Comm<int>>(comm);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
