/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SparsifiedLaplacian_h
#define SparsifiedLaplacian_h

#include "matrix_free/KokkosViewTypes.h"
#include "stk_mesh/base/Selector.hpp"

namespace sierra {
namespace nalu {

class TpetraLinearSystem;

void
compute_sparsified_edge_laplacian(
  int p,
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  const stk::mesh::NgpField<double>& coords,
  TpetraLinearSystem& linsys);

} // namespace nalu
} // namespace sierra

#endif
