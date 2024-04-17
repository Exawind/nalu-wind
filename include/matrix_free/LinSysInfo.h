// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LINSYS_INFO_H
#define LINSYS_INFO_H

#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/GetNgpField.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

using lid_type = typename Tpetra::Map<>::local_ordinal_type;
using gid_type = typename Tpetra::Map<>::global_ordinal_type;

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using const_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const;
using ra_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

struct linsys_info
{
  static constexpr auto gid_name = "tpet_global_id";

  static stk::mesh::NgpField<gid_type>
  get_gid_field(const stk::mesh::MetaData& meta)
  {
    STK_ThrowRequire(meta.get_field(stk::topology::NODE_RANK, gid_name));
    return stk::mesh::get_updated_ngp_field<gid_type>(
      *meta.get_field(stk::topology::NODE_RANK, gid_name));
  }
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
