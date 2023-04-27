// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPFIELDMANAGER_H
#define NGPFIELDMANAGER_H

/** \file
 *  \brief NGP Field Manager
 */

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

namespace sierra {
namespace nalu {
namespace nalu_ngp {

/** NGP Field Manager
 *
 *  This lightweight class wraps the new NgpField interface using the
 *  deprecated FieldManager workflow.
 */
class FieldManager
{
public:
  FieldManager(const stk::mesh::BulkData& bulk) : m_meta(bulk.mesh_meta_data())
  {
  }

  ~FieldManager() {}

  FieldManager& operator=(const FieldManager& rhs) = delete;

  FieldManager(const FieldManager& rhs) = delete;
  FieldManager(FieldManager&& rhs) = delete;

  unsigned size() const { return m_meta.get_fields().size(); }

  template <typename T>
  stk::mesh::NgpField<T>& get_field(unsigned fieldOrdinal) const
  {
    ThrowAssertMsg(
      m_meta.get_fields().size() > fieldOrdinal, "Invalid field ordinal.");
    stk::mesh::FieldBase* stkField = m_meta.get_fields()[fieldOrdinal];
    stk::mesh::NgpField<T>& tmp =
      stk::mesh::get_updated_ngp_field<T>(*stkField);
    return tmp;
  }

private:
  const stk::mesh::MetaData& m_meta;
};

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* NGPFIELDMANAGER_H */
