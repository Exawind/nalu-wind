// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPFIELDPARALLELCOMPAT_H
#define NGPFIELDPARALLELCOMPAT_H

#include "FieldTypeDef.h"
#include "KokkosInterface.h"

#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/NgpFieldParallel.hpp>

#include <vector>

namespace sierra {
namespace kynema_ugf {
namespace kynema_ugf_ngp {

inline void
parallel_sum(
  const stk::mesh::BulkData& bulk,
  const std::vector<NGPDoubleFieldType*>& ngpFields,
  const std::vector<const stk::mesh::FieldBase*>& hostFields,
  const bool syncResultToHost)
{
  if constexpr (requires {
                  stk::mesh::parallel_sum<DeviceSpace>(bulk, hostFields, true);
                }) {
    constexpr bool deterministic = false;
    stk::mesh::parallel_sum<DeviceSpace>(bulk, hostFields, deterministic);

    for (auto* field : ngpFields) {
      field->modify_on_device();
    }

    if (syncResultToHost) {
      for (auto* field : ngpFields) {
        field->sync_to_host();
      }
    }
  } else {
    const bool doFinalSyncToDevice = !syncResultToHost;
    stk::mesh::parallel_sum(bulk, ngpFields, doFinalSyncToDevice);
  }
}

} // namespace kynema_ugf_ngp
} // namespace kynema_ugf
} // namespace sierra

#endif
