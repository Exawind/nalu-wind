#include <utils/FieldHelpers.h>
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"
#include <stdexcept>

namespace sierra {
namespace nalu {
void
populate_dnv_states(
  const stk::mesh::MetaData& meta, unsigned nm1ID, unsigned nID, unsigned np1ID)
{
  np1ID = get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateNP1);
  const auto* dnv = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  switch (dnv->number_of_states()) {
  case 1:
    nID = np1ID;
    nm1ID = np1ID;
    break;
  case 2:
    nID = get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateN);
    nm1ID = np1ID;
    break;
  case 3:
    nID = get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateN);
    nm1ID = get_field_ordinal(meta, "dual_nodal_volume", stk::mesh::StateNM1);
    break;
  default:
    throw std::runtime_error(
      "Number of states for dual_nodal_volume is not 1,2,3 and is undefined");
  }
}

} // namespace nalu
} // namespace sierra