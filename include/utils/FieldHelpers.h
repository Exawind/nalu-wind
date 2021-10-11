#ifndef FIELDHELPERS_H_
#define FIELDHELPERS_H_

namespace stk {
namespace mesh {
class MetaData;
}
} // namespace stk

namespace sierra {
namespace nalu {
void populate_dnv_states(
  const stk::mesh::MetaData& meta,
  unsigned nmID,
  unsigned n1ID,
  unsigned np1ID);

}
} // namespace sierra

#endif /* FIELDHELPERS_H_ */
