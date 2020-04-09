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
