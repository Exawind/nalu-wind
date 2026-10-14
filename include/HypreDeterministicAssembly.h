// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#ifndef HYPREDETERMINISTICASSEMBLY_H
#define HYPREDETERMINISTICASSEMBLY_H

#include <HYPRE_utilities.h>
#include <mpi.h>
#include <cstddef>
#include <vector>

namespace sierra {
namespace kynema_ugf {

class HypreDeterministicAssembly
{
public:
  struct OwnedEntries
  {
    std::vector<HYPRE_Int> rows;
    std::vector<HYPRE_Int> cols;
    std::vector<double> values;
  };

  HypreDeterministicAssembly(MPI_Comm comm, HYPRE_Int lower, HYPRE_Int upper);
  ~HypreDeterministicAssembly();
  HypreDeterministicAssembly(const HypreDeterministicAssembly&) = delete;
  HypreDeterministicAssembly&
  operator=(const HypreDeterministicAssembly&) = delete;

  // Input is owned entries followed by shared entries. A null columns pointer
  // denotes a vector. Reduction never modifies the pre-assembly buffers.
  OwnedEntries reduce(
    std::size_t numOwned,
    std::size_t numShared,
    const HYPRE_Int* rows,
    const HYPRE_Int* cols,
    const double* values) const;

private:
  struct Entry
  {
    HYPRE_Int row;
    HYPRE_Int col;
    double value;
  };

  int owner(HYPRE_Int row) const;

  MPI_Comm comm_{MPI_COMM_NULL};
  MPI_Datatype entryType_{MPI_DATATYPE_NULL};
  std::vector<HYPRE_Int> lower_;
  std::vector<HYPRE_Int> upper_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif
