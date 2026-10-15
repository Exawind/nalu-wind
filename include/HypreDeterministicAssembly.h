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

  // Collective setup for a fixed graph. A null columns pointer denotes a
  // vector. Repeating setup replaces the plan, including its persistent
  // requests.
  void setup(
    std::size_t numOwned,
    std::size_t numShared,
    const HYPRE_Int* rows,
    const HYPRE_Int* cols,
    unsigned components = 1);

  bool isSetup() const { return isSetup_; }
  const OwnedEntries& entries() const { return entries_; }
  const std::vector<std::size_t>& localOffsets() const { return localOffsets_; }
  const std::vector<std::size_t>& localIndices() const { return localIndices_; }
  const std::vector<std::size_t>& receiveOffsets() const
  {
    return receiveOffsets_;
  }
  const std::vector<std::size_t>& receiveIndices() const
  {
    return receiveIndices_;
  }
  const std::vector<double>& receivedValues() const { return recvValues_; }

  // Shared input is component-major; received values are entry-major. Only
  // neighbors participate, using buffers and requests allocated during setup.
  void
  exchangeShared(const double* sharedValues, std::size_t componentStride = 0);

  // Host convenience API. The returned buffers are reused until the next setup.
  // Input is owned entries followed by shared entries, component-major.
  const OwnedEntries&
  reduce(const double* values, std::size_t componentStride = 0);

private:
  int owner(HYPRE_Int row) const;
  void freeRequests();

  MPI_Comm comm_{MPI_COMM_NULL};
  std::vector<HYPRE_Int> lower_;
  std::vector<HYPRE_Int> upper_;
  bool isSetup_{false};
  std::size_t numOwned_{0}, numShared_{0};
  unsigned components_{1};
  OwnedEntries entries_;
  std::vector<std::size_t> sendIndices_;
  std::vector<std::size_t> localOffsets_, localIndices_;
  std::vector<std::size_t> receiveOffsets_, receiveIndices_;
  std::vector<double> sendValues_, recvValues_;
  std::vector<MPI_Request> requests_;
};

} // namespace kynema_ugf
} // namespace sierra

#endif
