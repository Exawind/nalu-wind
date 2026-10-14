// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "HypreDeterministicAssembly.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace sierra {
namespace kynema_ugf {

HypreDeterministicAssembly::HypreDeterministicAssembly(
  MPI_Comm comm, HYPRE_Int lower, HYPRE_Int upper)
{
  // Isolate assembly traffic from other users of the realm communicator.
  MPI_Comm_dup(comm, &comm_);
  int size;
  MPI_Comm_size(comm_, &size);
  lower_.resize(size);
  upper_.resize(size);
  MPI_Datatype indexType;
  MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(HYPRE_Int), &indexType);
  MPI_Allgather(&lower, 1, indexType, lower_.data(), 1, indexType, comm_);
  MPI_Allgather(&upper, 1, indexType, upper_.data(), 1, indexType, comm_);

  int lengths[3] = {1, 1, 1};
  MPI_Aint offsets[3] = {
    offsetof(Entry, row), offsetof(Entry, col), offsetof(Entry, value)};
  MPI_Datatype types[3] = {indexType, indexType, MPI_DOUBLE};
  MPI_Datatype packedType;
  MPI_Type_create_struct(3, lengths, offsets, types, &packedType);
  MPI_Type_create_resized(packedType, 0, sizeof(Entry), &entryType_);
  MPI_Type_commit(&entryType_);
  MPI_Type_free(&packedType);
}

HypreDeterministicAssembly::~HypreDeterministicAssembly()
{
  MPI_Type_free(&entryType_);
  MPI_Comm_free(&comm_);
}

int
HypreDeterministicAssembly::owner(HYPRE_Int row) const
{
  // Empty partitions may have equal lower bounds.
  auto it = std::upper_bound(lower_.begin(), lower_.end(), row);
  if (it == lower_.begin())
    return -1;
  int rank = static_cast<int>(it - lower_.begin() - 1);
  return row <= upper_[rank] ? rank : -1;
}

HypreDeterministicAssembly::OwnedEntries
HypreDeterministicAssembly::reduce(
  size_t numOwned,
  size_t numShared,
  const HYPRE_Int* rows,
  const HYPRE_Int* cols,
  const double* values) const
{
  const int size = static_cast<int>(lower_.size());
  std::vector<std::vector<Entry>> send(size), recv(size);
  int invalid = 0;
  for (size_t i = numOwned; i < numOwned + numShared; ++i) {
    const int dest = owner(rows[i]);
    if (dest < 0) {
      invalid = 1;
      continue;
    }
    send[dest].push_back({rows[i], cols ? cols[i] : 0, values[i]});
  }

  std::vector<int> sendCounts(size), recvCounts(size);
  for (int rank = 0; rank < size; ++rank) {
    if (
      send[rank].size() > static_cast<size_t>(std::numeric_limits<int>::max()))
      invalid = 1;
    else
      sendCounts[rank] = static_cast<int>(send[rank].size());
  }
  // Fail collectively rather than leaving other ranks waiting for messages.
  int anyInvalid;
  MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, comm_);
  if (anyInvalid)
    throw std::runtime_error(
      "Invalid row or MPI count overflow in deterministic hypre assembly");

  MPI_Alltoall(
    sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm_);
  std::vector<MPI_Request> requests;
  requests.reserve(2 * size);
  for (int rank = 0; rank < size; ++rank) {
    recv[rank].resize(recvCounts[rank]);
    if (recvCounts[rank]) {
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(
        recv[rank].data(), recvCounts[rank], entryType_, rank, 0, comm_,
        &requests.back());
    }
  }
  for (int rank = 0; rank < size; ++rank) {
    if (sendCounts[rank]) {
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Isend(
        send[rank].data(), sendCounts[rank], entryType_, rank, 0, comm_,
        &requests.back());
    }
  }
  if (!requests.empty())
    MPI_Waitall(
      static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  // Local first, then source rank and sender buffer order. The map also
  // supplies deterministic slots for entries absent from the local graph.
  std::map<std::pair<HYPRE_Int, HYPRE_Int>, double> merged;
  auto add = [&merged](HYPRE_Int row, HYPRE_Int col, double value) {
    auto result = merged.emplace(std::make_pair(row, col), value);
    if (!result.second)
      result.first->second += value;
  };
  for (size_t i = 0; i < numOwned; ++i)
    add(rows[i], cols ? cols[i] : 0, values[i]);
  for (int rank = 0; rank < size; ++rank)
    for (const auto& entry : recv[rank])
      add(entry.row, entry.col, entry.value);

  OwnedEntries result;
  result.rows.reserve(merged.size());
  result.cols.reserve(merged.size());
  result.values.reserve(merged.size());
  for (const auto& entry : merged) {
    result.rows.push_back(entry.first.first);
    result.cols.push_back(entry.first.second);
    result.values.push_back(entry.second);
  }
  return result;
}

} // namespace kynema_ugf
} // namespace sierra
