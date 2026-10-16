// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "HypreDeterministicAssembly.h"

#include <algorithm>
#include <cstddef>
#include <limits>
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
}

HypreDeterministicAssembly::~HypreDeterministicAssembly()
{
  freeRequests();
  MPI_Comm_free(&comm_);
}

void
HypreDeterministicAssembly::freeRequests()
{
  for (auto& request : requests_)
    MPI_Request_free(&request);
  requests_.clear();
  isSetup_ = false;
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

void
HypreDeterministicAssembly::setup(
  size_t numOwned,
  size_t numShared,
  const HYPRE_Int* rows,
  const HYPRE_Int* cols,
  unsigned components)
{
  freeRequests();
  numOwned_ = numOwned;
  numShared_ = numShared;
  components_ = components;
  const int size = static_cast<int>(lower_.size());
  int myRank;
  MPI_Comm_rank(comm_, &myRank);
  std::vector<int> destinations(numShared);
  std::vector<int> sendCounts(size, 0), recvCounts(size);
  int invalid = 0;
  const auto maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
  for (size_t i = 0; i < numOwned; ++i)
    if (owner(rows[i]) != myRank)
      invalid = 1;
  for (size_t i = 0; i < numShared; ++i) {
    const int dest = owner(rows[numOwned + i]);
    destinations[i] = dest;
    if (dest < 0) {
      invalid = 1;
      continue;
    }
    if (static_cast<size_t>(sendCounts[dest]) == maxCount)
      invalid = 1;
    else
      ++sendCounts[dest];
  }

  // Both the index-pair exchange and the batched value exchange use int counts.
  for (const auto count : sendCounts)
    if (
      static_cast<size_t>(count) > maxCount / 2 ||
      (components && static_cast<size_t>(count) > maxCount / components))
      invalid = 1;
  unsigned minComponents, maxComponents;
  MPI_Allreduce(&components, &minComponents, 1, MPI_UNSIGNED, MPI_MIN, comm_);
  MPI_Allreduce(&components, &maxComponents, 1, MPI_UNSIGNED, MPI_MAX, comm_);
  invalid |= minComponents != maxComponents;
  // Fail collectively rather than leaving peers waiting for messages.
  int anyInvalid;
  MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, comm_);
  if (anyInvalid)
    throw std::runtime_error(
      "Invalid row or MPI count overflow in deterministic hypre assembly");

  MPI_Alltoall(
    sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm_);

  std::vector<size_t> sendOffsets(size + 1, 0), recvOffsets(size + 1, 0);
  for (int rank = 0; rank < size; ++rank) {
    sendOffsets[rank + 1] = sendOffsets[rank] + sendCounts[rank];
    recvOffsets[rank + 1] = recvOffsets[rank] + recvCounts[rank];
  }
  sendIndices_.resize(numShared);
  auto next = sendOffsets;
  for (size_t i = 0; i < numShared; ++i)
    sendIndices_[next[destinations[i]]++] = i;

  // Exchange the fixed graph once, keeping source rank and sender-buffer order.
  std::vector<HYPRE_Int> sendKeys(2 * numShared);
  std::vector<HYPRE_Int> recvKeys(2 * recvOffsets.back());
  for (size_t i = 0; i < numShared; ++i) {
    const auto input = numOwned + sendIndices_[i];
    sendKeys[2 * i] = rows[input];
    sendKeys[2 * i + 1] = cols ? cols[input] : 0;
  }
  MPI_Datatype indexType;
  MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(HYPRE_Int), &indexType);
  std::vector<MPI_Request> indexRequests;
  indexRequests.reserve(2 * size);
  for (int rank = 0; rank < size; ++rank) {
    if (recvCounts[rank]) {
      indexRequests.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(
        recvKeys.data() + 2 * recvOffsets[rank], 2 * recvCounts[rank],
        indexType, rank, 0, comm_, &indexRequests.back());
    }
  }
  for (int rank = 0; rank < size; ++rank) {
    if (sendCounts[rank]) {
      indexRequests.push_back(MPI_REQUEST_NULL);
      MPI_Isend(
        sendKeys.data() + 2 * sendOffsets[rank], 2 * sendCounts[rank],
        indexType, rank, 0, comm_, &indexRequests.back());
    }
  }
  if (!indexRequests.empty())
    MPI_Waitall(
      static_cast<int>(indexRequests.size()), indexRequests.data(),
      MPI_STATUSES_IGNORE);

  using Key = std::pair<HYPRE_Int, HYPRE_Int>;
  std::vector<Key> keys;
  keys.reserve(numOwned + recvOffsets.back());
  for (size_t i = 0; i < numOwned; ++i)
    keys.emplace_back(rows[i], cols ? cols[i] : 0);
  for (size_t i = 0; i < recvOffsets.back(); ++i)
    keys.emplace_back(recvKeys[2 * i], recvKeys[2 * i + 1]);
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  entries_.rows.resize(keys.size());
  entries_.cols.resize(keys.size());
  entries_.values.resize(keys.size() * components);
  for (size_t i = 0; i < keys.size(); ++i) {
    entries_.rows[i] = keys[i].first;
    entries_.cols[i] = keys[i].second;
  }
  auto slot = [&keys](HYPRE_Int row, HYPRE_Int col) {
    return static_cast<size_t>(
      std::lower_bound(keys.begin(), keys.end(), Key{row, col}) - keys.begin());
  };

  // CSR lists allow each output slot to be reduced independently on a device,
  // without changing the sequential floating-point order within that slot.
  std::vector<size_t> localSlots(numOwned), recvSlots(recvOffsets.back());
  for (size_t i = 0; i < numOwned; ++i)
    localSlots[i] = slot(rows[i], cols ? cols[i] : 0);
  for (size_t i = 0; i < recvSlots.size(); ++i)
    recvSlots[i] = slot(recvKeys[2 * i], recvKeys[2 * i + 1]);
  auto makeLists = [&keys](
                     const std::vector<size_t>& slots,
                     std::vector<size_t>& offsets,
                     std::vector<size_t>& indices) {
    offsets.assign(keys.size() + 1, 0);
    for (auto dest : slots)
      ++offsets[dest + 1];
    for (size_t i = 1; i < offsets.size(); ++i)
      offsets[i] += offsets[i - 1];
    auto cursor = offsets;
    indices.resize(slots.size());
    for (size_t i = 0; i < slots.size(); ++i)
      indices[cursor[slots[i]]++] = i;
  };
  makeLists(localSlots, localOffsets_, localIndices_);
  makeLists(recvSlots, receiveOffsets_, receiveIndices_);

  sendValues_.resize(numShared * components);
  recvValues_.resize(recvOffsets.back() * components);
  requests_.reserve(indexRequests.size());
  for (int rank = 0; rank < size; ++rank) {
    if (recvCounts[rank] && components) {
      requests_.push_back(MPI_REQUEST_NULL);
      MPI_Recv_init(
        recvValues_.data() + recvOffsets[rank] * components,
        recvCounts[rank] * components, MPI_DOUBLE, rank, 1, comm_,
        &requests_.back());
    }
  }
  for (int rank = 0; rank < size; ++rank) {
    if (sendCounts[rank] && components) {
      requests_.push_back(MPI_REQUEST_NULL);
      MPI_Send_init(
        sendValues_.data() + sendOffsets[rank] * components,
        sendCounts[rank] * components, MPI_DOUBLE, rank, 1, comm_,
        &requests_.back());
    }
  }
  isSetup_ = true;
}

void
HypreDeterministicAssembly::exchangeShared(
  const double* sharedValues, size_t componentStride)
{
  if (!isSetup_)
    throw std::logic_error("Deterministic hypre assembly plan is not set up");
  if (!componentStride)
    componentStride = numShared_;
  for (size_t i = 0; i < sendIndices_.size(); ++i)
    for (unsigned c = 0; c < components_; ++c)
      sendValues_[i * components_ + c] =
        sharedValues[c * componentStride + sendIndices_[i]];
  if (!requests_.empty()) {
    const int count = static_cast<int>(requests_.size());
    MPI_Startall(count, requests_.data());
    MPI_Waitall(count, requests_.data(), MPI_STATUSES_IGNORE);
  }
}

const HypreDeterministicAssembly::OwnedEntries&
HypreDeterministicAssembly::reduce(const double* values, size_t componentStride)
{
  if (!componentStride)
    componentStride = numOwned_ + numShared_;
  exchangeShared(
    numShared_ && components_ ? values + numOwned_ : nullptr, componentStride);
  const size_t count = entries_.rows.size();
  for (unsigned c = 0; c < components_; ++c) {
    for (size_t slot = 0; slot < count; ++slot) {
      auto local = localOffsets_[slot];
      auto remote = receiveOffsets_[slot];
      // Initialize from the first contribution, preserving signed zero.
      double value;
      if (local < localOffsets_[slot + 1])
        value = values[c * componentStride + localIndices_[local++]];
      else
        value = recvValues_[receiveIndices_[remote++] * components_ + c];
      for (; local < localOffsets_[slot + 1]; ++local)
        value += values[c * componentStride + localIndices_[local]];
      for (; remote < receiveOffsets_[slot + 1]; ++remote)
        value += recvValues_[receiveIndices_[remote] * components_ + c];
      entries_.values[c * count + slot] = value;
    }
  }
  return entries_;
}

} // namespace kynema_ugf
} // namespace sierra
