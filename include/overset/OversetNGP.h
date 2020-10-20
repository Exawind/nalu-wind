#ifndef OVERSETNGP_H
#define OVERSETNGP_H

#include "KokkosInterface.h"

namespace tioga_nalu {

template<typename T,
         typename Layout = Kokkos::LayoutRight,
         typename Space = sierra::nalu::MemSpace>
using OversetArrayViewType = Kokkos::View<T, Layout, Space>;

template<typename T,
         typename Layout = Kokkos::LayoutRight,
         typename Space = sierra::nalu::MemSpace>
struct OversetArrayType
{
  using ArrayType = OversetArrayViewType<T, Layout, Space>;
  using HostArrayType = typename ArrayType::HostMirror;

  ArrayType d_view;
  HostArrayType h_view;

  OversetArrayType() : d_view(), h_view() {}

  OversetArrayType(const std::string& label, const size_t len)
    : d_view(label, len), h_view(Kokkos::create_mirror_view(d_view))
  {
  }

  size_t size() const { return d_view.size(); }

  void init(const std::string& label, const size_t len)
  {
    d_view = ArrayType(label, len);
    h_view = Kokkos::create_mirror_view(d_view);
  }

  void sync_device() { Kokkos::deep_copy(d_view, h_view); }

  void sync_host() { Kokkos::deep_copy(h_view, d_view); }
};

}

#endif /* OVERSETNGP_H */
