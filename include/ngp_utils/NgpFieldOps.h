// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPFIELDOPS_H
#define NGPFIELDOPS_H

/** \file
 *  \brief Field update utilities within element loops with SIMD data.
 *
 *  NgpFieldOps provides two utility classes that deal with updates of nodal and
 *  element fields from within a SIMD-ized Kokkos::parallel_for loop.
 */

#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpScratchData.h"
#include "SimdInterface.h"

#include "stk_mesh/base/Ngp.hpp"

#include <type_traits>

namespace sierra {
namespace nalu {
namespace nalu_ngp {
namespace impl {

template <typename Mesh, typename Field>
struct SimpleNodeFieldOp
{
  static_assert(
    std::is_floating_point<typename Field::value_type>::value,
    "NGP field must have a floating type");

  /**
   *  @param ngpMesh Instance of the NGP mesh on device
   *  @param ngpField The nodal field instance that is being modified
   *  @param einfo Entity information for this loop instance
   */
  KOKKOS_INLINE_FUNCTION
  SimpleNodeFieldOp(const Mesh& ngpMesh, const Field& ngpField)
    : ngpMesh_(ngpMesh), ngpField_(ngpField)
  {
  }

  KOKKOS_DEFAULTED_FUNCTION ~SimpleNodeFieldOp() = default;

  /** Implementation of supported operators for nodal fields
   */
  struct Ops
  {
    KOKKOS_DEFAULTED_FUNCTION
    Ops() = default;

    KOKKOS_DEFAULTED_FUNCTION ~Ops() = default;

    KOKKOS_INLINE_FUNCTION
    void operator=(const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto& nodes = einfo_.entityNodes;
      fld.get(msh, nodes[ni], ic) = val;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto& nodes = einfo_.entityNodes;
      Kokkos::atomic_add(&fld.get(msh, nodes[ni], ic), val);
    }

    KOKKOS_INLINE_FUNCTION
    void operator-=(const double& val) const { Ops::operator+=(-val); }

    const SimpleNodeFieldOp<Mesh, Field>& obj_;

    const EntityInfo<Mesh>& einfo_;

    //! Index of the node in connectivity array
    unsigned ni;

    //! Component index for the field to be updated
    unsigned ic;
  };

  /** Get the operator object to perform field modifications
   *
   *  @param n Index of node in the element connectivity array
   *  @param ic Index of the component
   */
  KOKKOS_INLINE_FUNCTION
  const Ops operator()(
    const EntityInfo<Mesh>& einfo,
    const unsigned n,
    const unsigned ic = 0) const
  {
    return Ops{*this, einfo, n, ic};
  }

  //! NGP Mesh instance
  const Mesh ngpMesh_;

  const Field ngpField_;
};

/** Update an NGP field registered on NODE_RANK with SIMD right hand sides.
 */
template <typename Mesh, typename Field, typename SimdDataType>
struct NodeFieldOp
{
  static_assert(
    std::is_floating_point<typename Field::value_type>::value,
    "NGP field must have a floating type");

  /**
   *  @param ngpMesh Instance of the NGP mesh on device
   *  @param ngpField The nodal field instance that is being modified
   *  @param Element connectivity information for this loop instance
   */
  KOKKOS_INLINE_FUNCTION
  NodeFieldOp(const Mesh& ngpMesh, const Field& ngpField)
    : ngpMesh_(ngpMesh), ngpField_(ngpField)
  {
  }

  KOKKOS_DEFAULTED_FUNCTION
  NodeFieldOp() = default;

  KOKKOS_DEFAULTED_FUNCTION ~NodeFieldOp() = default;

  /** Implementation of the supported operators for the fields
   */
  struct Ops
  {
    KOKKOS_DEFAULTED_FUNCTION
    Ops() = default;

    KOKKOS_DEFAULTED_FUNCTION ~Ops() = default;

    KOKKOS_INLINE_FUNCTION
    void operator=(const DoubleType& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();
#ifdef STK_SIMD_NONE
      fld.get(msh, einfo[0].entityNodes[ni], ic) = stk::simd::get_data(val, 0);
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        fld.get(msh, einfo[is].entityNodes[ni], ic) =
          stk::simd::get_data(val, is);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const DoubleType& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();
#ifdef STK_SIMD_NONE
      Kokkos::atomic_add(
        &fld.get(msh, einfo[0].entityNodes[ni], ic),
        stk::simd::get_data(val, 0));
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        Kokkos::atomic_add(
          &fld.get(msh, einfo[is].entityNodes[ni], ic),
          stk::simd::get_data(val, is));
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();
#ifdef STK_SIMD_NONE
      fld.get(msh, einfo[0].entityNodes[ni], ic) = val;
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        fld.get(msh, einfo[is].entityNodes[ni], ic) = val;
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();
#ifdef STK_SIMD_NONE
      Kokkos::atomic_add(&fld.get(msh, einfo[0].entityNodes[ni], ic), val);
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        Kokkos::atomic_add(&fld.get(msh, einfo[is].entityNodes[ni], ic), val);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator-=(const DoubleType& val) const { Ops::operator+=(-val); }

    KOKKOS_INLINE_FUNCTION
    void operator-=(const double& val) const { Ops::operator+=(-val); }

    const NodeFieldOp<Mesh, Field, SimdDataType>& obj_;

    //! Connectivity data for SIMD group
    const SimdDataType& edata_;

    //! Index of the node in the element connectivity array
    unsigned ni;
    //! Component index of the field to be updated
    unsigned ic;
  };

  /** Get the operator object to perform field modifications
   *
   *  @param n Index of node in the element connectivity array
   *  @param ic Index of the component
   */
  KOKKOS_INLINE_FUNCTION
  const Ops operator()(
    const SimdDataType& edata, const unsigned n, const unsigned ic = 0) const
  {
    return Ops{*this, edata, n, ic};
  }

  //! NGP Mesh instance
  const Mesh ngpMesh_;

  //! NGP element field to be updated
  const Field ngpField_;
};

/** Update an NGP field registered to ELEM_RANk from within SIMD-ized loop
 */
template <typename Mesh, typename Field, typename SimdDataType>
struct ElemFieldOp
{
  static_assert(
    std::is_floating_point<typename Field::value_type>::value,
    "NGP field must have a floating type");

  KOKKOS_INLINE_FUNCTION
  ElemFieldOp(const Field& ngpField) : ngpField_(ngpField) {}

  KOKKOS_DEFAULTED_FUNCTION ~ElemFieldOp() = default;

  /** Implementation of the operators
   */
  struct Ops
  {
    KOKKOS_DEFAULTED_FUNCTION
    Ops() = default;

    KOKKOS_DEFAULTED_FUNCTION ~Ops() = default;

    KOKKOS_INLINE_FUNCTION
    void operator=(const DoubleType& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) = stk::simd::get_data(val, 0);
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) = stk::simd::get_data(val, is);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const DoubleType& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();

      // No atomic_add here as only one element active per thread
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) += stk::simd::get_data(val, 0);
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) += stk::simd::get_data(val, is);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(const double& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) = val;
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) = val;
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const double& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = edata_.info();
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) += val;
#else
      for (int is = 0; is < edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) += val;
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator-=(const DoubleType& val) const { Ops::operator+=(-val); }

    KOKKOS_INLINE_FUNCTION
    void operator-=(const double& val) const { Ops::operator+=(-val); }

    const ElemFieldOp<Mesh, Field, SimdDataType>& obj_;

    //! Connectivity data for SIMD group
    const SimdDataType& edata_;

    //! Index of the component to be updated
    unsigned ic;
  };

  //! Return the operator to perform field modifications
  KOKKOS_INLINE_FUNCTION
  const Ops operator()(const SimdDataType& edata, const unsigned ic) const
  {
    return Ops{*this, edata, ic};
  }

  //! NGP element field to be updated
  const Field ngpField_;
};

} // namespace impl

/** Wrapper to generate a nodal field updater instance
 */
template <typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION impl::NodeFieldOp<Mesh, Field, ElemSimdData<Mesh>>
simd_elem_nodal_field_updater(const Mesh& mesh, const Field& fld)
{
  STK_NGP_ThrowAssert(fld.get_rank() == stk::topology::NODE_RANK);
  return impl::NodeFieldOp<Mesh, Field, ElemSimdData<Mesh>>{mesh, fld};
}

template <typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION impl::NodeFieldOp<Mesh, Field, FaceElemSimdData<Mesh>>
simd_face_elem_nodal_field_updater(const Mesh& mesh, const Field& fld)
{
  STK_NGP_ThrowAssert(fld.get_rank() == stk::topology::NODE_RANK);
  return impl::NodeFieldOp<Mesh, Field, FaceElemSimdData<Mesh>>{mesh, fld};
}

/** Wrapper to generate an element field updater instance
 */
template <typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION impl::ElemFieldOp<Mesh, Field, ElemSimdData<Mesh>>
simd_elem_field_updater(const Mesh&, const Field& fld)
{
  STK_NGP_ThrowAssert(
    (fld.get_rank() == stk::topology::ELEM_RANK) ||
    (fld.get_rank() == stk::topology::FACE_RANK) ||
    (fld.get_rank() == stk::topology::EDGE_RANK));
  return impl::ElemFieldOp<Mesh, Field, ElemSimdData<Mesh>>{fld};
}

/** Wrapper to generate an element field updater instance
 */
template <typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION impl::ElemFieldOp<Mesh, Field, FaceElemSimdData<Mesh>>
simd_face_elem_field_updater(const Mesh&, const Field& fld)
{
  STK_NGP_ThrowAssert(
    (fld.get_rank() == stk::topology::FACE_RANK) ||
    (fld.get_rank() == stk::topology::EDGE_RANK));
  return impl::ElemFieldOp<Mesh, Field, FaceElemSimdData<Mesh>>{fld};
}

template <typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION impl::SimpleNodeFieldOp<Mesh, Field>
edge_nodal_field_updater(const Mesh& mesh, const Field& fld)
{
  STK_NGP_ThrowAssert(fld.get_rank() == stk::topology::NODE_RANK);
  return impl::SimpleNodeFieldOp<Mesh, Field>{mesh, fld};
}

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* NGPFIELDOPS_H */
