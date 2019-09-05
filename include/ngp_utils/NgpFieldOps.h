/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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

#include "stk_ngp/Ngp.hpp"

#include <type_traits>

namespace sierra {
namespace nalu {
namespace nalu_ngp {
namespace impl {

template<typename Mesh, typename Field>
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
  SimpleNodeFieldOp(
    const Mesh& ngpMesh,
    const Field& ngpField,
    const EntityInfo<Mesh>& einfo)
    : ngpMesh_(ngpMesh), ngpField_(ngpField), einfo_(einfo), ops_(*this)
  {}

  KOKKOS_FUNCTION ~SimpleNodeFieldOp() = default;

  /** Implementation of supported operators for nodal fields
   */
  struct Ops
  {
    KOKKOS_INLINE_FUNCTION
    Ops(SimpleNodeFieldOp<Mesh, Field>& obj) : obj_(obj)
    {}

    KOKKOS_FUNCTION ~Ops() = default;

    KOKKOS_INLINE_FUNCTION
    void operator=(const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto& nodes = obj_.einfo_.entityNodes;
      fld.get(msh, nodes[ni], ic) = val;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto& nodes = obj_.einfo_.entityNodes;
      Kokkos::atomic_add(&fld.get(msh, nodes[ni], ic), val);
    }

    KOKKOS_INLINE_FUNCTION
    void operator-=(const double& val) const { Ops::operator+=(-val); }

    SimpleNodeFieldOp<Mesh, Field>& obj_;

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
  const Ops& operator()(const int n, const int ic = 0) const
  {
    ops_.ni = n;
    ops_.ic = ic;
    return ops_;
  }

  //! NGP Mesh instance
  const Mesh ngpMesh_;

  const Field ngpField_;

  const EntityInfo<Mesh>& einfo_;

  mutable Ops ops_;
};

/** Update an NGP field registered on NODE_RANK with SIMD right hand sides.
 */
template <typename Mesh, typename Field>
struct NodeFieldOp
{
  static_assert(std::is_floating_point<typename Field::value_type>::value,
                "NGP field must have a floating type");

  /**
   *  @param ngpMesh Instance of the NGP mesh on device
   *  @param ngpField The nodal field instance that is being modified
   *  @param Element connectivity information for this loop instance
   */
  KOKKOS_INLINE_FUNCTION
  NodeFieldOp(
    const Mesh& ngpMesh,
    const Field& ngpField,
    const ElemSimdData<Mesh>& elemData
  ) : ngpMesh_(ngpMesh),
      ngpField_(ngpField),
      edata_(elemData),
      ops_(*this)
  {}

  KOKKOS_FUNCTION ~NodeFieldOp() = default;

  /** Implementation of the supported operators for the fields
   */
  struct Ops
  {
    KOKKOS_INLINE_FUNCTION
    Ops(NodeFieldOp<Mesh, Field>& obj) : obj_(obj)
    {}

    KOKKOS_FUNCTION ~Ops() = default;

    KOKKOS_INLINE_FUNCTION
    void operator= (const DoubleType& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;
#ifdef STK_SIMD_NONE
      fld.get(msh, einfo[0].entityNodes[ni], ic) =
        stk::simd::get_data(val, 0);
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        fld.get(msh, einfo[is].entityNodes[ni], ic) =
          stk::simd::get_data(val, is);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+= (const DoubleType& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;
#ifdef STK_SIMD_NONE
      Kokkos::atomic_add(
        &fld.get(msh, einfo[0].entityNodes[ni], ic),
        stk::simd::get_data(val, 0));
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        Kokkos::atomic_add(
          &fld.get(msh, einfo[is].entityNodes[ni], ic),
          stk::simd::get_data(val, is));
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator= (const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;
#ifdef STK_SIMD_NONE
      fld.get(msh, einfo[0].entityNodes[ni], ic) = val;
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        fld.get(msh, einfo[is].entityNodes[ni], ic) = val;
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+= (const double& val) const
    {
      const auto& msh = obj_.ngpMesh_;
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;
#ifdef STK_SIMD_NONE
      Kokkos::atomic_add(&fld.get(msh, einfo[0].entityNodes[ni], ic), val);
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        Kokkos::atomic_add(&fld.get(msh, einfo[is].entityNodes[ni], ic), val);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator-= (const DoubleType& val) const
    {
      Ops::operator+=(-val);
    }

    KOKKOS_INLINE_FUNCTION
    void operator-= (const double& val) const
    {
      Ops::operator+=(-val);
    }

    NodeFieldOp<Mesh, Field>& obj_;
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
  const Ops& operator()(const int n, const int ic = 0) const
  {
    ops_.ni = n;
    ops_.ic = ic;
    return ops_;
  }

  //! NGP Mesh instance
  const Mesh ngpMesh_;

  //! NGP element field to be updated
  const Field ngpField_;

  //! Connectivity data for SIMD group
  const ElemSimdData<Mesh>& edata_;

  mutable Ops ops_;
};

/** Update an NGP field registered to ELEM_RANk from within SIMD-ized loop
 */
template<typename Mesh, typename Field>
struct ElemFieldOp
{
  static_assert(std::is_floating_point<typename Field::value_type>::value,
                "NGP field must have a floating type");

  KOKKOS_INLINE_FUNCTION
  ElemFieldOp(
    const Mesh& ngpMesh,
    const Field& ngpField,
    const ElemSimdData<Mesh>& elemData
  ) : ngpMesh_(ngpMesh),
      ngpField_(ngpField),
      edata_(elemData),
      ops_(*this)
  {}

  KOKKOS_FUNCTION ~ElemFieldOp() = default;

  /** Implementation of the operators
   */
  struct Ops
  {
    KOKKOS_INLINE_FUNCTION
    Ops(ElemFieldOp<Mesh, Field>& obj) : obj_(obj)
    {}

    KOKKOS_FUNCTION ~Ops() = default;

    KOKKOS_INLINE_FUNCTION
    void operator= (const DoubleType& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) = stk::simd::get_data(val, 0);
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) = stk::simd::get_data(val, is);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+= (const DoubleType& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;

      // No atomic_add here as only one element active per thread
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) += stk::simd::get_data(val, 0);
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) += stk::simd::get_data(val, is);
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator= (const double& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) = val;
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) = val;
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator+= (const double& val) const
    {
      const auto& fld = obj_.ngpField_;
      const auto* einfo = obj_.edata_.elemInfo;
#ifdef STK_SIMD_NONE
      fld.get(einfo[0].meshIdx, ic) += val;
#else
      for (int is=0; is < obj_.edata_.numSimdElems; ++is) {
        fld.get(einfo[is].meshIdx, ic) += val;
      }
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator-= (const DoubleType& val) const
    {
      Ops::operator+=(-val);
    }

    KOKKOS_INLINE_FUNCTION
    void operator-= (const double& val) const
    {
      Ops::operator+=(-val);
    }

    ElemFieldOp<Mesh, Field>& obj_;

    //! Index of the component to be updated
    unsigned ic;
  };

  //! Return the operator to perform field modifications
  KOKKOS_INLINE_FUNCTION
  const Ops& operator()(const int ic) const
  {
    ops_.ic = ic;
    return ops_;
  }

  //! NGP Mesh instance
  const Mesh ngpMesh_;

  //! NGP element field to be updated
  const Field ngpField_;

  //! Connectivity data for SIMD group
  const ElemSimdData<Mesh>& edata_;

  mutable Ops ops_;
};

}  // impl

/** Wrapper to generate a nodal field updater instance
 */
template <typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION
impl::NodeFieldOp<Mesh, Field>
simd_nodal_field_updater(
  const Mesh& mesh, const Field& fld, const ElemSimdData<Mesh>& edata)
{
  NGP_ThrowAssert(fld.get_rank() == stk::topology::NODE_RANK);
  return impl::NodeFieldOp<Mesh, Field>{mesh, fld, edata};
}

/** Wrapper to generate an element field updater instance
 */
template <typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION
impl::ElemFieldOp<Mesh, Field>
simd_elem_field_updater(
  const Mesh& mesh, const Field& fld, const ElemSimdData<Mesh>& edata)
{
  NGP_ThrowAssert(
    (fld.get_rank() == stk::topology::ELEM_RANK)
    || (fld.get_rank() == stk::topology::FACE_RANK)
    || (fld.get_rank() == stk::topology::EDGE_RANK));
  return impl::ElemFieldOp<Mesh, Field>{mesh, fld, edata};
}

template<typename Mesh, typename Field>
KOKKOS_INLINE_FUNCTION
impl::SimpleNodeFieldOp<Mesh, Field>
edge_nodal_field_updater(
  const Mesh& mesh, const Field& fld, const EntityInfo<Mesh>& einfo)
{
  NGP_ThrowAssert(fld.get_rank() == stk::topology::NODE_RANK);
  return impl::SimpleNodeFieldOp<Mesh, Field>{mesh, fld, einfo};
}

}  // nalu_ngp
}  // nalu
}  // sierra


#endif /* NGPFIELDOPS_H */
