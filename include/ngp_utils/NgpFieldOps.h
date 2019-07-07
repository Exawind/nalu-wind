/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NGPFIELDOPS_H
#define NGPFIELDOPS_H

/** \file
 *  \brief Helper objects for field updates when using SIMD datatypes
 */

#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpScratchData.h"
#include "SimdInterface.h"

#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {
namespace nalu_ngp {

/** Helper object to perform operations between SIMD data and non-SIMD
 *  ngp::Field objects.
 *
 *  This updater is used when looping over the elements of a mesh
 */
template<typename Mesh, typename FieldDataType>
struct ElemFieldOp
{
  /**
   *  @param ngpMesh Instance of the ngp::Mesh
   *  @param ngpField Field to be updated
   *  @param elemData Element connectivity data structure
   */
  KOKKOS_FORCEINLINE_FUNCTION
  ElemFieldOp(
    const Mesh& ngpMesh,
    const ngp::Field<FieldDataType>& ngpField,
    const ElemSimdData<Mesh>& elemData
  ) : ngpMesh_(ngpMesh),
      ngpField_(ngpField),
      edata_(elemData)
  {}

  KOKKOS_FUNCTION ~ElemFieldOp() = default;

  /** Set an ELEM_RANK quantity
   *
   *  This method sets an element quantity for the desired component (e.g.,
   *  integration point for a scalar) by scattering the contents of the SIMD
   *  group. On non-SIMD architectures, it just simply copies the value.
   *
   *  @param component The array index to be set
   *  @param value The SIMD data to be scattered
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void ip_set(const int component, const DoubleType& value) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    for (int is=0; is < edata_.numSimdElems; ++is) {
      ngpField_.get(edata_.elemInfo[is].meshIdx, component) =
        stk::simd::get_data(value, is);
    }
#else
    ngpField_.get(edata_.elemInfo[0].meshIdx, component) =
      stk::simd::get_data(value, 0);
#endif
  }

  /** Add to an ELEM_RANK quantity
   *
   *  This method atomically adds to an element quantity for the desired
   *  component (e.g., integration point for a scalar) by scattering the
   *  contents of the SIMD group.
   *
   *  @param component The array index to be set
   *  @param value The SIMD data to be added to various elements
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void ip_add(const int component, const DoubleType& value) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    for (int is=0; is < edata_.numSimdElems; ++is) {
      Kokkos::atomic_add(
        &ngpField_.get(edata_.elemInfo[is].meshIdx, component),
        stk::simd::get_data(value, is));
    }
#else
    Kokkos::atomic_add(
      &ngpField_.get(edata_.elemInfo[0].meshIdx, component),
      stk::simd::get_data(value, 0));
#endif
  }

  /** Set value for a NODE_RANK field from within an element loop
   *
   *  This method sets the i-th component of a nodal quantity belonging to the
   *  n-th node connected to an element from within an element loop.
   *
   *  @param n The node index into the element connectivity array
   *  @param ic The component index for the field data array
   *  @param value The SIMD data to to be scattered
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void set(const int n, const int ic, const DoubleType& value) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    for (int is=0; is < edata_.numSimdElems; ++is) {
      ngpField_.get(ngpMesh_, edata_.elemInfo[is].entityNodes[n], ic) =
        stk::simd::get_data(value, is);
    }
#else
    ngpField_.get(ngpMesh_, edata_.elemInfo[0].entityNodes[n], ic) =
      stk::simd::get_data(value, 0);
#endif
  }

  /** Add to the value of a NODE_RANK field from within an element loop
   *
   *  This method adds to the i-th component of a nodal quantity belonging to
   *  the n-th node connected to an element from within an element loop.
   *
   *  @param n The node index into the element connectivity array
   *  @param ic The component index for the field data array
   *  @param value The SIMD data to to be atomically added
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void add(const int n, const int ic, const DoubleType& value) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    for (int is=0; is < edata_.numSimdElems; ++is) {
      Kokkos::atomic_add(
        &ngpField_.get(ngpMesh_, edata_.elemInfo[is].entityNodes[n], ic),
        stk::simd::get_data(value, is));
    }
#else
    Kokkos::atomic_add(
      &ngpField_.get(ngpMesh_, edata_.elemInfo[0].entityNodes[n], ic),
      stk::simd::get_data(value, 0));
#endif
  }

  //! NGP Mesh instance
  const Mesh& ngpMesh_;

  //! NGP element field to be updated
  const ngp::Field<FieldDataType>& ngpField_;

  //! Connectivity data for SIMD group
  const ElemSimdData<Mesh>& edata_;
};

/** Create a field updater instance for modifying fields within an NGP loop
 *
 *  @param mesh The NGP-mesh instance
 *  @param field The NGP-field that is being modified
 *  @param edata The element scratch data within a SIMD loop
 */
template<typename Mesh, typename FieldDataType>
KOKKOS_FORCEINLINE_FUNCTION
ElemFieldOp<Mesh, FieldDataType>
simd_field_updater(
  const Mesh& mesh,
  const ngp::Field<FieldDataType>& field,
  const ElemSimdData<Mesh>& edata)
{
  return ElemFieldOp<Mesh, FieldDataType>(mesh, field, edata);
}


}  // nalu_ngp
}  // nalu
}  // sierra



#endif /* NGPFIELDOPS_H */
