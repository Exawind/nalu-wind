/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScratchViewsHO_h
#define ScratchViewsHO_h


#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_ngp/Ngp.hpp>

#include <ElemDataRequests.h>
#include <master_element/MasterElement.h>
#include <KokkosInterface.h>

#include <SimdInterface.h>

#include <element_promotion/NodeMapMaker.h>
#include <BuildTemplates.h>
#include <ScratchViews.h>


#include <set>
#include <type_traits>

namespace sierra{
namespace nalu{

template<typename T>
class ScratchViewsHO
{
public:
  typedef T value_type;

  ScratchViewsHO(const TeamHandleType& team,
               const ngp::Mesh& ngpMesh,
               unsigned totalNumFields,
               int order, int dim,
               const ElemDataRequests& dataNeeded);


  virtual ~ScratchViewsHO() = default;

  T* get_scratch_view_ptr(const stk::mesh::FieldBase& field)
  {
    return fieldViews[field.mesh_meta_data_ordinal()].data();
  }

  T* get_scratch_view_ptr(const ngp::Field<double>& field)
  {
    return fieldViews[field.get_ordinal()].data();
  }

  template <typename CompiledTimeSizedViewType> CompiledTimeSizedViewType get_scratch_view(const stk::mesh::FieldBase& field)
  {
    return CompiledTimeSizedViewType(get_scratch_view_ptr(field));
  }

  template <typename CompiledTimeSizedViewType> CompiledTimeSizedViewType get_scratch_view(const ngp::Field<double>& field)
  {
    return CompiledTimeSizedViewType(get_scratch_view_ptr(field));
  }

  template <typename ViewType> ViewType get_scratch_view(const stk::mesh::FieldBase& field, int n0)
  {
    return ViewType(get_scratch_view_ptr(field), n0);
  }

  template <typename ViewType> ViewType get_scratch_view(const ngp::Field<double>& field, int n0)
  {
    return ViewType(get_scratch_view_ptr(field), n0);
  }

  template <typename ViewType> ViewType get_scratch_view(const stk::mesh::FieldBase& field, int n0, int n1)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1);
  }

  template <typename ViewType> ViewType get_scratch_view(const ngp::Field<double>& field, int n0, int n1)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1);
  }

  template <typename ViewType> ViewType get_scratch_view(const stk::mesh::FieldBase& field,  int n0, int n1, int n2)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1, n2);
  }

  template <typename ViewType> ViewType get_scratch_view(const ngp::Field<double>& field,  int n0, int n1, int n2)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1, n2);
  }

  template <typename ViewType> ViewType get_scratch_view(const stk::mesh::FieldBase& field, int n0, int n1, int n2, int n3)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1, n2, n3);
  }

  template <typename ViewType> ViewType get_scratch_view(const ngp::Field<double>& field, int n0, int n1, int n2, int n3)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1, n2, n3);
  }

  template <typename ViewType> ViewType get_scratch_view(const stk::mesh::FieldBase& field, int n0, int n1, int n2, int n3, int n4)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1, n2, n3, n4);
  }

  template <typename ViewType> ViewType get_scratch_view(const ngp::Field<double>& field, int n0, int n1, int n2, int n3, int n4)
  {
    return ViewType(get_scratch_view_ptr(field), n0, n1, n2, n3, n4);
  }

  int total_bytes() const { return num_bytes_required; }

  std::array<ngp::Mesh::ConnectedNodes, simdLen> elemNodes;
  int numSimdElems{simdLen};

  const std::vector<SharedMemView<T*>>& get_field_views() const { return fieldViews; }

private:
  std::vector<SharedMemView<T*>> fieldViews{};
  int num_bytes_required{0};
};


template<typename T>
ScratchViewsHO<T>::ScratchViewsHO(const TeamHandleType& team,
             const ngp::Mesh& ngpMesh,
             unsigned totalNumFields,
             int order, int dim,
             const ElemDataRequests& dataNeeded)
{
  int numScalars = 0;
  fieldViews.resize(totalNumFields);

  const FieldSet& neededFields = dataNeeded.get_fields();
  for(const FieldInfo& fieldInfo : neededFields) {
    ThrowAssert(fieldInfo.field.get_rank() == stk::topology::NODE_RANK);
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;

    const int n1D = order + 1;
    if (scalarsDim1 == 1u) {
      fieldViews[fieldInfo.field.get_ordinal()] = get_shmem_view_1D<T>(team, n1D * n1D * n1D);
      numScalars += n1D * n1D * n1D;
    }

    if (scalarsDim1 == 3u) {
      fieldViews[fieldInfo.field.get_ordinal()] = get_shmem_view_1D<T>(team, 3 * n1D * n1D * n1D);
      numScalars += 3 * n1D * n1D * n1D;
    }

  }
  // Track total bytes required for field allocation
  num_bytes_required += numScalars * sizeof(T);
}

} // namespace nalu
} // namespace Sierra

#endif
