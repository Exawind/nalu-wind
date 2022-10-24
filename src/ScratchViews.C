// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ScratchViews.h>
#include <ngp_utils/NgpMEUtils.h>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Types.hpp>

#include <NaluEnv.h>

namespace sierra {
namespace nalu {

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void
gather_elem_node_field(
  const NGPDoubleFieldType& field,
  const stk::mesh::NgpMesh& ngpMesh,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  ViewType& shmemView)
{
  for (unsigned i = 0; i < elemNodes.size(); ++i) {
    shmemView[i] = field.get(ngpMesh, elemNodes[i], 0);
  }
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void
gather_elem_node_tensor_field(
  const NGPDoubleFieldType& field,
  const stk::mesh::NgpMesh& ngpMesh,
  int numNodes,
  int tensorDim1,
  int tensorDim2,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  ViewType& shmemView)
{
  NGP_ThrowRequireMsg(
    numNodes == (int)elemNodes.size(),
    "gather_elem_node_tensor_field, numNodes = mismatch with elemNodes.size()");
  for (int i = 0; i < numNodes; ++i) {
    unsigned counter = 0;
    for (int d1 = 0; d1 < tensorDim1; ++d1) {
      for (int d2 = 0; d2 < tensorDim2; ++d2) {
        shmemView(i, d1, d2) = field.get(ngpMesh, elemNodes[i], counter++);
      }
    }
  }
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void
gather_elem_tensor_field(
  const NGPDoubleFieldType& field,
  stk::mesh::FastMeshIndex elem,
  int tensorDim1,
  int tensorDim2,
  ViewType& shmemView)
{
  unsigned counter = 0;
  for (int d1 = 0; d1 < tensorDim1; ++d1) {
    for (int d2 = 0; d2 < tensorDim2; ++d2) {
      shmemView(d1, d2) = field.get(elem, counter++);
    }
  }
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void
gather_elem_vector_field(
  const NGPDoubleFieldType& field,
  stk::mesh::FastMeshIndex elem,
  int len,
  ViewType& shmemView)
{
  for (int i = 0; i < len; ++i) {
    shmemView(i) = field.get(elem, i);
  }
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void
gather_elem_node_field_3D(
  const NGPDoubleFieldType& field,
  const stk::mesh::NgpMesh& ngpMesh,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  ViewType& shmemView)
{
  for (unsigned i = 0; i < elemNodes.size(); ++i) {
    shmemView(i, 0) = field.get(ngpMesh, elemNodes[i], 0);
    shmemView(i, 1) = field.get(ngpMesh, elemNodes[i], 1);
    shmemView(i, 2) = field.get(ngpMesh, elemNodes[i], 2);
  }
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION void
gather_elem_node_field(
  const NGPDoubleFieldType& field,
  const stk::mesh::NgpMesh& ngpMesh,
  int scalarsPerNode,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  ViewType& shmemView)
{
  for (unsigned i = 0; i < elemNodes.size(); ++i) {
    for (int d = 0; d < scalarsPerNode; ++d) {
      shmemView(i, d) = field.get(ngpMesh, elemNodes[i], d);
    }
  }
}

inline void
gather_elem_node_field(
  const stk::mesh::FieldBase& field,
  int numNodes,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  SharedMemView<double*>& shmemView)
{
  for (int i = 0; i < numNodes; ++i) {
    shmemView[i] =
      *static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
  }
}

inline void
gather_elem_node_tensor_field(
  const stk::mesh::FieldBase& field,
  int numNodes,
  int tensorDim1,
  int tensorDim2,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  SharedMemView<double***>& shmemView)
{
  for (int i = 0; i < numNodes; ++i) {
    const double* dataPtr =
      static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
    unsigned counter = 0;
    for (int d1 = 0; d1 < tensorDim1; ++d1) {
      for (int d2 = 0; d2 < tensorDim2; ++d2) {
        shmemView(i, d1, d2) = dataPtr[counter++];
      }
    }
  }
}

inline void
gather_elem_tensor_field(
  const stk::mesh::FieldBase& field,
  stk::mesh::Entity elem,
  int tensorDim1,
  int tensorDim2,
  SharedMemView<double**>& shmemView)
{
  const double* dataPtr =
    static_cast<const double*>(stk::mesh::field_data(field, elem));
  unsigned counter = 0;
  for (int d1 = 0; d1 < tensorDim1; ++d1) {
    for (int d2 = 0; d2 < tensorDim2; ++d2) {
      shmemView(d1, d2) = dataPtr[counter++];
    }
  }
}

inline void
gather_elem_node_field_3D(
  const stk::mesh::FieldBase& field,
  int numNodes,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  SharedMemView<double**>& shmemView)
{
  for (int i = 0; i < numNodes; ++i) {
    const double* dataPtr =
      static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
    shmemView(i, 0) = dataPtr[0];
    shmemView(i, 1) = dataPtr[1];
    shmemView(i, 2) = dataPtr[2];
  }
}

inline void
gather_elem_node_field(
  const stk::mesh::FieldBase& field,
  int numNodes,
  int scalarsPerNode,
  const stk::mesh::NgpMesh::ConnectedNodes& elemNodes,
  SharedMemView<double**>& shmemView)
{
  for (int i = 0; i < numNodes; ++i) {
    const double* dataPtr =
      static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
    for (int d = 0; d < scalarsPerNode; ++d) {
      shmemView(i, d) = dataPtr[d];
    }
  }
}

int
get_num_scalars_pre_req_data(
  const ElemDataRequestsGPU& dataNeeded, int nDim, const ElemReqType reqType)
{
  /* master elements are allowed to be null if they are not required */
  MasterElement* meFC = dataNeeded.get_cvfem_face_me();
  MasterElement* meSCS = dataNeeded.get_cvfem_surface_me();
  MasterElement* meSCV = dataNeeded.get_cvfem_volume_me();
  MasterElement* meFEM = dataNeeded.get_fem_volume_me();

  const bool hasSCS = (meSCS != nullptr);
  // A MasterElement corresponding to ELEM_RANK has been registered
  const bool hasElemME = (hasSCS || meSCV != nullptr || meFEM != nullptr);
  // A MasterElement corresponding to side_rank() has been registered
  const bool hasFaceME = (meFC != nullptr);

  switch (reqType) {
  case ElemReqType::ELEM:
    NGP_ThrowRequireMsg(
      hasElemME, "Requesting ELEM data, but no ELEM_RANK master element has "
                 "been registered");
    break;

  case ElemReqType::FACE:
    NGP_ThrowRequireMsg(
      hasFaceME || hasSCS, "Request SIDE_RANK data, but no SIDE_RANK master "
                           "element has been registered");
    break;

  case ElemReqType::FACE_ELEM:
    // In case of FACE_ELEM register meFC so that numFaceIp can be queried
    NGP_ThrowRequireMsg(
      (hasSCS && hasFaceME),
      "Requesting FACE_ELEM data but does not have necessary MasterElements");
    break;
  }

  // The previous check guarantees that we get the correct nodesPerEntity for
  // all request types
  const int nodesPerEntity = nodes_per_entity(dataNeeded);

  int numScalars = 0;

  const ElemDataRequestsGPU::FieldInfoView::HostMirror& neededFields =
    dataNeeded.get_host_fields();
  for (unsigned f = 0; f < neededFields.size(); ++f) {
    const FieldInfoNGP& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field.get_rank();
    unsigned scalarsPerEntity = fieldInfo.scalarsDim1;
    unsigned entitiesPerElem =
      fieldEntityRank == stk::topology::NODE_RANK ? nodesPerEntity : 1;

    if (fieldInfo.scalarsDim2 > 1) {
      scalarsPerEntity *= fieldInfo.scalarsDim2;
    }
    numScalars += entitiesPerElem * scalarsPerEntity;
  }

  const int numFaceIp = num_integration_points(dataNeeded, METype::FACE);
  const int numScsIp = num_integration_points(dataNeeded, METype::SCS);
  const int numScvIp = num_integration_points(dataNeeded, METype::SCV);
  const int numFemIp = num_integration_points(dataNeeded, METype::FEM);

  const ElemDataRequestsGPU::CoordsTypesView::HostMirror& coordsTypes =
    dataNeeded.get_host_coordinates_types();
  for (unsigned i = 0; i < coordsTypes.size(); ++i) {
    auto cType = coordsTypes(i);
    const ElemDataRequestsGPU::DataEnumView::HostMirror& dataEnums =
      dataNeeded.get_host_data_enums(cType);
    int dndxLength = 0, dndxLengthFC = 0, gUpperLength = 0, gLowerLength = 0;

    // Updated logic for data sharing of deriv and det_j
    bool needDeriv = false;
    bool needDerivScv = false;
    bool needDerivFem = false;
    bool needDerivFC = false;
    bool needDetj = false;
    bool needDetjScv = false;
    bool needDetjFem = false;
    bool needDetjFC = false;

    for (unsigned d = 0; d < dataEnums.size(); ++d) {
      ELEM_DATA_NEEDED data = dataEnums(d);
      switch (data) {
      case FC_AREAV:
        numScalars += nDim * numFaceIp;
        break;
      case FC_SHAPE_FCN:
      case FC_SHIFTED_SHAPE_FCN:
        numScalars += numFaceIp * nodesPerEntity;
        break;
      case SCS_AREAV:
        numScalars += nDim * numScsIp;
        break;
      case SCS_FACE_GRAD_OP:
      case SCS_SHIFTED_FACE_GRAD_OP:
        dndxLengthFC = nodesPerEntity * numFaceIp * nDim;
        needDerivFC = true;
        needDetjFC = true;
        numScalars += dndxLengthFC;
        break;
      case SCS_GRAD_OP:
      case SCS_SHIFTED_GRAD_OP:
        dndxLength = nodesPerEntity * numScsIp * nDim;
        needDeriv = true;
        needDetj = true;
        numScalars += dndxLength;
        break;
      case SCS_SHAPE_FCN:
      case SCS_SHIFTED_SHAPE_FCN:
        numScalars += nodesPerEntity * numScsIp;
        break;
      case SCV_VOLUME:
        numScalars += numScvIp;
        break;
      case SCV_GRAD_OP:
        dndxLength = nodesPerEntity * numScvIp * nDim;
        needDerivScv = true;
        needDetjScv = true;
        numScalars += dndxLength;
        break;
      case SCV_SHAPE_FCN:
      case SCV_SHIFTED_SHAPE_FCN:
        numScalars += nodesPerEntity * numScvIp;
        break;
      case SCS_GIJ:
        gUpperLength = nDim * nDim * numScsIp;
        gLowerLength = nDim * nDim * numScsIp;
        needDeriv = true;
        numScalars += (gUpperLength + gLowerLength);
        break;
      case FEM_GRAD_OP:
      case FEM_SHIFTED_GRAD_OP:
        dndxLength = nodesPerEntity * numFemIp * nDim;
        needDerivFem = true;
        needDetjFem = true;
        numScalars += dndxLength;
        break;
      case FEM_SHAPE_FCN:
      case FEM_SHIFTED_SHAPE_FCN:
        numScalars += nodesPerEntity * numFemIp;
        break;
      default:
        break;
      }
    }

    if (needDerivFC)
      numScalars += nodesPerEntity * numFaceIp * nDim;

    if (needDeriv)
      numScalars += nodesPerEntity * numScsIp * nDim;

    if (needDerivScv)
      numScalars += nodesPerEntity * numScvIp * nDim;

    if (needDerivFem)
      numScalars += nodesPerEntity * numFemIp * nDim;

    if (needDetjFC)
      numScalars += numFaceIp;

    if (needDetj)
      numScalars += numScsIp;

    if (needDetjScv)
      numScalars += numScvIp;

    if (needDetjFem)
      numScalars += numFemIp;
  }

  // Add a 64 byte padding to the buffer size requested
  return numScalars + 8;
}

template <typename T>
KOKKOS_FUNCTION void
fill_pre_req_data(
  const ElemDataRequestsGPU& dataNeeded,
  const stk::mesh::NgpMesh& ngpMesh,
  stk::mesh::EntityRank entityRank,
  stk::mesh::Entity entity,
  ScratchViews<T, DeviceTeamHandleType, DeviceShmem>& prereqData)
{
  stk::mesh::FastMeshIndex entityIndex = ngpMesh.fast_mesh_index(entity);
  prereqData.elemNodes = ngpMesh.get_nodes(entityRank, entityIndex);
  int nodesPerElem = prereqData.elemNodes.size();

  const ElemDataRequestsGPU::FieldInfoView& neededFields =
    dataNeeded.get_fields();
  for (unsigned f = 0; f < neededFields.size(); ++f) {
    const FieldInfoNGP& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    bool isTensorField = fieldInfo.scalarsDim2 > 1;

    if (
      fieldEntityRank == stk::topology::EDGE_RANK ||
      fieldEntityRank == stk::topology::FACE_RANK ||
      fieldEntityRank == stk::topology::ELEM_RANK) {
      if (isTensorField) {
        auto& shmemView =
          prereqData.get_scratch_view_2D(get_field_ordinal(fieldInfo));
        gather_elem_tensor_field(
          fieldInfo.field, entityIndex, scalarsDim1, fieldInfo.scalarsDim2,
          shmemView);
      } else {
        auto& shmemView =
          prereqData.get_scratch_view_1D(get_field_ordinal(fieldInfo));
        unsigned len = shmemView.extent(0);
        gather_elem_vector_field(fieldInfo.field, entityIndex, len, shmemView);
      }
    } else if (fieldEntityRank == stk::topology::NODE_RANK) {
      if (isTensorField) {
        auto& shmemView3D =
          prereqData.get_scratch_view_3D(get_field_ordinal(fieldInfo));
        gather_elem_node_tensor_field(
          fieldInfo.field, ngpMesh, nodesPerElem, scalarsDim1,
          fieldInfo.scalarsDim2, prereqData.elemNodes, shmemView3D);
      } else {
        if (scalarsDim1 == 1) {
          auto& shmemView1D =
            prereqData.get_scratch_view_1D(get_field_ordinal(fieldInfo));
          gather_elem_node_field(
            fieldInfo.field, ngpMesh, prereqData.elemNodes, shmemView1D);
        } else {
          auto& shmemView2D =
            prereqData.get_scratch_view_2D(get_field_ordinal(fieldInfo));
          if (scalarsDim1 == 3) {
            gather_elem_node_field_3D(
              fieldInfo.field, ngpMesh, prereqData.elemNodes, shmemView2D);
          } else {
            gather_elem_node_field(
              fieldInfo.field, ngpMesh, scalarsDim1, prereqData.elemNodes,
              shmemView2D);
          }
        }
      }
    } else {
      NGP_ThrowRequireMsg(
        false, "Unknown stk-rank in ScratchViewsNGP.C::fill_pre_req_data");
    }
  }
}

template KOKKOS_FUNCTION void fill_pre_req_data(
  const ElemDataRequestsGPU& dataNeeded,
  const stk::mesh::NgpMesh& ngpMesh,
  stk::mesh::EntityRank entityRank,
  stk::mesh::Entity entity,
  ScratchViews<double, DeviceTeamHandleType, DeviceShmem>& prereqData);

template KOKKOS_FUNCTION void fill_pre_req_data(
  const ElemDataRequestsGPU& dataNeeded,
  const stk::mesh::NgpMesh& ngpMesh,
  stk::mesh::EntityRank entityRank,
  stk::mesh::Entity entity,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& prereqData);
} // namespace nalu
} // namespace sierra
