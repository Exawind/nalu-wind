/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScratchViews_h
#define ScratchViews_h

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>

#include <ElemDataRequestsNGP.h>
#include <ElemDataRequestsGPU.h>
#include <master_element/MasterElement.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>
#include <MultiDimViews.h>

#include <set>
#include <type_traits>

namespace sierra{
namespace nalu{

struct ScratchMeInfo {
  int nodalGatherSize_;
  int nodesPerFace_;
  int nodesPerElement_;
  int numFaceIp_;
  int numScsIp_;
  int numScvIp_;
  int numFemIp_;
};

template<typename ELEMDATAREQUESTSTYPE>
KOKKOS_INLINE_FUNCTION
NumNeededViews count_needed_field_views(const ELEMDATAREQUESTSTYPE& dataNeeded)
{
  NumNeededViews numNeededViews = {0, 0, 0, 0};

  const typename ELEMDATAREQUESTSTYPE::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned i=0; i<neededFields.size(); ++i) {
    const typename ELEMDATAREQUESTSTYPE::FieldInfoType& fieldInfo = neededFields(i);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    unsigned scalarsDim2 = fieldInfo.scalarsDim2;

    if (fieldEntityRank==stk::topology::EDGE_RANK ||
        fieldEntityRank==stk::topology::FACE_RANK ||
        fieldEntityRank==stk::topology::ELEM_RANK) {
      if (scalarsDim2 == 0) {
        numNeededViews.num1DViews++;
      }
      else {
        numNeededViews.num2DViews++;
      }
    }
    else if (fieldEntityRank==stk::topology::NODE_RANK) {
      if (scalarsDim2 == 0) {
        if (scalarsDim1 == 1) {
          numNeededViews.num1DViews++;
        }
        else {
          numNeededViews.num2DViews++;
        }
      }
      else {
          numNeededViews.num3DViews++;
      }
    }
    else {
      NGP_ThrowRequireMsg(false,"Unknown stk-rank");
    }
  }

  return numNeededViews;
}

template<typename T>
class MasterElementViews
{
public:
  typedef T value_type;

  MasterElementViews() = default;
  virtual ~MasterElementViews() = default;

  int create_master_element_views(
    const TeamHandleType& team,
    const ElemDataRequestsNGP::DataEnumView& dataEnums,
    int nDim, int nodesPerFace, int nodesPerElem,
    int numFaceIp, int numScsIp, int numScvIp, int numFemIp);

#ifdef KOKKOS_ENABLE_CUDA
  int create_master_element_views(
    const DeviceTeamHandleType& team,
    const ElemDataRequestsGPU::DataEnumView& dataEnums,
    int nDim, int nodesPerFace, int nodesPerElem,
    int numFaceIp, int numScsIp, int numScvIp, int numFemIp);
#endif

  void fill_master_element_views(
    const ElemDataRequestsNGP::DataEnumView& dataEnums,
    SharedMemView<double**>* coordsView,
    MasterElement* meFC,
    MasterElement* meSCS,
    MasterElement* meSCV,
    MasterElement* meFEM,
    int faceOrdinal = 0);

  void fill_master_element_views_new_me(
    const ElemDataRequestsNGP::DataEnumView& dataEnums,
    SharedMemView<DoubleType**>* coordsView,
    MasterElement* meFC,
    MasterElement* meSCS,
    MasterElement* meSCV,
    MasterElement* meFEM,
    int faceOrdinal = 0);

  SharedMemView<T**> fc_areav;
  SharedMemView<T**> scs_areav;
  SharedMemView<T***> dndx_fc_scs;
  SharedMemView<T***> dndx_shifted_fc_scs;
  SharedMemView<T***> dndx;
  SharedMemView<T***> dndx_shifted;
  SharedMemView<T***> dndx_scv;
  SharedMemView<T***> dndx_scv_shifted;
  SharedMemView<T***> dndx_fem;
  SharedMemView<T***> deriv_fc_scs;
  SharedMemView<T***> deriv;
  SharedMemView<T***> deriv_scv;
  SharedMemView<T***> deriv_fem;
  SharedMemView<T*> det_j_fc_scs;
  SharedMemView<T*> det_j;
  SharedMemView<T*> det_j_scv;
  SharedMemView<T*> det_j_fem;
  SharedMemView<T*> scv_volume;
  SharedMemView<T***> gijUpper;
  SharedMemView<T***> gijLower;
  SharedMemView<T***> metric;
};

template<typename T,typename SHMEM,typename TEAMHANDLETYPE,typename ELEMDATAREQUESTSTYPE, typename MULTIDIMVIEWSTYPE>
KOKKOS_FUNCTION
int create_needed_field_views(const TEAMHANDLETYPE& team,
                               const ELEMDATAREQUESTSTYPE& dataNeeded,
                               int nodesPerEntity,
                               MULTIDIMVIEWSTYPE& fieldViews)
{
  int numScalars = 0;

  const typename ELEMDATAREQUESTSTYPE::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned i=0; i<neededFields.size(); ++i) {
    const typename ELEMDATAREQUESTSTYPE::FieldInfoType& fieldInfo = neededFields(i);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    unsigned scalarsDim2 = fieldInfo.scalarsDim2;

    if (fieldEntityRank==stk::topology::EDGE_RANK ||
        fieldEntityRank==stk::topology::FACE_RANK ||
        fieldEntityRank==stk::topology::ELEM_RANK) {
      if (scalarsDim2 == 0) {
        fieldViews.add_1D_view(get_field_ordinal(fieldInfo), get_shmem_view_1D<T,TEAMHANDLETYPE,SHMEM>(team, scalarsDim1));
        numScalars += scalarsDim1;
      }
      else {
        fieldViews.add_2D_view(get_field_ordinal(fieldInfo), get_shmem_view_2D<T,TEAMHANDLETYPE,SHMEM>(team, scalarsDim1, scalarsDim2));
        numScalars += scalarsDim1 * scalarsDim2;
      }
    }
    else if (fieldEntityRank==stk::topology::NODE_RANK) {
      if (scalarsDim2 == 0) {
        if (scalarsDim1 == 1) {
          fieldViews.add_1D_view(get_field_ordinal(fieldInfo), get_shmem_view_1D<T,TEAMHANDLETYPE,SHMEM>(team, nodesPerEntity));
          numScalars += nodesPerEntity;
        }
        else {
          fieldViews.add_2D_view(get_field_ordinal(fieldInfo), get_shmem_view_2D<T,TEAMHANDLETYPE,SHMEM>(team, nodesPerEntity, scalarsDim1));
          numScalars += nodesPerEntity*scalarsDim1;
        }
      }
      else {
          fieldViews.add_3D_view(get_field_ordinal(fieldInfo), get_shmem_view_3D<T,TEAMHANDLETYPE,SHMEM>(team, nodesPerEntity, scalarsDim1, scalarsDim2));
          numScalars += nodesPerEntity*scalarsDim1*scalarsDim2;
      }
    }
    else {
      ThrowRequireMsg(false,"Unknown stk-rank" << fieldEntityRank);
    }
  }

  return numScalars;
}

template<typename T, typename TEAMHANDLETYPE=TeamHandleType, typename SHMEM=HostShmem>
class ScratchViews
{
public:
  typedef T value_type;

  KOKKOS_FUNCTION
  ScratchViews(const TEAMHANDLETYPE& team,
               unsigned nDim,
               int nodesPerEntity,
               const ElemDataRequestsNGP& dataNeeded);

  KOKKOS_FUNCTION
  ScratchViews(const TEAMHANDLETYPE& team,
               unsigned nDim,
               int nodesPerEntity,
               const ElemDataRequestsGPU& dataNeeded);

  KOKKOS_FUNCTION
  ScratchViews(const TEAMHANDLETYPE& team,
               unsigned nDim,
               const ScratchMeInfo &meInfo,
               const ElemDataRequestsNGP& dataNeeded);

  KOKKOS_FUNCTION
  ScratchViews(const TEAMHANDLETYPE& team,
               unsigned nDim,
               const ScratchMeInfo &meInfo,
               const ElemDataRequestsGPU& dataNeeded);

  KOKKOS_FUNCTION
  virtual ~ScratchViews() {
  }

  inline
  SharedMemView<T*,SHMEM>& get_scratch_view_1D(const stk::mesh::FieldBase& field);

  inline
  SharedMemView<T**,SHMEM>& get_scratch_view_2D(const stk::mesh::FieldBase& field);

  inline
  SharedMemView<T***,SHMEM>& get_scratch_view_3D(const stk::mesh::FieldBase& field);

  inline
  SharedMemView<T****,SHMEM>& get_scratch_view_4D(const stk::mesh::FieldBase& field);

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T*,SHMEM>& get_scratch_view_1D(const unsigned fieldOrdinal);

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T**,SHMEM>& get_scratch_view_2D(const unsigned fieldOrdinal);

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T***,SHMEM>& get_scratch_view_3D(const unsigned fieldOrdinal);

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T****,SHMEM>& get_scratch_view_4D(const unsigned fieldOrdinal);

  inline
  MasterElementViews<T>& get_me_views(const COORDS_TYPES cType)
  {
    ThrowRequire(hasCoordField[cType] == true);
    return meViews[cType];
  }

  KOKKOS_INLINE_FUNCTION
  bool has_coord_field(const COORDS_TYPES cType) const { return hasCoordField[cType]; }

  KOKKOS_INLINE_FUNCTION
  int total_bytes() const { return num_bytes_required; }

  ngp::Mesh::ConnectedNodes elemNodes;

  KOKKOS_INLINE_FUNCTION
        MultiDimViews<T,TEAMHANDLETYPE,SHMEM>& get_field_views()       { return fieldViews; }
  KOKKOS_INLINE_FUNCTION
  const MultiDimViews<T,TEAMHANDLETYPE,SHMEM>& get_field_views() const { return fieldViews; }

private:
  void create_needed_master_element_views(const TEAMHANDLETYPE& team,
                                          const ElemDataRequestsNGP& dataNeeded,
                                          int nDim, int nodesPerFace, int nodesPerElem,
                                          int numFaceIp, int numScsIp, int numScvIp, int numFemIp);

  void create_needed_master_element_views(const TEAMHANDLETYPE& team,
                                          const ElemDataRequestsGPU& dataNeeded,
                                          int nDim, int nodesPerFace, int nodesPerElem,
                                          int numFaceIp, int numScsIp, int numScvIp, int numFemIp);

  MultiDimViews<T,TEAMHANDLETYPE, SHMEM> fieldViews;
  MasterElementViews<T> meViews[MAX_COORDS_TYPES];
  bool hasCoordField[MAX_COORDS_TYPES] = {false, false};
  int num_bytes_required{0};
};

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
SharedMemView<T*,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_1D(const stk::mesh::FieldBase& field)
{ 
//  ThrowAssertMsg(fieldViews[field.mesh_meta_data_ordinal()] != nullptr, "ScratchViews ERROR, trying to get 1D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
//  ViewT<SharedMemView<T*>>* vt = static_cast<ViewT<SharedMemView<T*>>*>(fieldViews[field.mesh_meta_data_ordinal()]);
  return fieldViews.get_scratch_view_1D(field.mesh_meta_data_ordinal());
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
SharedMemView<T**,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_2D(const stk::mesh::FieldBase& field)
{ 
//  ThrowAssertMsg(fieldViews[field.mesh_meta_data_ordinal()] != nullptr, "ScratchViews ERROR, trying to get 2D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
//  ViewT<SharedMemView<T**>>* vt = static_cast<ViewT<SharedMemView<T**>>*>(fieldViews[field.mesh_meta_data_ordinal()]);
  return fieldViews.get_scratch_view_2D(field.mesh_meta_data_ordinal());
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
SharedMemView<T***,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_3D(const stk::mesh::FieldBase& field)
{ 
//  ThrowAssertMsg(fieldViews[field.mesh_meta_data_ordinal()] != nullptr, "ScratchViews ERROR, trying to get 3D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
//  ViewT<SharedMemView<T***>>* vt = static_cast<ViewT<SharedMemView<T***>>*>(fieldViews[field.mesh_meta_data_ordinal()]);
  return fieldViews.get_scratch_view_3D(field.mesh_meta_data_ordinal());
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
SharedMemView<T****,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_4D(const stk::mesh::FieldBase& field)
{
//  ThrowAssertMsg(fieldViews[field.mesh_meta_data_ordinal()] != nullptr, "ScratchViews ERROR, trying to get 4D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
//  ViewT<SharedMemView<T****>>* vt = static_cast<ViewT<SharedMemView<T****>>*>(fieldViews[field.mesh_meta_data_ordinal()]);
  return fieldViews.get_scratch_view_4D(field.mesh_meta_data_ordinal());
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T*,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_1D(const unsigned fieldOrdinal)
{ 
  return fieldViews.get_scratch_view_1D(fieldOrdinal);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T**,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_2D(const unsigned fieldOrdinal)
{ 
  return fieldViews.get_scratch_view_2D(fieldOrdinal);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T***,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_3D(const unsigned fieldOrdinal)
{
  return fieldViews.get_scratch_view_3D(fieldOrdinal);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T****,SHMEM>& ScratchViews<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_4D(const unsigned fieldOrdinal)
{
  return fieldViews.get_scratch_view_4D(fieldOrdinal);
}

template<typename T>
int MasterElementViews<T>::create_master_element_views(
  const TeamHandleType& team,
  const ElemDataRequestsNGP::DataEnumView& dataEnums,
  int nDim, int /* nodesPerFace */, int nodesPerElem,
  int numFaceIp, int numScsIp, int numScvIp, int numFemIp)
{
  int numScalars = 0;
  bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
  bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;
  bool femGradOp = false; bool femShiftedGradOp = false;
  for(unsigned i=0; i<dataEnums.size(); ++i) {
    switch(dataEnums(i))
    {
      case FC_AREAV:
          ThrowRequireMsg(numFaceIp > 0, "ERROR, meFC must be non-null if FC_AREAV is requested.");
          fc_areav = get_shmem_view_2D<T>(team, numFaceIp, nDim);
          numScalars += numFaceIp * nDim;
          break;
      case SCS_FACE_GRAD_OP:
          ThrowRequireMsg(numFaceIp > 0, "ERROR, meSCS must be non-null if SCS_FACE_GRAD_OP is requested.");
          dndx_fc_scs = get_shmem_view_3D<T>(team, numFaceIp, nodesPerElem, nDim);
          numScalars += nodesPerElem * numFaceIp * nDim;
          needDerivFC = true;
          needDetjFC = true;
          break;
      case SCS_SHIFTED_FACE_GRAD_OP:
          ThrowRequireMsg(numFaceIp > 0, "ERROR, meSCS must be non-null if SCS_SHIFTED_FACE_GRAD_OP is requested.");
          dndx_shifted_fc_scs = get_shmem_view_3D<T>(team, numFaceIp, nodesPerElem, nDim);
          numScalars += nodesPerElem * numFaceIp * nDim;
          needDerivFC = true;
          needDetjFC = true;
          break;
      case SCS_AREAV:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_AREAV is requested.");
         scs_areav = get_shmem_view_2D<T>(team, numScsIp, nDim);
         numScalars += numScsIp * nDim;
         break;

      case SCS_GRAD_OP:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_GRAD_OP is requested.");
         dndx = get_shmem_view_3D<T>(team, numScsIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numScsIp * nDim;
         needDeriv = true;
         needDetj = true;
         break;

      case SCS_SHIFTED_GRAD_OP:
        ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_SHIFTED_GRAD_OP is requested.");
        dndx_shifted = get_shmem_view_3D<T>(team, numScsIp, nodesPerElem, nDim);
        numScalars += nodesPerElem * numScsIp * nDim;
        needDeriv = true;
        needDetj = true;
        break;

      case SCS_GIJ:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_GIJ is requested.");
         gijUpper = get_shmem_view_3D<T>(team, numScsIp, nDim, nDim);
         gijLower = get_shmem_view_3D<T>(team, numScsIp, nDim, nDim);
         numScalars += 2 * numScsIp * nDim * nDim;
         needDeriv = true;
         break;

      case SCV_MIJ:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCV must be non-null if SCV_MIJ is requested.");
         metric = get_shmem_view_3D<T>(team, numScvIp, nDim, nDim);
         numScalars += numScvIp * nDim * nDim;
         needDeriv = true;
         break;

      case SCV_VOLUME:
         ThrowRequireMsg(numScvIp > 0, "ERROR, meSCV must be non-null if SCV_VOLUME is requested.");
         scv_volume = get_shmem_view_1D<T>(team, numScvIp);
         numScalars += numScvIp;
         break;

      case SCV_GRAD_OP:
         ThrowRequireMsg(numScvIp > 0, "ERROR, meSCV must be non-null if SCV_GRAD_OP is requested.");
         dndx_scv = get_shmem_view_3D<T>(team, numScvIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numScvIp * nDim;
         needDerivScv = true;
         needDetjScv = true;
         break;

      case SCV_SHIFTED_GRAD_OP:
         ThrowRequireMsg(numScvIp > 0, "ERROR, meSCV must be non-null if SCV_SHIFTED_GRAD_OP is requested.");
         dndx_scv_shifted = get_shmem_view_3D<T>(team, numScvIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numScvIp * nDim;
         needDerivScv = true;
         needDetjScv = true;
         break;

      case FEM_GRAD_OP:
         ThrowRequireMsg(numFemIp > 0, "ERROR, meFEM must be non-null if FEM_GRAD_OP is requested.");
         dndx_fem = get_shmem_view_3D<T>(team, numFemIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numFemIp * nDim;
         needDerivFem = true;
         needDetjFem = true;
         femGradOp = true;
         break;

      case FEM_SHIFTED_GRAD_OP:
         ThrowRequireMsg(numFemIp > 0, "ERROR, meFEM must be non-null if FEM_SHIFTED_GRAD_OP is requested.");
         dndx_fem = get_shmem_view_3D<T>(team, numFemIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numFemIp * nDim;
         needDerivFem = true;
         needDetjFem = true;
         femShiftedGradOp = true;
         break;

      default: break;
    }
  }

  if (needDerivFC) {
    deriv_fc_scs = get_shmem_view_3D<T>(team, numFaceIp,nodesPerElem,nDim);
    numScalars += numFaceIp * nodesPerElem * nDim;
  }

  if (needDeriv) {
    deriv = get_shmem_view_3D<T>(team, numScsIp,nodesPerElem,nDim);
    numScalars += numScsIp * nodesPerElem * nDim;
  }

  if (needDerivScv) {
    deriv_scv = get_shmem_view_3D<T>(team, numScvIp,nodesPerElem,nDim);
    numScalars += numScvIp * nodesPerElem * nDim;
  }

  if (needDerivFem) {
    deriv_fem = get_shmem_view_3D<T>(team, numFemIp,nodesPerElem,nDim);
    numScalars += numFemIp * nodesPerElem * nDim;
  }

  if (needDetjFC) {
    det_j_fc_scs = get_shmem_view_1D<T>(team, numFaceIp);
    numScalars += numFaceIp;
  }

  if (needDetj) {
    det_j = get_shmem_view_1D<T>(team, numScsIp);
    numScalars += numScsIp;
  }

  if (needDetjScv) {
    det_j_scv = get_shmem_view_1D<T>(team, numScvIp);
    numScalars += numScvIp;
  }

  if (needDetjFem) {
    det_j_fem = get_shmem_view_1D<T>(team, numFemIp);
    numScalars += numFemIp;
  }

  // error check
  if ( femGradOp && femShiftedGradOp )
    ThrowRequireMsg(numFemIp > 0, "ERROR, femGradOp and femShiftedGradOp both requested.");

  return numScalars;
}

#ifdef KOKKOS_ENABLE_CUDA
template<typename T>
int MasterElementViews<T>::create_master_element_views(
  const DeviceTeamHandleType& team,
  const ElemDataRequestsGPU::DataEnumView& dataEnums,
  int nDim, int nodesPerFace, int nodesPerElem,
  int numFaceIp, int numScsIp, int numScvIp, int numFemIp)
{
  int numScalars = 0;
  bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
  bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;
  bool femGradOp = false; bool femShiftedGradOp = false;
  for(unsigned i=0; i<dataEnums.size(); ++i) {
    switch(dataEnums(i))
    {
      case FC_AREAV:
          ThrowRequireMsg(numFaceIp > 0, "ERROR, meFC must be non-null if FC_AREAV is requested.");
          fc_areav = get_shmem_view_2D<T>(team, numFaceIp, nDim);
          numScalars += numFaceIp * nDim;
          break;
      case SCS_FACE_GRAD_OP:
          ThrowRequireMsg(numFaceIp > 0, "ERROR, meSCS must be non-null if SCS_FACE_GRAD_OP is requested.");
          dndx_fc_scs = get_shmem_view_3D<T>(team, numFaceIp, nodesPerElem, nDim);
          numScalars += nodesPerElem * numFaceIp * nDim;
          needDerivFC = true;
          needDetjFC = true;
          break;
      case SCS_SHIFTED_FACE_GRAD_OP:
          ThrowRequireMsg(numFaceIp > 0, "ERROR, meSCS must be non-null if SCS_SHIFTED_FACE_GRAD_OP is requested.");
          dndx_shifted_fc_scs = get_shmem_view_3D<T>(team, numFaceIp, nodesPerElem, nDim);
          numScalars += nodesPerElem * numFaceIp * nDim;
          needDerivFC = true;
          needDetjFC = true;
          break;
      case SCS_AREAV:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_AREAV is requested.");
         scs_areav = get_shmem_view_2D<T>(team, numScsIp, nDim);
         numScalars += numScsIp * nDim;
         break;

      case SCS_GRAD_OP:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_GRAD_OP is requested.");
         dndx = get_shmem_view_3D<T>(team, numScsIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numScsIp * nDim;
         needDeriv = true;
         needDetj = true;
         break;

      case SCS_SHIFTED_GRAD_OP:
        ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_SHIFTED_GRAD_OP is requested.");
        dndx_shifted = get_shmem_view_3D<T>(team, numScsIp, nodesPerElem, nDim);
        numScalars += nodesPerElem * numScsIp * nDim;
        needDeriv = true;
        needDetj = true;
        break;

      case SCS_GIJ:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCS must be non-null if SCS_GIJ is requested.");
         gijUpper = get_shmem_view_3D<T>(team, numScsIp, nDim, nDim);
         gijLower = get_shmem_view_3D<T>(team, numScsIp, nDim, nDim);
         numScalars += 2 * numScsIp * nDim * nDim;
         needDeriv = true;
         break;

      case SCV_MIJ:
         ThrowRequireMsg(numScsIp > 0, "ERROR, meSCV must be non-null if SCV_MIJ is requested.");
         metric = get_shmem_view_3D<T>(team, numScvIp, nDim, nDim);
         numScalars += numScvIp * nDim * nDim;
         needDeriv = true;
         break;

      case SCV_VOLUME:
         ThrowRequireMsg(numScvIp > 0, "ERROR, meSCV must be non-null if SCV_VOLUME is requested.");
         scv_volume = get_shmem_view_1D<T>(team, numScvIp);
         numScalars += numScvIp;
         break;

      case SCV_GRAD_OP:
         ThrowRequireMsg(numScvIp > 0, "ERROR, meSCV must be non-null if SCV_GRAD_OP is requested.");
         dndx_scv = get_shmem_view_3D<T>(team, numScvIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numScvIp * nDim;
         needDerivScv = true;
         needDetjScv = true;
         break;

      case SCV_SHIFTED_GRAD_OP:
         ThrowRequireMsg(numScvIp > 0, "ERROR, meSCV must be non-null if SCV_SHIFTED_GRAD_OP is requested.");
         dndx_scv_shifted = get_shmem_view_3D<T>(team, numScvIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numScvIp * nDim;
         needDerivScv = true;
         needDetjScv = true;
         break;

      case FEM_GRAD_OP:
         ThrowRequireMsg(numFemIp > 0, "ERROR, meFEM must be non-null if FEM_GRAD_OP is requested.");
         dndx_fem = get_shmem_view_3D<T>(team, numFemIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numFemIp * nDim;
         needDerivFem = true;
         needDetjFem = true;
         femGradOp = true;
         break;

      case FEM_SHIFTED_GRAD_OP:
         ThrowRequireMsg(numFemIp > 0, "ERROR, meFEM must be non-null if FEM_SHIFTED_GRAD_OP is requested.");
         dndx_fem = get_shmem_view_3D<T>(team, numFemIp, nodesPerElem, nDim);
         numScalars += nodesPerElem * numFemIp * nDim;
         needDerivFem = true;
         needDetjFem = true;
         femShiftedGradOp = true;
         break;

      default: break;
    }
  }

  if (needDerivFC) {
    deriv_fc_scs = get_shmem_view_3D<T>(team, numFaceIp,nodesPerElem,nDim);
    numScalars += numFaceIp * nodesPerElem * nDim;
  }

  if (needDeriv) {
    deriv = get_shmem_view_3D<T>(team, numScsIp,nodesPerElem,nDim);
    numScalars += numScsIp * nodesPerElem * nDim;
  }

  if (needDerivScv) {
    deriv_scv = get_shmem_view_3D<T>(team, numScvIp,nodesPerElem,nDim);
    numScalars += numScvIp * nodesPerElem * nDim;
  }

  if (needDerivFem) {
    deriv_fem = get_shmem_view_3D<T>(team, numFemIp,nodesPerElem,nDim);
    numScalars += numFemIp * nodesPerElem * nDim;
  }

  if (needDetjFC) {
    det_j_fc_scs = get_shmem_view_1D<T>(team, numFaceIp);
    numScalars += numFaceIp;
  }

  if (needDetj) {
    det_j = get_shmem_view_1D<T>(team, numScsIp);
    numScalars += numScsIp;
  }

  if (needDetjScv) {
    det_j_scv = get_shmem_view_1D<T>(team, numScvIp);
    numScalars += numScvIp;
  }

  if (needDetjFem) {
    det_j_fem = get_shmem_view_1D<T>(team, numFemIp);
    numScalars += numFemIp;
  }

  // error check
  if ( femGradOp && femShiftedGradOp )
    ThrowRequireMsg(numFemIp > 0, "ERROR, femGradOp and femShiftedGradOp both requested.");

  return numScalars;
}
#endif

template<typename T>
void MasterElementViews<T>::fill_master_element_views(
  const ElemDataRequestsNGP::DataEnumView& dataEnums,
  SharedMemView<double**>* coordsView,
  MasterElement* /* meFC */,
  MasterElement* meSCS,
  MasterElement* meSCV,
  MasterElement* meFEM,
  int /* faceOrdinal */)
{
  // Guard against calling MasterElement methods on SIMD data structures
  static_assert(std::is_same<T, double>::value,
                "Cannot populate MasterElement Views for non-double data types");

  double error = 0.0;
  for(unsigned i=0; i<dataEnums.size(); ++i) {
    switch(dataEnums(i))
    {
      case FC_AREAV:
        ThrowRequireMsg(false, "ERROR, non-interleaving FC_AREAV is not supported.");
        break;
      case SCS_AREAV:
        ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_AREAV is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_AREAV requested.");
        meSCS->determinant(1, &((*coordsView)(0, 0)), &scs_areav(0, 0), &error);
        break;
      case SCS_FACE_GRAD_OP:
        ThrowRequireMsg(false, "ERROR, non-interleaving FACE_GRAD_OP is not supported.");
        break;
      case SCS_SHIFTED_FACE_GRAD_OP:
        ThrowRequireMsg(false, "ERROR, non-interleaving SCS_SHIFTED_FACE_GRAD_OP is not supported.");
        break;
      case SCS_GRAD_OP:
        ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_GRAD_OP requested.");
        meSCS->grad_op(1, &((*coordsView)(0, 0)), &dndx(0, 0, 0), &deriv(0, 0, 0), &det_j(0), &error);
        break;
      case SCS_SHIFTED_GRAD_OP:
        ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_GRAD_OP requested.");
        meSCS->shifted_grad_op(1, &((*coordsView)(0, 0)), &dndx_shifted(0, 0, 0), &deriv(0, 0, 0), &det_j(0), &error);
        break;
      case SCS_GIJ:
        ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_GIJ is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_GIJ requested.");
        meSCS->gij(&((*coordsView)(0, 0)), &gijUpper(0, 0, 0), &gijLower(0, 0, 0), &deriv(0, 0, 0));
        break;
      case SCS_MIJ:
        ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCS needs to be non-null if SCS_MIJ is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_MIJ requested.");
        meSCS->Mij(&((*coordsView)(0,0)), &metric(0,0,0), &deriv(0,0,0));
        break;
      case SCV_MIJ:
        ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCV needs to be non-null if SCV_MIJ is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCV_MIJ requested.");
        meSCV->Mij(&((*coordsView)(0,0)), &metric(0,0,0), &deriv_scv(0,0,0));
        break;
      case SCV_VOLUME:
        ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCV needs to be non-null if SCV_VOLUME is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCV_VOLUME requested.");
        meSCV->determinant(1, &((*coordsView)(0, 0)), &scv_volume(0), &error);
        break;
      case SCV_GRAD_OP:
        ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCV needs to be non-null if SCV_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCV_GRAD_OP requested.");
        meSCV->grad_op(1, &((*coordsView)(0, 0)), &dndx_scv(0, 0, 0), &deriv_scv(0, 0, 0), &det_j_scv(0), &error);
        break;
      case FEM_GRAD_OP:
        ThrowRequireMsg(meFEM != nullptr, "ERROR, meFEM needs to be non-null if FEM_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but FEM_GRAD_OP requested.");
        meFEM->grad_op(1, &((*coordsView)(0, 0)), &dndx_fem(0, 0, 0), &deriv_fem(0, 0, 0), &det_j_fem(0), &error);
        break;
      case FEM_SHIFTED_GRAD_OP:
        ThrowRequireMsg(meFEM != nullptr, "ERROR, meFEM needs to be non-null if FEM_SHIFTED_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but FEM_GRAD_OP requested.");
        meFEM->shifted_grad_op(1, &((*coordsView)(0, 0)), &dndx_fem(0, 0, 0), &deriv_fem(0, 0, 0), &det_j_fem(0), &error);
        break;

      default:
        break;
    }
  }
}

template<typename T>
void MasterElementViews<T>::fill_master_element_views_new_me(
  const ElemDataRequestsNGP::DataEnumView& dataEnums,
  SharedMemView<DoubleType**>* coordsView,
  MasterElement* /* meFC */,
  MasterElement* meSCS,
  MasterElement* meSCV,
  MasterElement* meFEM,
  int faceOrdinal)
{
  for(unsigned i=0; i<dataEnums.size(); ++i) {
    switch(dataEnums(i))
    {
      case FC_AREAV:
        ThrowRequireMsg(false, "FC_AREAV not implemented yet.");
        break;
      case SCS_AREAV:
         ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_AREAV is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_AREAV requested.");
         meSCS->determinant(*coordsView, scs_areav);
         break;
      case SCS_FACE_GRAD_OP:
         ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_FACE_GRAD_OP is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_FACE_GRAD_OP requested.");
         meSCS->face_grad_op(faceOrdinal, *coordsView, dndx_fc_scs);
       break;
      case SCS_SHIFTED_FACE_GRAD_OP:
         ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_SHIFTED_FACE_GRAD_OP is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_SHIFTED_FACE_GRAD_OP requested.");
         meSCS->shifted_face_grad_op(faceOrdinal, *coordsView, dndx_shifted_fc_scs);
       break;
      case SCS_GRAD_OP:
         ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_GRAD_OP is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_GRAD_OP requested.");
         meSCS->grad_op(*coordsView, dndx, deriv);
         break;
      case SCS_SHIFTED_GRAD_OP:
        ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_GRAD_OP requested.");
        meSCS->shifted_grad_op(*coordsView, dndx_shifted, deriv);
        break;
      case SCS_GIJ:
         ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCS needs to be non-null if SCS_GIJ is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_GIJ requested.");
         meSCS->gij(*coordsView, gijUpper, gijLower, deriv);
         break;
      case SCS_MIJ:
         ThrowRequireMsg(meSCS != nullptr, "ERROR, meSCV needs to be non-null if SCS_MIJ is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCS_MIJ requested.");
         meSCS->Mij(*coordsView, metric, deriv);
         break;
      case SCV_MIJ:
         ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCV needs to be non-null if SCV_MIJ is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCV_MIJ requested.");
         meSCV->Mij(*coordsView, metric, deriv_scv);
         break;
      case SCV_VOLUME:
         ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCV needs to be non-null if SCV_VOLUME is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCV_VOLUME requested.");
         meSCV->determinant(*coordsView, scv_volume);
         break;
      case SCV_GRAD_OP:
        ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCV needs to be non-null if SCV_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCV_GRAD_OP requested.");
        meSCV->grad_op(*coordsView, dndx_scv, deriv_scv);
        break;
      case SCV_SHIFTED_GRAD_OP:
        ThrowRequireMsg(meSCV != nullptr, "ERROR, meSCV needs to be non-null if SCV_SHIFTED_GRAD_OP is requested.");
        ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but SCV_SHIFTED_GRAD_OP requested.");
        meSCV->shifted_grad_op(*coordsView, dndx_scv_shifted, deriv_scv);
        break;
      case FEM_GRAD_OP:
         ThrowRequireMsg(meFEM != nullptr, "ERROR, meFEM needs to be non-null if FEM_GRAD_OP is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but FEM_GRAD_OP requested.");
         meFEM->grad_op_fem(*coordsView, dndx_fem, deriv_fem, det_j_fem);
         break;
      case FEM_SHIFTED_GRAD_OP:
         ThrowRequireMsg(meFEM != nullptr, "ERROR, meFEM needs to be non-null if FEM_SHIFTED_GRAD_OP is requested.");
         ThrowRequireMsg(coordsView != nullptr, "ERROR, coords null but FEM_GRAD_OP requested.");
         meFEM->shifted_grad_op_fem(*coordsView, dndx_fem, deriv_fem, det_j_fem);
         break;

      default: break;
    }
  }
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
ScratchViews<T,TEAMHANDLETYPE,SHMEM>::ScratchViews(const TEAMHANDLETYPE& team,
             unsigned nDim,
             int nodalGatherSize,
             const ElemDataRequestsNGP& dataNeeded)
 : fieldViews(team, dataNeeded.get_total_num_fields(), count_needed_field_views(dataNeeded))
{
  num_bytes_required = create_needed_field_views<T,SHMEM>(team, dataNeeded, nodalGatherSize, fieldViews) * sizeof(T);

#ifndef KOKKOS_ENABLE_CUDA
  /* master elements are allowed to be null if they are not required */
  MasterElement *meFC = dataNeeded.get_cvfem_face_me();
  MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
  MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
  MasterElement *meFEM = dataNeeded.get_fem_volume_me();

  int nodesPerFace = meFC != nullptr ? meFC->nodesPerElement_ : 0;
  int nodesPerElem = meSCS != nullptr
          ? meSCS->nodesPerElement_ : meSCV != nullptr
          ? meSCV->nodesPerElement_ : meFEM != nullptr
          ? meFEM->nodesPerElement_ : 0;
  int numFaceIp= meFC  != nullptr ? meFC->numIntPoints_  : 0;
  int numScsIp = meSCS != nullptr ? meSCS->numIntPoints_ : 0;
  int numScvIp = meSCV != nullptr ? meSCV->numIntPoints_ : 0;
  int numFemIp = meFEM != nullptr ? meFEM->numIntPoints_ : 0;

  create_needed_master_element_views(team, dataNeeded, nDim, nodesPerFace, nodesPerElem, numFaceIp, numScsIp, numScvIp, numFemIp);
#endif
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
ScratchViews<T,TEAMHANDLETYPE,SHMEM>::ScratchViews(const TEAMHANDLETYPE& team,
             unsigned nDim,
             int nodalGatherSize,
             const ElemDataRequestsGPU& dataNeeded)
 : fieldViews(team, dataNeeded.get_total_num_fields(), count_needed_field_views(dataNeeded))
{
  num_bytes_required = create_needed_field_views<T,SHMEM>(team, dataNeeded, nodalGatherSize, fieldViews) * sizeof(T);

#ifndef KOKKOS_ENABLE_CUDA
  /* master elements are allowed to be null if they are not required */
  MasterElement *meFC = dataNeeded.get_cvfem_face_me();
  MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
  MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
  MasterElement *meFEM = dataNeeded.get_fem_volume_me();

  int nodesPerFace = meFC != nullptr ? meFC->nodesPerElement_ : 0;
  int nodesPerElem = meSCS != nullptr
          ? meSCS->nodesPerElement_ : meSCV != nullptr
          ? meSCV->nodesPerElement_ : meFEM != nullptr
          ? meFEM->nodesPerElement_ : 0;
  int numFaceIp= meFC  != nullptr ? meFC->numIntPoints_  : 0;
  int numScsIp = meSCS != nullptr ? meSCS->numIntPoints_ : 0;
  int numScvIp = meSCV != nullptr ? meSCV->numIntPoints_ : 0;
  int numFemIp = meFEM != nullptr ? meFEM->numIntPoints_ : 0;

  create_needed_master_element_views(team, dataNeeded, nDim, nodesPerFace, nodesPerElem, numFaceIp, numScsIp, numScvIp, numFemIp);
#endif
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
ScratchViews<T,TEAMHANDLETYPE,SHMEM>::ScratchViews(const TEAMHANDLETYPE& team,
             unsigned nDim,
             const ScratchMeInfo &meInfo,
             const ElemDataRequestsNGP& dataNeeded)
 : fieldViews(team, dataNeeded.get_total_num_fields(), count_needed_field_views(dataNeeded))
{
  num_bytes_required = create_needed_field_views<T,SHMEM>(team, dataNeeded, meInfo.nodalGatherSize_, fieldViews) * sizeof(T);
#ifndef KOKKOS_ENABLE_CUDA
  create_needed_master_element_views(team, dataNeeded, nDim, meInfo.nodesPerFace_, meInfo.nodesPerElement_, meInfo.numFaceIp_, meInfo.numScsIp_, meInfo.numScvIp_, meInfo.numFemIp_);
#endif
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
ScratchViews<T,TEAMHANDLETYPE,SHMEM>::ScratchViews(const TEAMHANDLETYPE& team,
             unsigned nDim,
             const ScratchMeInfo &meInfo,
             const ElemDataRequestsGPU& dataNeeded)
 : fieldViews(team, dataNeeded.get_total_num_fields(), count_needed_field_views(dataNeeded))
{
  num_bytes_required = create_needed_field_views<T,SHMEM>(team, dataNeeded, meInfo.nodalGatherSize_, fieldViews) * sizeof(T);
#ifndef KOKKOS_ENABLE_CUDA
  create_needed_master_element_views(team, dataNeeded, nDim, meInfo.nodesPerFace_, meInfo.nodesPerElement_, meInfo.numFaceIp_, meInfo.numScsIp_, meInfo.numScvIp_, meInfo.numFemIp_);
#endif
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
void ScratchViews<T,TEAMHANDLETYPE,SHMEM>::create_needed_master_element_views(const TEAMHANDLETYPE& team,
                                        const ElemDataRequestsNGP& dataNeeded,
                                        int nDim, int nodesPerFace, int nodesPerElem,
                                        int numFaceIp, int numScsIp, int numScvIp, int numFemIp)
{
  int numScalars = 0;

  const ElemDataRequestsNGP::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();

  for(unsigned i=0; i<coordsTypes.size(); ++i) {
    hasCoordField[coordsTypes(i)] = true;
    numScalars += meViews[coordsTypes(i)].create_master_element_views(
      team, dataNeeded.get_data_enums(coordsTypes(i)),
      nDim, nodesPerFace, nodesPerElem, numFaceIp, numScsIp, numScvIp, numFemIp);
  }

  num_bytes_required += numScalars * sizeof(T);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
void ScratchViews<T,TEAMHANDLETYPE,SHMEM>::create_needed_master_element_views(const TEAMHANDLETYPE& team,
                                        const ElemDataRequestsGPU& dataNeeded,
                                        int nDim, int nodesPerFace, int nodesPerElem,
                                        int numFaceIp, int numScsIp, int numScvIp, int numFemIp)
{
  int numScalars = 0;

//going to have to fix this for GPU !!!
#ifndef KOKKOS_ENABLE_CUDA
  const ElemDataRequestsGPU::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();

  for(unsigned i=0; i<coordsTypes.size(); ++i) {
    hasCoordField[coordsTypes(i)] = true;
    numScalars += meViews[coordsTypes(i)].create_master_element_views(
      team, dataNeeded.get_data_enums(coordsTypes(i)),
      nDim, nodesPerFace, nodesPerElem, numFaceIp, numScsIp, numScvIp, numFemIp);
  }
#endif

  num_bytes_required += numScalars * sizeof(T);
}

int get_num_scalars_pre_req_data(const ElemDataRequestsNGP& dataNeededBySuppAlgs, int nDim);
int get_num_scalars_pre_req_data(const ElemDataRequestsNGP& dataNeededBySuppAlgs, int nDim, const ScratchMeInfo &meInfo);
int get_num_scalars_pre_req_data(const ElemDataRequestsGPU& dataNeededBySuppAlgs, int nDim);
int get_num_scalars_pre_req_data(const ElemDataRequestsGPU& dataNeededBySuppAlgs, int nDim, const ScratchMeInfo &meInfo);

void fill_pre_req_data(const ElemDataRequestsNGP& dataNeeded,
                       const stk::mesh::BulkData& bulkData,
                       stk::mesh::Entity elem,
                       ScratchViews<double,TeamHandleType,HostShmem>& prereqData,
                       bool fillMEViews = true);

KOKKOS_FUNCTION
void fill_pre_req_data(const ElemDataRequestsGPU& dataNeeded,
                       const ngp::Mesh& ngpMesh,
                       stk::mesh::EntityRank entityRank,
                       stk::mesh::Entity elem,
                       ScratchViews<double,DeviceTeamHandleType,DeviceShmem>& prereqData,
                       bool fillMEViews = true);

template<typename ELEMDATAREQUESTSTYPE,typename SCRATCHVIEWSTYPE>
void fill_master_element_views(ELEMDATAREQUESTSTYPE& dataNeeded,
                               SCRATCHVIEWSTYPE& prereqData,
                               int faceOrdinal = 0)
{
#ifndef KOKKOS_ENABLE_CUDA
    MasterElement *meFC  = dataNeeded.get_cvfem_face_me();
    MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
    MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
    MasterElement *meFEM = dataNeeded.get_fem_volume_me();

    const ElemDataRequestsNGP::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();
    const ElemDataRequestsNGP::FieldView& coordsFields = dataNeeded.get_coordinates_fields();
    for(unsigned i=0; i<coordsTypes.size(); ++i) {
      auto cType = coordsTypes(i);
      const stk::mesh::FieldBase* coordField = coordsFields(i);

      const ElemDataRequestsNGP::DataEnumView& dataEnums = dataNeeded.get_data_enums(cType);
      auto* coordsView = &prereqData.get_scratch_view_2D(*coordField);
      auto& meData = prereqData.get_me_views(cType);

      meData.fill_master_element_views_new_me(dataEnums, coordsView, meFC, meSCS, meSCV, meFEM, faceOrdinal);
    }
#endif
}


template<typename T, typename ELEMDATAREQUESTSTYPE>
int get_num_bytes_pre_req_data(const ELEMDATAREQUESTSTYPE& dataNeededBySuppAlgs, int nDim)
{
  return sizeof(T) * get_num_scalars_pre_req_data(dataNeededBySuppAlgs, nDim);
}
template<typename T, typename ELEMDATAREQUESTSTYPE>
int get_num_bytes_pre_req_data(const ELEMDATAREQUESTSTYPE& dataNeededBySuppAlgs, int nDim, const ScratchMeInfo &meInfo)
{
  return sizeof(T) * get_num_scalars_pre_req_data(dataNeededBySuppAlgs, nDim, meInfo);
}

template<typename ELEMDATAREQUESTSTYPE>
inline
int calculate_shared_mem_bytes_per_thread(int lhsSize, int rhsSize, int scratchIdsSize, int nDim,
                                          const ELEMDATAREQUESTSTYPE& dataNeededByKernels)
{
    int bytes_per_thread = (rhsSize + lhsSize)*sizeof(double) + (2*scratchIdsSize)*sizeof(int) +
                         + get_num_bytes_pre_req_data<double>(dataNeededByKernels, nDim)
                         + MultiDimViews<double>::bytes_needed(dataNeededByKernels.get_total_num_fields(),
                                                 count_needed_field_views(dataNeededByKernels));

    bytes_per_thread *= 2*simdLen;
    return bytes_per_thread;
}

template<typename ELEMDATAREQUESTSTYPE>
inline
int calculate_shared_mem_bytes_per_thread(int lhsSize, int rhsSize, int scratchIdsSize, int nDim,
                                      const ELEMDATAREQUESTSTYPE& faceDataNeeded,
                                      const ELEMDATAREQUESTSTYPE& elemDataNeeded,
                                      const sierra::nalu::ScratchMeInfo &meInfo)
{
    int bytes_per_thread = (rhsSize + lhsSize)*sizeof(double) + (2*scratchIdsSize)*sizeof(int)
                         + sierra::nalu::get_num_bytes_pre_req_data<double>(faceDataNeeded, nDim)
                         + sierra::nalu::get_num_bytes_pre_req_data<double>(elemDataNeeded, nDim, meInfo)
                         + MultiDimViews<double>::bytes_needed(faceDataNeeded.get_total_num_fields(),
                                                 count_needed_field_views(faceDataNeeded))
                         + MultiDimViews<double>::bytes_needed(elemDataNeeded.get_total_num_fields(),
                                                 count_needed_field_views(elemDataNeeded));

    bytes_per_thread *= 2*simdLen;
    return bytes_per_thread;
}

template<typename T>
void set_zero(T* values, unsigned length)
{
    for(unsigned i=0; i<length; ++i) {
        values[i] = 0;
    }
}

} // namespace nalu
} // namespace Sierra

#endif
