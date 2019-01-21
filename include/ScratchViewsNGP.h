/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScratchViewsNGP_h
#define ScratchViewsNGP_h

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>

#include <MultiDimViews.h>
#include <ElemDataRequestsGPU.h>
#include <master_element/MasterElement.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>
#include <ScratchViews.h>

#include <set>
#include <type_traits>
#include <string>

namespace sierra{
namespace nalu{

#if 0
struct ScratchMeInfo {
  int nodalGatherSize_;
  int nodesPerFace_;
  int nodesPerElement_;
  int numFaceIp_;
  int numScsIp_;
  int numScvIp_;
  int numFemIp_;
};

template<typename T>
class MasterElementViews
{
public:
  typedef T value_type;

  MasterElementViews() = default;
  virtual ~MasterElementViews() = default;

  int create_master_element_views(
    const TEAMHANDLETYPE& team,
    const ElemDataRequestsGPU::DataEnumView& dataEnums,
    int nDim, int nodesPerFace, int nodesPerElem,
    int numFaceIp, int numScsIp, int numScvIp, int numFemIp);

  void fill_master_element_views(
    const ElemDataRequestsGPU::DataEnumView& dataEnums,
    SharedMemView<double**>* coordsView,
    MasterElement* meFC,
    MasterElement* meSCS,
    MasterElement* meSCV,
    MasterElement* meFEM,
    int faceOrdinal = 0);

  void fill_master_element_views_new_me(
    const ElemDataRequestsGPU::DataEnumView& dataEnums,
    SharedMemView<DoubleType**>* coordsView,
    MasterElement* meFC,
    MasterElement* meSCS,
    MasterElement* meSCV,
    MasterElement* meFEM,
    int faceOrdinal = 0);

  SharedMemView<T**,SHMEM> fc_areav;
  SharedMemView<T**,SHMEM> scs_areav;
  SharedMemView<T***,SHMEM> dndx_fc_scs;
  SharedMemView<T***,SHMEM> dndx_shifted_fc_scs;
  SharedMemView<T***,SHMEM> dndx;
  SharedMemView<T***,SHMEM> dndx_shifted;
  SharedMemView<T***,SHMEM> dndx_scv;
  SharedMemView<T***,SHMEM> dndx_scv_shifted;
  SharedMemView<T***,SHMEM> dndx_fem;
  SharedMemView<T***,SHMEM> deriv_fc_scs;
  SharedMemView<T***,SHMEM> deriv;
  SharedMemView<T***,SHMEM> deriv_scv;
  SharedMemView<T***,SHMEM> deriv_fem;
  SharedMemView<T*,SHMEM> det_j_fc_scs;
  SharedMemView<T*,SHMEM> det_j;
  SharedMemView<T*,SHMEM> det_j_scv;
  SharedMemView<T*,SHMEM> det_j_fem;
  SharedMemView<T*,SHMEM> scv_volume;
  SharedMemView<T***,SHMEM> gijUpper;
  SharedMemView<T***,SHMEM> gijLower;
};

#endif

template<typename T, typename TEAMHANDLETYPE=DeviceTeamHandleType, typename SHMEM=DeviceShmem>
class ScratchViewsNGP
{
public:
  typedef T value_type;

  KOKKOS_FUNCTION
  ScratchViewsNGP(const TEAMHANDLETYPE& team,
               unsigned nDim,
               int nodesPerEntity,
               const ElemDataRequestsGPU& dataNeeded);

  KOKKOS_FUNCTION
  ScratchViewsNGP(const TEAMHANDLETYPE& team,
               unsigned nDim,
               const ScratchMeInfo &meInfo,
               const ElemDataRequestsGPU& dataNeeded);

  KOKKOS_FUNCTION
  virtual ~ScratchViewsNGP() {
  }

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T*,SHMEM>& get_scratch_view_1D(const unsigned fieldOrdinal);

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T**,SHMEM>& get_scratch_view_2D(const unsigned fieldOrdinal);

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T***,SHMEM>& get_scratch_view_3D(const unsigned fieldOrdinal);

  KOKKOS_INLINE_FUNCTION
  SharedMemView<T****,SHMEM>& get_scratch_view_4D(const unsigned fieldOrdinal);

  KOKKOS_INLINE_FUNCTION
  MasterElementViews<T>& get_me_views(const COORDS_TYPES cType)
  {
    NGP_ThrowRequire(hasCoordField[cType] == true);
    return meViews[cType];
  }
  KOKKOS_INLINE_FUNCTION
  bool has_coord_field(const COORDS_TYPES cType) const { return hasCoordField[cType]; }

  KOKKOS_INLINE_FUNCTION
  int total_bytes() const { return num_bytes_required; }

  ngp::Mesh::ConnectedNodes elemNodes;

  KOKKOS_INLINE_FUNCTION       MultiDimViews<T,TEAMHANDLETYPE,SHMEM>& get_field_views()       { return fieldViews; }
  KOKKOS_INLINE_FUNCTION const MultiDimViews<T,TEAMHANDLETYPE,SHMEM>& get_field_views() const { return fieldViews; }

private:
  KOKKOS_FUNCTION
  void create_needed_field_views(const TEAMHANDLETYPE& team,
                                 const ElemDataRequestsGPU& dataNeeded,
                                 int nodesPerElem);

  KOKKOS_FUNCTION
  void create_needed_master_element_views(const TEAMHANDLETYPE& team,
                                          const ElemDataRequestsGPU& dataNeeded,
                                          int nDim, int nodesPerFace, int nodesPerElem,
                                          int numFaceIp, int numScsIp, int numScvIp, int numFemIp);

  MultiDimViews<T,TEAMHANDLETYPE,SHMEM> fieldViews;
  MasterElementViews<T> meViews[MAX_COORDS_TYPES];
  bool hasCoordField[MAX_COORDS_TYPES] = {false, false};
  int num_bytes_required{0};
};

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T*,SHMEM>& ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_1D(const unsigned fieldOrdinal)
{ 
//  ThrowAssertMsg(fieldOrdinal < fieldViews.ordinals.size() && fieldViews.ordinals(fieldOrdinal) < (int)fieldViews.views_1D.size(),
//    "ScratchViewsNGP ERROR, trying to get 1D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
  return fieldViews.get_scratch_view_1D(fieldOrdinal);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T**,SHMEM>& ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_2D(const unsigned fieldOrdinal)
{ 
//  ThrowAssertMsg(fieldOrdinal < fieldViews.ordinals.size() && fieldViews.ordinals(fieldOrdinal) < (int)fieldViews.views_2D.size(),
 //   "ScratchViewsNGP ERROR, trying to get 2D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
  return fieldViews.get_scratch_view_2D(fieldOrdinal);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T***,SHMEM>& ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_3D(const unsigned fieldOrdinal)
{
//  ThrowAssertMsg(fieldOrdinal < fieldViews.ordinals.size() && fieldViews.ordinals(fieldOrdinal) < (int)fieldViews.views_3D.size(),
//    "ScratchViewsNGP ERROR, trying to get 3D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
  return fieldViews.get_scratch_view_3D(fieldOrdinal);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
  KOKKOS_INLINE_FUNCTION
SharedMemView<T****,SHMEM>& ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::get_scratch_view_4D(const unsigned fieldOrdinal)
{
//  ThrowAssertMsg(fieldOrdinal < fieldViews.ordinals.size() && fieldViews.ordinals(fieldOrdinal) < (int)fieldViews.views_4D.size(),
//    "ScratchViewsNGP ERROR, trying to get 4D scratch-view for field "<<field.name()<<" which wasn't declared as pre-req field.");
  return fieldViews.get_scratch_view_4D(fieldOrdinal);
}

#if 0

template<typename T>
int MasterElementViews<T>::create_master_element_views(
  const TEAMHANDLETYPE& team,
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

template<typename T>
void MasterElementViews<T>::fill_master_element_views(
  const ElemDataRequestsGPU::DataEnumView& dataEnums,
  SharedMemView<double**>* coordsView,
  MasterElement* meFC,
  MasterElement* meSCS,
  MasterElement* meSCV,
  MasterElement* meFEM,
  int faceOrdinal)
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
  const ElemDataRequestsGPU::DataEnumView& dataEnums,
  SharedMemView<DoubleType**>* coordsView,
  MasterElement* meFC,
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

#endif

KOKKOS_INLINE_FUNCTION
NumNeededViews count_needed_field_views(const ElemDataRequestsGPU& dataNeeded)
{
  NumNeededViews numNeededViews = {0, 0, 0, 0};

  const ElemDataRequestsGPU::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned i=0; i<neededFields.size(); ++i) {
    FieldInfoNGP& fieldInfo = neededFields(i);
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field.get_rank();
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

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
KOKKOS_FUNCTION
ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::ScratchViewsNGP(const TEAMHANDLETYPE& team,
             unsigned nDim,
             int nodalGatherSize,
             const ElemDataRequestsGPU& dataNeeded)
 : fieldViews(team, dataNeeded.get_total_num_fields(), count_needed_field_views(dataNeeded))
{
  /* master elements are allowed to be null if they are not required */
//  MasterElement *meFC = dataNeeded.get_cvfem_face_me();
//  MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
//  MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
//  MasterElement *meFEM = dataNeeded.get_fem_volume_me();

//  int nodesPerFace = meFC != nullptr ? meFC->nodesPerElement_ : 0;
//  int nodesPerElem = meSCS != nullptr
//          ? meSCS->nodesPerElement_ : meSCV != nullptr
//          ? meSCV->nodesPerElement_ : meFEM != nullptr
//          ? meFEM->nodesPerElement_ : 0;
//  int numFaceIp= meFC  != nullptr ? meFC->numIntPoints_  : 0;
//  int numScsIp = meSCS != nullptr ? meSCS->numIntPoints_ : 0;
//  int numScvIp = meSCV != nullptr ? meSCV->numIntPoints_ : 0;
//  int numFemIp = meFEM != nullptr ? meFEM->numIntPoints_ : 0;

  create_needed_field_views(team, dataNeeded, nodalGatherSize);

//  create_needed_master_element_views(team, dataNeeded, nDim, nodesPerFace, nodesPerElem, numFaceIp, numScsIp, numScvIp, numFemIp);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
KOKKOS_FUNCTION
ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::ScratchViewsNGP(const TEAMHANDLETYPE& team,
             unsigned nDim,
             const ScratchMeInfo &meInfo,
             const ElemDataRequestsGPU& dataNeeded)
 : fieldViews(team, dataNeeded.get_total_num_fields(), count_needed_field_views(dataNeeded))
{
  create_needed_field_views(team, dataNeeded, meInfo.nodalGatherSize_);
  create_needed_master_element_views(team, dataNeeded, nDim, meInfo.nodesPerFace_, meInfo.nodesPerElement_, meInfo.numFaceIp_, meInfo.numScsIp_, meInfo.numScvIp_, meInfo.numFemIp_);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
KOKKOS_FUNCTION
void ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::create_needed_field_views(const TEAMHANDLETYPE& team,
                               const ElemDataRequestsGPU& dataNeeded,
                               int nodesPerEntity)
{
  int numScalars = 0;

  const ElemDataRequestsGPU::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned i=0; i<neededFields.size(); ++i) {
    const FieldInfoNGP& fieldInfo = neededFields(i);
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field.get_rank();
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    unsigned scalarsDim2 = fieldInfo.scalarsDim2;

    if (fieldEntityRank==stk::topology::EDGE_RANK ||
        fieldEntityRank==stk::topology::FACE_RANK ||
        fieldEntityRank==stk::topology::ELEM_RANK) {
      if (scalarsDim2 == 0) {
        fieldViews.add_1D_view(fieldInfo.field.get_ordinal(), get_shmem_view_1D<T,DeviceTeamHandleType,DeviceShmem>(team, scalarsDim1));
        numScalars += scalarsDim1;
      }
      else {
        fieldViews.add_2D_view(fieldInfo.field.get_ordinal(), get_shmem_view_2D<T,DeviceTeamHandleType,DeviceShmem>(team, scalarsDim1, scalarsDim2));
        numScalars += scalarsDim1 * scalarsDim2;
      }
    }
    else if (fieldEntityRank==stk::topology::NODE_RANK) {
      if (scalarsDim2 == 0) {
        if (scalarsDim1 == 1) {
          fieldViews.add_1D_view(fieldInfo.field.get_ordinal(), get_shmem_view_1D<T,DeviceTeamHandleType,DeviceShmem>(team, nodesPerEntity));
          numScalars += nodesPerEntity;
        }
        else {
          fieldViews.add_2D_view(fieldInfo.field.get_ordinal(), get_shmem_view_2D<T,DeviceTeamHandleType,DeviceShmem>(team, nodesPerEntity, scalarsDim1));
          numScalars += nodesPerEntity*scalarsDim1;
        }
      }
        else {
          fieldViews.add_3D_view(fieldInfo.field.get_ordinal(), get_shmem_view_3D<T,DeviceTeamHandleType,DeviceShmem>(team, nodesPerEntity, scalarsDim1, scalarsDim2));
            numScalars += nodesPerEntity*scalarsDim1*scalarsDim2;
      }
    }
    else {
      NGP_ThrowRequireMsg(
        false,
        "Unknown stk-rank in ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::create_needed_field_views");
    }
  }

  // Track total bytes required for field allocations
  num_bytes_required += numScalars * sizeof(T);
}

template<typename T,typename TEAMHANDLETYPE,typename SHMEM>
KOKKOS_FUNCTION
void ScratchViewsNGP<T,TEAMHANDLETYPE,SHMEM>::create_needed_master_element_views(const TEAMHANDLETYPE& team,
                                        const ElemDataRequestsGPU& dataNeeded,
                                        int nDim, int nodesPerFace, int nodesPerElem,
                                        int numFaceIp, int numScsIp, int numScvIp, int numFemIp)
{
  int numScalars = 0;

  const ElemDataRequestsGPU::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();

  for(unsigned i=0; i<coordsTypes.size(); ++i) {
    hasCoordField[coordsTypes(i)] = true;
//    numScalars += meViews[coordsTypes(i)].create_master_element_views(
//      team, dataNeeded.get_data_enums(coordsTypes(i)),
//      nDim, nodesPerFace, nodesPerElem, numFaceIp, numScsIp, numScvIp, numFemIp);
  }

  num_bytes_required += numScalars * sizeof(T);
}

int get_num_scalars_pre_req_data(ElemDataRequestsGPU& dataNeededBySuppAlgs, int nDim);
int get_num_scalars_pre_req_data(ElemDataRequestsGPU& dataNeededBySuppAlgs, int nDim, const ScratchMeInfo &meInfo);

KOKKOS_FUNCTION
void fill_pre_req_data(const ElemDataRequestsGPU& dataNeeded,
                       const ngp::Mesh& ngpMesh,
                       stk::mesh::EntityRank entityRank,
                       stk::mesh::Entity elem,
                       ScratchViewsNGP<double,DeviceTeamHandleType,DeviceShmem>& prereqData,
                       bool fillMEViews = true);

void fill_master_element_views(ElemDataRequestsGPU& dataNeeded,
                               ScratchViewsNGP<DoubleType>& prereqData,
                               int faceOrdinal = 0);
template<typename T = double>
int get_num_bytes_pre_req_data(ElemDataRequestsGPU& dataNeededBySuppAlgs, int nDim)
{
  return sizeof(T) * get_num_scalars_pre_req_data(dataNeededBySuppAlgs, nDim);
}
template<typename T = double>
int get_num_bytes_pre_req_data(ElemDataRequestsGPU& dataNeededBySuppAlgs, int nDim, const ScratchMeInfo &meInfo)
{
  return sizeof(T) * get_num_scalars_pre_req_data(dataNeededBySuppAlgs, nDim, meInfo);
}

inline
int calculate_shared_mem_bytes_per_thread(int lhsSize, int rhsSize, int scratchIdsSize, int nDim,
                                      ElemDataRequestsGPU& dataNeededByKernels)
{
    int bytes_per_thread = (rhsSize + lhsSize)*sizeof(double) + (2*scratchIdsSize)*sizeof(int)
                         + get_num_bytes_pre_req_data<double>(dataNeededByKernels, nDim)
                         + MultiDimViews<double>::bytes_needed(dataNeededByKernels.get_total_num_fields(),
                                                 count_needed_field_views(dataNeededByKernels));
    bytes_per_thread *= 2*simdLen;
    return bytes_per_thread;
}

inline
int calculate_shared_mem_bytes_per_thread(int lhsSize, int rhsSize, int scratchIdsSize, int nDim,
                                      sierra::nalu::ElemDataRequestsGPU& faceDataNeeded,
                                      sierra::nalu::ElemDataRequestsGPU& elemDataNeeded,
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

#if 0
template<typename T>
void set_zero(T* values, unsigned length)
{
    for(unsigned i=0; i<length; ++i) {
        values[i] = 0;
    }
}
#endif
} // namespace nalu
} // namespace Sierra

#endif
