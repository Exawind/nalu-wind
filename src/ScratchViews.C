/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <ScratchViews.h>

#include <NaluEnv.h>

namespace sierra {
namespace nalu {

template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void gather_elem_node_field(const NGPDoubleFieldType& field,
                            const ngp::Mesh& ngpMesh,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            ViewType& shmemView)
{
  for(unsigned i=0; i<elemNodes.size(); ++i) {
    shmemView[i] = field.get(ngpMesh, elemNodes[i], 0); 
  }
}

template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void gather_elem_node_tensor_field(const NGPDoubleFieldType& field,
                            const ngp::Mesh& ngpMesh,
                            int numNodes,
                            int tensorDim1,
                            int tensorDim2,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            ViewType& shmemView)
{
  NGP_ThrowRequireMsg(
      numNodes==(int)elemNodes.size(),
      "gather_elem_node_tensor_field, numNodes = mismatch with elemNodes.size()"  );   
  for(int i=0; i<numNodes; ++i) {
    const double* dataPtr = static_cast<const double*>(&field.get(ngpMesh, elemNodes[i], 0));
    unsigned counter = 0;
    for(int d1=0; d1<tensorDim1; ++d1) { 
      for(int d2=0; d2<tensorDim2; ++d2) {
        shmemView(i,d1,d2) = dataPtr[counter++];
      }   
    }   
  }
}

template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void gather_elem_tensor_field(const NGPDoubleFieldType& field,
                              stk::mesh::FastMeshIndex elem,
                              int tensorDim1,
                              int tensorDim2,
                              ViewType& shmemView)
{
  const double* dataPtr = static_cast<const double*>(&field.get(elem, 0));
  unsigned counter = 0;
  for(int d1=0; d1<tensorDim1; ++d1) { 
    for(int d2=0; d2<tensorDim2; ++d2) {
      shmemView(d1,d2) = dataPtr[counter++];
    }   
  }
}

template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void gather_elem_node_field_3D(const NGPDoubleFieldType& field,
                               const ngp::Mesh& ngpMesh,
                               const ngp::Mesh::ConnectedNodes& elemNodes,
                               ViewType& shmemView)
{
  for(unsigned i=0; i<elemNodes.size(); ++i) {
    const double* dataPtr = &field.get(ngpMesh, elemNodes[i], 0); 
    shmemView(i,0) = dataPtr[0];
    shmemView(i,1) = dataPtr[1];
    shmemView(i,2) = dataPtr[2];
  }
}

template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void gather_elem_node_field(const NGPDoubleFieldType& field,
                            const ngp::Mesh& ngpMesh,
                            int scalarsPerNode,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            ViewType& shmemView)
{
  for(unsigned i=0; i<elemNodes.size(); ++i) {
    const double* dataPtr = &field.get(ngpMesh, elemNodes[i], 0);
    for(int d=0; d<scalarsPerNode; ++d) {
      shmemView(i,d) = dataPtr[d];
    }
  }
}

inline
void gather_elem_node_field(const stk::mesh::FieldBase& field,
                            int numNodes,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            SharedMemView<double*>& shmemView)
{
  for(int i=0; i<numNodes; ++i) {
    shmemView[i] = *static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
  }
}

inline
void gather_elem_node_tensor_field(const stk::mesh::FieldBase& field,
                            int numNodes,
                            int tensorDim1,
                            int tensorDim2,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            SharedMemView<double***>& shmemView)
{
  for(int i=0; i<numNodes; ++i) {
    const double* dataPtr = static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
    unsigned counter = 0;
    for(int d1=0; d1<tensorDim1; ++d1) { 
      for(int d2=0; d2<tensorDim2; ++d2) {
        shmemView(i,d1,d2) = dataPtr[counter++];
      }
    }
  }
}

inline
void gather_elem_tensor_field(const stk::mesh::FieldBase& field,
                              stk::mesh::Entity elem,
                              int tensorDim1,
                              int tensorDim2,
                              SharedMemView<double**>& shmemView)
{
  const double* dataPtr = static_cast<const double*>(stk::mesh::field_data(field, elem));
  unsigned counter = 0;
  for(int d1=0; d1<tensorDim1; ++d1) { 
    for(int d2=0; d2<tensorDim2; ++d2) {
      shmemView(d1,d2) = dataPtr[counter++];
    }
  }
}

inline
void gather_elem_node_field_3D(const stk::mesh::FieldBase& field,
                               int numNodes,
                               const ngp::Mesh::ConnectedNodes& elemNodes,
                               SharedMemView<double**>& shmemView)
{
  for(int i=0; i<numNodes; ++i) {
    const double* dataPtr = static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
    shmemView(i,0) = dataPtr[0];
    shmemView(i,1) = dataPtr[1];
    shmemView(i,2) = dataPtr[2];
  }
}

inline
void gather_elem_node_field(const stk::mesh::FieldBase& field,
                            int numNodes,
                            int scalarsPerNode,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            SharedMemView<double**>& shmemView)
{
  for(int i=0; i<numNodes; ++i) {
    const double* dataPtr = static_cast<const double*>(stk::mesh::field_data(field, elemNodes[i]));
    for(int d=0; d<scalarsPerNode; ++d) {
      shmemView(i,d) = dataPtr[d];
    }
  }
}

int get_num_scalars_pre_req_data(const ElemDataRequestsGPU& dataNeeded, int nDim)
{
  /* master elements are allowed to be null if they are not required */
  MasterElement *meFC  = dataNeeded.get_cvfem_face_me();
  MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
  MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
  MasterElement *meFEM = dataNeeded.get_fem_volume_me();

  const bool faceDataNeeded = meFC != nullptr
    && meSCS == nullptr && meSCV == nullptr && meFEM == nullptr;
  const bool elemDataNeeded = meFC == nullptr
    && (meSCS != nullptr || meSCV != nullptr || meFEM != nullptr);

  NGP_ThrowRequireMsg(faceDataNeeded != elemDataNeeded,
    "An algorithm has been registered with conflicting face/element data requests");

  const int nodesPerEntity = meSCS != nullptr ? meSCS->nodesPerElement_
    : meSCV != nullptr ? meSCV->nodesPerElement_
    : meFEM != nullptr ? meFEM->nodesPerElement_
    : meFC  != nullptr ? meFC->nodesPerElement_
    : 0;

  int numScalars = 0;

  const ElemDataRequestsGPU::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned f=0; f<neededFields.size(); ++f) {
    const FieldInfoNGP& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field.get_rank();
    unsigned scalarsPerEntity = fieldInfo.scalarsDim1;
    unsigned entitiesPerElem = fieldEntityRank==stk::topology::NODE_RANK ? nodesPerEntity : 1;

    // Catch errors if user requests nodal field but has not registered any
    // MasterElement we need to get nodesPerEntity
    NGP_ThrowRequire(entitiesPerElem > 0);
    if (fieldInfo.scalarsDim2 > 1) {
      scalarsPerEntity *= fieldInfo.scalarsDim2;
    }
    numScalars += entitiesPerElem*scalarsPerEntity;
  }

  const int numFaceIp = meFC != nullptr ? meFC->numIntPoints_ : 0;
  const int numScsIp = meSCS != nullptr ? meSCS->numIntPoints_ : 0;
  const int numScvIp = meSCV != nullptr ? meSCV->numIntPoints_ : 0;
  const int numFemIp = meFEM != nullptr ? meFEM->numIntPoints_ : 0;

  const ElemDataRequestsGPU::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();
  for(unsigned i=0; i<coordsTypes.size(); ++i) {
    auto cType = coordsTypes(i);
    const ElemDataRequestsGPU::DataEnumView& dataEnums = dataNeeded.get_data_enums(cType);
    int dndxLength = 0, dndxLengthFC = 0, gUpperLength = 0, gLowerLength = 0;

    // Updated logic for data sharing of deriv and det_j
    bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
    bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;

    for(unsigned d=0; d<dataEnums.size(); ++d) {
      ELEM_DATA_NEEDED data = dataEnums(d);
      switch(data)
      {
        case FC_AREAV:
          numScalars += nDim * numFaceIp;
          break;
        case SCS_AREAV:
          numScalars += nDim * numScsIp;
          break;
        case SCS_FACE_GRAD_OP:
        case SCS_SHIFTED_FACE_GRAD_OP:
          dndxLengthFC = nodesPerEntity*numFaceIp*nDim;
          needDerivFC = true;
          needDetjFC = true;
          numScalars += dndxLengthFC;
          break;
        case SCS_GRAD_OP:
        case SCS_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numScsIp*nDim;
          needDeriv = true;
          needDetj = true;
          numScalars += dndxLength;
          break;
        case SCV_VOLUME:
          numScalars += numScvIp;
          break;
        case SCV_GRAD_OP:
          dndxLength = nodesPerEntity*numScvIp*nDim;
          needDerivScv = true;
          needDetjScv = true;
          numScalars += dndxLength;
          break;
        case SCS_GIJ:
          gUpperLength = nDim*nDim*numScsIp;
          gLowerLength = nDim*nDim*numScsIp;
          needDeriv = true;
          numScalars += (gUpperLength + gLowerLength );
          break;
        case FEM_GRAD_OP:
        case FEM_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numFemIp*nDim;
          needDerivFem = true;
          needDetjFem = true;
          numScalars += dndxLength;
          break;
        default: break;
      }
    }

    if (needDerivFC)
      numScalars += nodesPerEntity*numFaceIp*nDim;

    if (needDeriv)
      numScalars += nodesPerEntity*numScsIp*nDim;

    if (needDerivScv)
      numScalars += nodesPerEntity*numScvIp*nDim;

    if (needDerivFem)
      numScalars += nodesPerEntity*numFemIp*nDim;

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

int get_num_scalars_pre_req_data(const ElemDataRequestsGPU& dataNeeded, int nDim, const ScratchMeInfo &meInfo)
{
  const int nodesPerEntity = meInfo.nodalGatherSize_;
  const int numFaceIp = meInfo.numFaceIp_;
  const int numScsIp = meInfo.numScsIp_;
  const int numScvIp = meInfo.numScvIp_;
  const int numFemIp = meInfo.numFemIp_;
  int numScalars = 0;

  const ElemDataRequestsGPU::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned f=0; f<neededFields.size(); ++f) {
    const FieldInfoNGP& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsPerEntity = fieldInfo.scalarsDim1;
    unsigned entitiesPerElem = fieldEntityRank==stk::topology::NODE_RANK ? nodesPerEntity : 1;

    // Catch errors if user requests nodal field but has not registered any
    // MasterElement we need to get nodesPerEntity
    NGP_ThrowRequire(entitiesPerElem > 0);
    if (fieldInfo.scalarsDim2 > 1) {
      scalarsPerEntity *= fieldInfo.scalarsDim2;
    }
    numScalars += entitiesPerElem*scalarsPerEntity;
  }

  const ElemDataRequestsGPU::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();
  for(unsigned i=0; i<coordsTypes.size(); ++i) {
    auto cType = coordsTypes(i);
    int dndxLength = 0, dndxLengthFC = 0, gUpperLength = 0, gLowerLength = 0;

    // Updated logic for data sharing of deriv and det_j
    bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
    bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;

    const ElemDataRequestsGPU::DataEnumView& dataEnums = dataNeeded.get_data_enums(cType);
    for(unsigned d=0; d<dataEnums.size(); ++d) {
      ELEM_DATA_NEEDED data = dataEnums(d);
      switch(data)
      {
        case FC_AREAV:
          numScalars += nDim * numFaceIp;
          break;
        case SCS_AREAV:
          numScalars += nDim * numScsIp;
          break;
        case SCS_FACE_GRAD_OP:
        case SCS_SHIFTED_FACE_GRAD_OP:
          dndxLengthFC = nodesPerEntity*numFaceIp*nDim;
          needDerivFC = true;
          needDetjFC = true;
          numScalars += dndxLengthFC;
          break;
        case SCS_GRAD_OP:
        case SCS_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numScsIp*nDim;
          needDeriv = true;
          needDetj = true;
          numScalars += dndxLength;
          break;
        case SCV_VOLUME:
          numScalars += numScvIp;
          break;
        case SCV_GRAD_OP:
          dndxLength = nodesPerEntity*numScvIp*nDim;
          needDerivScv = true;
          needDetjScv = true;
          numScalars += dndxLength;
          break;
        case SCS_GIJ:
          gUpperLength = nDim*nDim*numScsIp;
          gLowerLength = nDim*nDim*numScsIp;
          needDeriv = true;
          numScalars += (gUpperLength + gLowerLength );
          break;
        case FEM_GRAD_OP:
        case FEM_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numFemIp*nDim;
          needDerivFem = true;
          needDetjFem = true;
          numScalars += dndxLength;
          break;
        default: break;
      }
    }

    if (needDerivFC)
      numScalars += nodesPerEntity*numFaceIp*nDim;

    if (needDeriv)
      numScalars += nodesPerEntity*numScsIp*nDim;

    if (needDerivScv)
      numScalars += nodesPerEntity*numScvIp*nDim;

    if (needDerivFem)
      numScalars += nodesPerEntity*numFemIp*nDim;

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

int get_num_scalars_pre_req_data(const ElemDataRequestsNGP& dataNeeded, int nDim)
{
  /* master elements are allowed to be null if they are not required */
  MasterElement *meFC  = dataNeeded.get_cvfem_face_me();
  MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
  MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
  MasterElement *meFEM = dataNeeded.get_fem_volume_me();
  
  const bool faceDataNeeded = meFC != nullptr
    && meSCS == nullptr && meSCV == nullptr && meFEM == nullptr;
  const bool elemDataNeeded = meFC == nullptr
    && (meSCS != nullptr || meSCV != nullptr || meFEM != nullptr);

  ThrowRequireMsg(faceDataNeeded != elemDataNeeded,
    "An algorithm has been registered with conflicting face/element data requests");

  const int nodesPerEntity = meSCS != nullptr ? meSCS->nodesPerElement_
    : meSCV != nullptr ? meSCV->nodesPerElement_ 
    : meFEM != nullptr ? meFEM->nodesPerElement_
    : meFC  != nullptr ? meFC->nodesPerElement_
    : 0;

  const int numFaceIp = meFC != nullptr ? meFC->numIntPoints_ : 0;
  const int numScsIp = meSCS != nullptr ? meSCS->numIntPoints_ : 0;
  const int numScvIp = meSCV != nullptr ? meSCV->numIntPoints_ : 0;
  const int numFemIp = meFEM != nullptr ? meFEM->numIntPoints_ : 0;
  int numScalars = 0;
  
  const ElemDataRequestsNGP::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned f=0; f<neededFields.size(); ++f) {
    const FieldInfo& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsPerEntity = fieldInfo.scalarsDim1;
    unsigned entitiesPerElem = fieldEntityRank==stk::topology::NODE_RANK ? nodesPerEntity : 1;

    // Catch errors if user requests nodal field but has not registered any
    // MasterElement we need to get nodesPerEntity
    ThrowRequire(entitiesPerElem > 0);
    if (fieldInfo.scalarsDim2 > 1) {
      scalarsPerEntity *= fieldInfo.scalarsDim2;
    }
    numScalars += entitiesPerElem*scalarsPerEntity;
  }

  const ElemDataRequestsNGP::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();
  for(unsigned i=0; i<coordsTypes.size(); ++i) {
    auto cType = coordsTypes(i);
    const ElemDataRequestsNGP::DataEnumView& dataEnums = dataNeeded.get_data_enums(cType);
    int dndxLength = 0, dndxLengthFC = 0, gUpperLength = 0, gLowerLength = 0, metricLength = 0;

    // Updated logic for data sharing of deriv and det_j
    bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
    bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;

    for(unsigned d=0; d<dataEnums.size(); ++d) {
      ELEM_DATA_NEEDED data = dataEnums(d);
      switch(data)
      {
        case FC_AREAV:
          numScalars += nDim * numFaceIp;
          break;
        case SCS_AREAV:
          numScalars += nDim * numScsIp;
          break;
        case SCS_FACE_GRAD_OP:
        case SCS_SHIFTED_FACE_GRAD_OP:
          dndxLengthFC = nodesPerEntity*numFaceIp*nDim;
          needDerivFC = true;
          needDetjFC = true;
          numScalars += dndxLengthFC;
          break;
        case SCS_GRAD_OP:
        case SCS_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numScsIp*nDim;
          needDeriv = true;
          needDetj = true;
          numScalars += dndxLength;
          break;
        case SCV_VOLUME:
          numScalars += numScvIp;
          break;
        case SCV_GRAD_OP:
          dndxLength = nodesPerEntity*numScvIp*nDim;
          needDerivScv = true;
          needDetjScv = true;
          numScalars += dndxLength;
          break;
        case SCS_GIJ:
          gUpperLength = nDim*nDim*numScsIp;
          gLowerLength = nDim*nDim*numScsIp;
          needDeriv = true;
          numScalars += (gUpperLength + gLowerLength );
          break;
        case SCV_MIJ:
          metricLength = nDim*nDim*numScvIp;
          needDeriv = true;
          numScalars += metricLength;
          break;
        case FEM_GRAD_OP:
        case FEM_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numFemIp*nDim;
          needDerivFem = true;
          needDetjFem = true;
          numScalars += dndxLength;
          break;
        default: break;
      }
    }

    if (needDerivFC)
      numScalars += nodesPerEntity*numFaceIp*nDim;

    if (needDeriv)
      numScalars += nodesPerEntity*numScsIp*nDim;

    if (needDerivScv)
      numScalars += nodesPerEntity*numScvIp*nDim;
    
    if (needDerivFem)
      numScalars += nodesPerEntity*numFemIp*nDim;
    
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

int get_num_scalars_pre_req_data(const ElemDataRequestsNGP& dataNeeded, int nDim, const ScratchMeInfo &meInfo)
{
  const int nodesPerEntity = meInfo.nodalGatherSize_;
  const int numFaceIp = meInfo.numFaceIp_;
  const int numScsIp = meInfo.numScsIp_;
  const int numScvIp = meInfo.numScvIp_;
  const int numFemIp = meInfo.numFemIp_;
  int numScalars = 0;

  const ElemDataRequestsNGP::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned f=0; f<neededFields.size(); ++f) {
    const FieldInfo& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsPerEntity = fieldInfo.scalarsDim1;
    unsigned entitiesPerElem = fieldEntityRank==stk::topology::NODE_RANK ? nodesPerEntity : 1;

    // Catch errors if user requests nodal field but has not registered any
    // MasterElement we need to get nodesPerEntity
    ThrowRequire(entitiesPerElem > 0);
    if (fieldInfo.scalarsDim2 > 1) {
      scalarsPerEntity *= fieldInfo.scalarsDim2;
    }
    numScalars += entitiesPerElem*scalarsPerEntity;
  }

  const ElemDataRequestsNGP::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();
  for(unsigned i=0; i<coordsTypes.size(); ++i) {
    auto cType = coordsTypes(i);
    int dndxLength = 0, dndxLengthFC = 0, gUpperLength = 0, gLowerLength = 0;

    // Updated logic for data sharing of deriv and det_j
    bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
    bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;

    const ElemDataRequestsNGP::DataEnumView& dataEnums = dataNeeded.get_data_enums(cType);
    for(unsigned d=0; d<dataEnums.size(); ++d) {
      ELEM_DATA_NEEDED data = dataEnums(d);
      switch(data)
      {
        case FC_AREAV:
          numScalars += nDim * numFaceIp;
          break;
        case SCS_AREAV:
          numScalars += nDim * numScsIp;
          break;
        case SCS_FACE_GRAD_OP:
        case SCS_SHIFTED_FACE_GRAD_OP:
          dndxLengthFC = nodesPerEntity*numFaceIp*nDim;
          needDerivFC = true;
          needDetjFC = true;
          numScalars += dndxLengthFC;
          break;
        case SCS_GRAD_OP:
        case SCS_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numScsIp*nDim;
          needDeriv = true;
          needDetj = true;
          numScalars += dndxLength;
          break;
        case SCV_VOLUME:
          numScalars += numScvIp;
          break;
        case SCV_GRAD_OP:
          dndxLength = nodesPerEntity*numScvIp*nDim;
          needDerivScv = true;
          needDetjScv = true;
          numScalars += dndxLength;
          break;
        case SCS_GIJ:
          gUpperLength = nDim*nDim*numScsIp;
          gLowerLength = nDim*nDim*numScsIp;
          needDeriv = true;
          numScalars += (gUpperLength + gLowerLength );
          break;
        case FEM_GRAD_OP:
        case FEM_SHIFTED_GRAD_OP:
          dndxLength = nodesPerEntity*numFemIp*nDim;
          needDerivFem = true;
          needDetjFem = true;
          numScalars += dndxLength;
          break;
        default: break;
      }
    }

    if (needDerivFC)
      numScalars += nodesPerEntity*numFaceIp*nDim;

    if (needDeriv)
      numScalars += nodesPerEntity*numScsIp*nDim;

    if (needDerivScv)
      numScalars += nodesPerEntity*numScvIp*nDim;

    if (needDerivFem)
      numScalars += nodesPerEntity*numFemIp*nDim;

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

KOKKOS_FUNCTION
void fill_pre_req_data(
  const ElemDataRequestsGPU& dataNeeded,
  const ngp::Mesh& ngpMesh,
  stk::mesh::EntityRank entityRank,
  stk::mesh::Entity entity,
  ScratchViews<double,DeviceTeamHandleType,DeviceShmem>& prereqData,
  bool  /* fillMEViews */)
{
  //MasterElement *meFC  = dataNeeded.get_cvfem_face_me();
  //MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
  //MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
  //MasterElement *meFEM = dataNeeded.get_fem_volume_me();

  stk::mesh::FastMeshIndex entityIndex = ngpMesh.fast_mesh_index(entity);
  prereqData.elemNodes = ngpMesh.get_nodes(entityRank, entityIndex);
  int nodesPerElem = prereqData.elemNodes.size();

  const ElemDataRequestsGPU::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned f=0; f<neededFields.size(); ++f) {
    const FieldInfoNGP& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    bool isTensorField = fieldInfo.scalarsDim2 > 1;

    if (fieldEntityRank==stk::topology::EDGE_RANK || fieldEntityRank==stk::topology::FACE_RANK || fieldEntityRank==stk::topology::ELEM_RANK) {
      if (isTensorField) {
        auto& shmemView = prereqData.get_scratch_view_2D(get_field_ordinal(fieldInfo));
        gather_elem_tensor_field(fieldInfo.field, entityIndex, scalarsDim1, fieldInfo.scalarsDim2, shmemView);
      }
      else {
        auto& shmemView = prereqData.get_scratch_view_1D(get_field_ordinal(fieldInfo));
        unsigned len = shmemView.extent(0);
        double* fieldDataPtr = static_cast<double*>(&fieldInfo.field.get(entityIndex,0));
        for(unsigned i=0; i<len; ++i) {
          shmemView(i) = fieldDataPtr[i];
        }
      }
    }
    else if (fieldEntityRank == stk::topology::NODE_RANK) {
      if (isTensorField) {
        auto& shmemView3D = prereqData.get_scratch_view_3D(get_field_ordinal(fieldInfo));
        gather_elem_node_tensor_field(fieldInfo.field, ngpMesh, nodesPerElem, scalarsDim1, fieldInfo.scalarsDim2, prereqData.elemNodes, shmemView3D);
      }
      else {
        if (scalarsDim1 == 1) {
          auto& shmemView1D = prereqData.get_scratch_view_1D(get_field_ordinal(fieldInfo));
          gather_elem_node_field(fieldInfo.field, ngpMesh, prereqData.elemNodes, shmemView1D);
        }
        else {
          auto& shmemView2D = prereqData.get_scratch_view_2D(get_field_ordinal(fieldInfo));
          if (scalarsDim1 == 3) {
            gather_elem_node_field_3D(fieldInfo.field, ngpMesh, prereqData.elemNodes, shmemView2D);
          }
          else {
            gather_elem_node_field(fieldInfo.field, ngpMesh, scalarsDim1, prereqData.elemNodes, shmemView2D);
          }
        }
      }
    }
    else {
      NGP_ThrowRequireMsg(false,"Unknown stk-rank in ScratchViewsNGP.C::fill_pre_req_data" );
    }
  }
/*
  if (fillMEViews)
  {
    const ElemDataRequestsGPU::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();
    const ElemDataRequestsGPU::FieldView& coordsFields = dataNeeded.get_coordinates_fields();
    for(unsigned i=0; i<coordsTypes.size(); ++i) {
      auto cType = coordsTypes(i);
      auto& coordField = coordsFields(i);

      const ElemDataRequestsGPU::DataEnumView& dataEnums = dataNeeded.get_data_enums(cType);
      auto* coordsView = &prereqData.get_scratch_view_2D(coordField.get_ordinal());
      auto& meData = prereqData.get_me_views(cType);

//      meData.fill_master_element_views(dataEnums, coordsView, meFC, meSCS, meSCV, meFEM);
    }
  }
*/
}

void fill_pre_req_data(
  const ElemDataRequestsNGP& dataNeeded,
  const stk::mesh::BulkData& bulkData,
  stk::mesh::Entity elem,
  ScratchViews<double,TeamHandleType,HostShmem>& prereqData,
  bool fillMEViews)
{
  int nodesPerElem = bulkData.num_nodes(elem);

  MasterElement *meFC  = dataNeeded.get_cvfem_face_me();
  MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
  MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
  MasterElement *meFEM = dataNeeded.get_fem_volume_me();
  prereqData.elemNodes = ngp::Mesh::ConnectedNodes(bulkData.begin_nodes(elem), bulkData.num_nodes(elem));

  const ElemDataRequestsNGP::FieldInfoView& neededFields = dataNeeded.get_fields();
  for(unsigned f=0; f<neededFields.size(); ++f) {
    const FieldInfo& fieldInfo = neededFields(f);
    stk::mesh::EntityRank fieldEntityRank = get_entity_rank(fieldInfo);
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    bool isTensorField = fieldInfo.scalarsDim2 > 1;

    if (fieldEntityRank==stk::topology::EDGE_RANK || fieldEntityRank==stk::topology::FACE_RANK || fieldEntityRank==stk::topology::ELEM_RANK) {
      if (isTensorField) {
        SharedMemView<double**>& shmemView = prereqData.get_scratch_view_2D(*fieldInfo.field);
        gather_elem_tensor_field(*fieldInfo.field, elem, scalarsDim1, fieldInfo.scalarsDim2, shmemView);
      }
      else {
        SharedMemView<double*>& shmemView = prereqData.get_scratch_view_1D(*fieldInfo.field);
        unsigned len = shmemView.extent(0);
        double* fieldDataPtr = static_cast<double*>(stk::mesh::field_data(*fieldInfo.field, elem));
        for(unsigned i=0; i<len; ++i) {
          shmemView(i) = fieldDataPtr[i];
        }
      }
    }
    else if (fieldEntityRank == stk::topology::NODE_RANK) {
      if (isTensorField) {
        auto shmemView3D = prereqData.get_scratch_view_3D(*fieldInfo.field);
        gather_elem_node_tensor_field(*fieldInfo.field, nodesPerElem, scalarsDim1, fieldInfo.scalarsDim2, prereqData.elemNodes, shmemView3D);
      }
      else {
        if (scalarsDim1 == 1) {
          auto shmemView1D = prereqData.get_scratch_view_1D(*fieldInfo.field);
          gather_elem_node_field(*fieldInfo.field, nodesPerElem, prereqData.elemNodes, shmemView1D);
        }
        else {
          auto shmemView2D = prereqData.get_scratch_view_2D(*fieldInfo.field);
          if (scalarsDim1 == 3) {
            gather_elem_node_field_3D(*fieldInfo.field, nodesPerElem, prereqData.elemNodes, shmemView2D);
          }
          else {
            gather_elem_node_field(*fieldInfo.field, nodesPerElem, scalarsDim1, prereqData.elemNodes, shmemView2D);
          }
        }
      }
    }
    else {
      ThrowRequireMsg(false,"Unknown stk-rank" << fieldEntityRank);
    }
  } 

  if (fillMEViews)
  {
    const ElemDataRequestsNGP::CoordsTypesView& coordsTypes = dataNeeded.get_coordinates_types();
    const ElemDataRequestsNGP::FieldView& coordsFields = dataNeeded.get_coordinates_fields();
    for(unsigned i=0; i<coordsTypes.size(); ++i) {
      auto cType = coordsTypes(i);
      const stk::mesh::FieldBase* coordField = coordsFields(i);

      const ElemDataRequestsNGP::DataEnumView& dataEnums = dataNeeded.get_data_enums(cType);
      auto coordsView = &prereqData.get_scratch_view_2D(*coordField);
      auto& meData = prereqData.get_me_views(cType);

      meData.fill_master_element_views(dataEnums, coordsView, meFC, meSCS, meSCV, meFEM);
    }
  }
}

}
}

