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

inline
void gather_elem_node_field(const ngp::Field<double>& field,
                            const ngp::Mesh& ngpMesh,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            SharedMemView<double*>& shmemView)
{
  for(unsigned i=0; i<elemNodes.size(); ++i) {
    shmemView[i] = field.get(ngpMesh, elemNodes[i], 0);
  }
}

inline
void gather_elem_node_tensor_field(const ngp::Field<double>& field,
                            const ngp::Mesh& ngpMesh,
                            int numNodes,
                            int tensorDim1,
                            int tensorDim2,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            SharedMemView<double***>& shmemView)
{
  ThrowRequireMsg(numNodes==(int)elemNodes.size(),"gather_elem_node_tensor_field, numNodes="<<numNodes<<" but elemNodes.size()="<<elemNodes.size());
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

inline
void gather_elem_tensor_field(const ngp::Field<double>& field,
                              stk::mesh::FastMeshIndex elem,
                              int tensorDim1,
                              int tensorDim2,
                              SharedMemView<double**>& shmemView)
{
  const double* dataPtr = static_cast<const double*>(&field.get(elem, 0));
  unsigned counter = 0;
  for(int d1=0; d1<tensorDim1; ++d1) { 
    for(int d2=0; d2<tensorDim2; ++d2) {
      shmemView(d1,d2) = dataPtr[counter++];
    }
  }
}

inline
void gather_elem_node_field_3D(const ngp::Field<double>& field,
                               const ngp::Mesh& ngpMesh,
                               const ngp::Mesh::ConnectedNodes& elemNodes,
                               SharedMemView<double**>& shmemView)
{
  for(unsigned i=0; i<elemNodes.size(); ++i) {
    const double* dataPtr = &field.get(ngpMesh, elemNodes[i], 0);
    shmemView(i,0) = dataPtr[0];
    shmemView(i,1) = dataPtr[1];
    shmemView(i,2) = dataPtr[2];
  }
}

inline
void gather_elem_node_field(const ngp::Field<double>& field,
                            const ngp::Mesh& ngpMesh,
                            int scalarsPerNode,
                            const ngp::Mesh::ConnectedNodes& elemNodes,
                            SharedMemView<double**>& shmemView)
{
  for(unsigned i=0; i<elemNodes.size(); ++i) {
    const double* dataPtr = &field.get(ngpMesh, elemNodes[i], 0);
    for(int d=0; d<scalarsPerNode; ++d) {
      shmemView(i,d) = dataPtr[d];
    }
  }
}

int get_num_scalars_pre_req_data(ElemDataRequests& dataNeededBySuppAlgs, int nDim)
{
  /* master elements are allowed to be null if they are not required */
  MasterElement *meFC  = dataNeededBySuppAlgs.get_cvfem_face_me();
  MasterElement *meSCS = dataNeededBySuppAlgs.get_cvfem_surface_me();
  MasterElement *meSCV = dataNeededBySuppAlgs.get_cvfem_volume_me();
  MasterElement *meFEM = dataNeededBySuppAlgs.get_fem_volume_me();
  
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
  
  const FieldSet& neededFields = dataNeededBySuppAlgs.get_fields();
  for(const FieldInfo& fieldInfo : neededFields) {
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field.get_rank();
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

  for (auto it = dataNeededBySuppAlgs.get_coordinates_map().begin();
       it != dataNeededBySuppAlgs.get_coordinates_map().end(); ++it)
  {
    const std::set<ELEM_DATA_NEEDED>& dataEnums =
      dataNeededBySuppAlgs.get_data_enums(it->first);
    int dndxLength = 0, dndxLengthFC = 0, gUpperLength = 0, gLowerLength = 0;

    // Updated logic for data sharing of deriv and det_j
    bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
    bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;

    for(ELEM_DATA_NEEDED data : dataEnums) {
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

int get_num_scalars_pre_req_data(ElemDataRequests& dataNeededBySuppAlgs, int nDim, const ScratchMeInfo &meInfo)
{
  const int nodesPerEntity = meInfo.nodalGatherSize_;
  const int numFaceIp = meInfo.numFaceIp_;
  const int numScsIp = meInfo.numScsIp_;
  const int numScvIp = meInfo.numScvIp_;
  const int numFemIp = meInfo.numFemIp_;
  int numScalars = 0;

  const FieldSet& neededFields = dataNeededBySuppAlgs.get_fields();
  for(const FieldInfo& fieldInfo : neededFields) {
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field.get_rank();
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

  for (auto it = dataNeededBySuppAlgs.get_coordinates_map().begin();
       it != dataNeededBySuppAlgs.get_coordinates_map().end(); ++it)
  {
    const std::set<ELEM_DATA_NEEDED>& dataEnums =
      dataNeededBySuppAlgs.get_data_enums(it->first);
    int dndxLength = 0, dndxLengthFC = 0, gUpperLength = 0, gLowerLength = 0;

    // Updated logic for data sharing of deriv and det_j
    bool needDeriv = false; bool needDerivScv = false; bool needDerivFem = false; bool needDerivFC = false;
    bool needDetj = false; bool needDetjScv = false; bool needDetjFem = false; bool needDetjFC = false;

    for(ELEM_DATA_NEEDED data : dataEnums) {
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


void fill_pre_req_data(
  const ElemDataRequests& dataNeeded,
  const ngp::Mesh& ngpMesh,
  stk::mesh::EntityRank entityRank,
  stk::mesh::Entity entity,
  ScratchViews<double>& prereqData,
  bool fillMEViews)
{
  MasterElement *meFC  = dataNeeded.get_cvfem_face_me();
  MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
  MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
  MasterElement *meFEM = dataNeeded.get_fem_volume_me();

  stk::mesh::FastMeshIndex entityIndex = ngpMesh.fast_mesh_index(entity);
  prereqData.elemNodes = ngpMesh.get_nodes(entityRank, entityIndex);
  int nodesPerElem = prereqData.elemNodes.size();

  const FieldSet& neededFields = dataNeeded.get_fields();
  for(const FieldInfo& fieldInfo : neededFields) {
    stk::mesh::EntityRank fieldEntityRank = fieldInfo.field.get_rank();
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;
    bool isTensorField = fieldInfo.scalarsDim2 > 1;

    if (fieldEntityRank==stk::topology::EDGE_RANK || fieldEntityRank==stk::topology::FACE_RANK || fieldEntityRank==stk::topology::ELEM_RANK) {
      if (isTensorField) {
        SharedMemView<double**>& shmemView = prereqData.get_scratch_view_2D(fieldInfo.field);
        gather_elem_tensor_field(fieldInfo.field, entityIndex, scalarsDim1, fieldInfo.scalarsDim2, shmemView);
      }
      else {
        SharedMemView<double*>& shmemView = prereqData.get_scratch_view_1D(fieldInfo.field);
        unsigned len = shmemView.dimension(0);
        double* fieldDataPtr = static_cast<double*>(&fieldInfo.field.get(entityIndex,0));
        for(unsigned i=0; i<len; ++i) {
          shmemView(i) = fieldDataPtr[i];
        }
      }
    }
    else if (fieldEntityRank == stk::topology::NODE_RANK) {
      if (isTensorField) {
        SharedMemView<double***>& shmemView3D = prereqData.get_scratch_view_3D(fieldInfo.field);
        gather_elem_node_tensor_field(fieldInfo.field, ngpMesh, nodesPerElem, scalarsDim1, fieldInfo.scalarsDim2, prereqData.elemNodes, shmemView3D);
      }
      else {
        if (scalarsDim1 == 1) {
          SharedMemView<double*>& shmemView1D = prereqData.get_scratch_view_1D(fieldInfo.field);
          gather_elem_node_field(fieldInfo.field, ngpMesh, prereqData.elemNodes, shmemView1D);
        }
        else {
          SharedMemView<double**>& shmemView2D = prereqData.get_scratch_view_2D(fieldInfo.field);
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
      ThrowRequireMsg(false,"Unknown stk-rank" << fieldEntityRank);
    }
  } 

  if (fillMEViews)
  {
    for (auto it = dataNeeded.get_coordinates_map().begin();
         it != dataNeeded.get_coordinates_map().end(); ++it) {
      auto cType = it->first;
      auto coordField = it->second;
  
      const std::set<ELEM_DATA_NEEDED>& dataEnums = dataNeeded.get_data_enums(cType);
      SharedMemView<double**>* coordsView = &prereqData.get_scratch_view_2D(coordField);
      auto& meData = prereqData.get_me_views(cType);
  
      meData.fill_master_element_views(dataEnums, coordsView, meFC, meSCS, meSCV, meFEM);
    }
  }
}

void fill_master_element_views(
  ElemDataRequests& dataNeeded,
  ScratchViews<DoubleType>& prereqData,
  int faceOrdinal)
{
    MasterElement *meFC  = dataNeeded.get_cvfem_face_me();
    MasterElement *meSCS = dataNeeded.get_cvfem_surface_me();
    MasterElement *meSCV = dataNeeded.get_cvfem_volume_me();
    MasterElement *meFEM = dataNeeded.get_fem_volume_me();

    for (auto it = dataNeeded.get_coordinates_map().begin();
         it != dataNeeded.get_coordinates_map().end(); ++it) {
      auto cType = it->first;
      auto coordField = it->second;
  
      const std::set<ELEM_DATA_NEEDED>& dataEnums = dataNeeded.get_data_enums(cType);
      SharedMemView<DoubleType**>* coordsView = &prereqData.get_scratch_view_2D(coordField);
      auto& meData = prereqData.get_me_views(cType);
  
      meData.fill_master_element_views_new_me(dataEnums, coordsView, meFC, meSCS, meSCV, meFEM, faceOrdinal);
    }
}

}
}

