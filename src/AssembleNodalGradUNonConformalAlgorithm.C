// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <AssembleNodalGradUNonConformalAlgorithm.h>
#include <Algorithm.h>
#include <DgInfo.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradUNonConformalAlgorithm - assemble bc nodal grad; both
//                                          edge and elem
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradUNonConformalAlgorithm::
  AssembleNodalGradUNonConformalAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    VectorFieldType* vectorQ,
    TensorFieldType* dqdx)
  : Algorithm(realm, part),
    vectorQ_(vectorQ),
    dqdx_(dqdx),
    dualNodalVolume_(NULL),
    exposedAreaVec_(NULL)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  dualNodalVolume_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");
  exposedAreaVec_ =
    meta_data.get_field<double>(meta_data.side_rank(), "exposed_area_vector");

  // what do we need ghosted for this alg to work?
  ghostFieldVec_.push_back(vectorQ_);
  ghostFieldVec_.push_back(dualNodalVolume_);
  ghostFieldVec_.push_back(dqdx_);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradUNonConformalAlgorithm::
  ~AssembleNodalGradUNonConformalAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradUNonConformalAlgorithm::execute()
{
  stk::mesh::BulkData& bulk_data = realm_.bulk_data();
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // ip values; both boundary and opposing surface
  std::vector<double> currentIsoParCoords(nDim);
  std::vector<double> opposingIsoParCoords(nDim);

  // space for current/opposing interpolated value for scalarQ
  std::vector<double> currentVectorQBip(nDim);
  std::vector<double> opposingVectorQBip(nDim);

  // interpolate nodal values to point-in-elem
  const int sizeOfVectorField = nDim;

  // nodal fields to gather
  std::vector<double> ws_c_vectorQ;
  std::vector<double> ws_o_vectorQ;

  // parallel communicate ghosted entities
  if (NULL != realm_.nonConformalManager_->nonConformalGhosting_)
    stk::mesh::communicate_field_data(
      *(realm_.nonConformalManager_->nonConformalGhosting_), ghostFieldVec_);

  dqdx_->clear_sync_state();
  // iterate nonConformalManager's dgInfoVec
  std::vector<NonConformalInfo*>::iterator ii;
  for (ii = realm_.nonConformalManager_->nonConformalInfoVec_.begin();
       ii != realm_.nonConformalManager_->nonConformalInfoVec_.end(); ++ii) {

    // extract vector of DgInfo
    std::vector<std::vector<DgInfo*>>& dgInfoVec = (*ii)->dgInfoVec_;

    std::vector<std::vector<DgInfo*>>::iterator idg;
    for (idg = dgInfoVec.begin(); idg != dgInfoVec.end(); ++idg) {

      std::vector<DgInfo*>& faceDgInfoVec = (*idg);

      // now loop over all the DgInfo objects on this particular exposed face
      for (size_t k = 0; k < faceDgInfoVec.size(); ++k) {

        DgInfo* dgInfo = faceDgInfoVec[k];

        // extract current/opposing face/element
        stk::mesh::Entity currentFace = dgInfo->currentFace_;
        stk::mesh::Entity opposingFace = dgInfo->opposingFace_;

        // master element
        MasterElement* meFCCurrent = dgInfo->meFCCurrent_;
        MasterElement* meFCOpposing = dgInfo->meFCOpposing_;

        // local ip, ordinals, etc
        const int currentGaussPointId = dgInfo->currentGaussPointId_;
        currentIsoParCoords = dgInfo->currentIsoParCoords_;
        opposingIsoParCoords = dgInfo->opposingIsoParCoords_;

        // mapping from ip to nodes for this ordinal
        const int* faceIpNodeMap = meFCCurrent->ipNodeMap();

        // extract some master element info
        const int currentNodesPerFace = meFCCurrent->nodesPerElement_;
        const int opposingNodesPerFace = meFCOpposing->nodesPerElement_;

        // algorithm related; face
        ws_c_vectorQ.resize(currentNodesPerFace * nDim);
        ws_o_vectorQ.resize(opposingNodesPerFace * nDim);

        double* p_c_vectorQ = &ws_c_vectorQ[0];
        double* p_o_vectorQ = &ws_o_vectorQ[0];

        // gather current face data
        stk::mesh::Entity const* current_face_node_rels =
          bulk_data.begin_nodes(currentFace);
        const int current_num_face_nodes = bulk_data.num_nodes(currentFace);
        for (int ni = 0; ni < current_num_face_nodes; ++ni) {
          stk::mesh::Entity node = current_face_node_rels[ni];
          // gather...
          const double* qNp1 = stk::mesh::field_data(*vectorQ_, node);
          for (int i = 0; i < nDim; ++i) {
            const int offSet = i * current_num_face_nodes + ni;
            p_c_vectorQ[offSet] = qNp1[i];
          }
        }

        // gather opposing face data
        stk::mesh::Entity const* opposing_face_node_rels =
          bulk_data.begin_nodes(opposingFace);
        const int opposing_num_face_nodes = bulk_data.num_nodes(opposingFace);
        for (int ni = 0; ni < opposing_num_face_nodes; ++ni) {
          stk::mesh::Entity node = opposing_face_node_rels[ni];
          // gather...
          const double* qNp1 = stk::mesh::field_data(*vectorQ_, node);
          for (int i = 0; i < nDim; ++i) {
            const int offSet = i * opposing_num_face_nodes + ni;
            p_o_vectorQ[offSet] = qNp1[i];
          }
        }

        meFCCurrent->interpolatePoint(
          sizeOfVectorField, &(dgInfo->currentIsoParCoords_[0]),
          &ws_c_vectorQ[0], &currentVectorQBip[0]);

        meFCOpposing->interpolatePoint(
          sizeOfVectorField, &(dgInfo->opposingIsoParCoords_[0]),
          &ws_o_vectorQ[0], &opposingVectorQBip[0]);

        // extract pointers to nearest node fields
        const int nn = faceIpNodeMap[currentGaussPointId];
        stk::mesh::Entity nNode = current_face_node_rels[nn];

        const double volNN = *stk::mesh::field_data(*dualNodalVolume_, nNode);
        double* dqdx = stk::mesh::field_data(*dqdx_, nNode);

        // nearest node inverse volume
        double inv_volNN = 1.0 / volNN;

        // pointer to face data
        const double* c_areaVec =
          stk::mesh::field_data(*exposedAreaVec_, currentFace);

        // assemble to nearest node
        for (int i = 0; i < nDim; ++i) {
          const int row_gradQ = i * nDim;
          double qip = 0.5 * (currentVectorQBip[i] + opposingVectorQBip[i]);
          for (int j = 0; j < nDim; ++j) {
            double fac = qip * c_areaVec[currentGaussPointId * nDim + j];
            dqdx[row_gradQ + j] += fac * inv_volNN;
          }
        }
      }
    }
  }

  dqdx_->modify_on_host();
}

} // namespace nalu
} // namespace sierra
