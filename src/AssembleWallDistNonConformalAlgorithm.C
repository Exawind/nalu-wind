/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "AssembleWallDistNonConformalAlgorithm.h"

#include "DgInfo.h"
#include "EquationSystem.h"
#include "LinearSystem.h"
#include "NonConformalInfo.h"
#include "NonConformalManager.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Part.hpp"
#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

AssembleWallDistNonConformalAlgorithm::AssembleWallDistNonConformalAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    useCurrentNormal_(realm.get_nc_alg_current_normal())
{
  auto& meta = realm.meta_data();

  coordinates_ = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm.get_coordinates_name());
  exposedAreaVec_ = meta.get_field<GenericFieldType>(
    meta.side_rank(), "exposed_area_vector");

  ghostFieldVec_.push_back(coordinates_);
}

void
AssembleWallDistNonConformalAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildNonConformalNodeGraph(partVec_);
}

void
AssembleWallDistNonConformalAlgorithm::execute()
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const int nDim = meta.spatial_dimension();

  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  std::vector<double> ws_c_gen_shpf;
  std::vector<double> ws_o_gen_shpf;
  std::vector<double> ws_oppCoords;
  std::vector<double> ws_curElemCoords;
  std::vector<double> ws_oppElemCoords;
  std::vector<double> cur_detj(1);
  std::vector<double> opp_detj(1);
  std::vector<double> cur_dndx;
  std::vector<double> opp_dndx;

  std::vector<double> curNx(nDim);
  std::vector<double> oppNx(nDim);
  std::vector<double> curElemIsoParCrd(nDim);
  std::vector<double> oppElemIsoParCrd(nDim);

  if (realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, ghostFieldVec_);

  for(auto dgi: realm_.nonConformalManager_->nonConformalInfoVec_) {
    auto& dgInfoVec = dgi->dgInfoVec_;

    for (auto fdgi: dgInfoVec) {
      for (size_t k=0; k < fdgi.size(); ++k) {
        auto* dgInfo = fdgi[k];

        auto curFace = dgInfo->currentFace_;
        auto oppFace = dgInfo->opposingFace_;
        auto curElem = dgInfo->currentElement_;
        auto oppElem = dgInfo->opposingElement_;
        const int curFaceOrd = dgInfo->currentFaceOrdinal_;
        const int oppFaceOrd = dgInfo->opposingFaceOrdinal_;

        auto* curMEFC = dgInfo->meFCCurrent_;
        auto* oppMEFC = dgInfo->meFCOpposing_;
        auto* curMESCS = dgInfo->meSCSCurrent_;
        auto* oppMESCS = dgInfo->meSCSOpposing_;

        const int curGaussPId = dgInfo->currentGaussPointId_;
        auto& curIsoParCrd = dgInfo->currentIsoParCoords_;
        auto& oppIsoParCrd = dgInfo->opposingIsoParCoords_;

        const int curNPF = curMEFC->nodesPerElement_;
        const int oppNPF = oppMEFC->nodesPerElement_;
        const int curNPE = curMESCS->nodesPerElement_;
        const int oppNPE = oppMESCS->nodesPerElement_;

        const int totalNodes = curNPE + oppNPE;
        const int* ipNodeMap = curMESCS->ipNodeMap(curFaceOrd);
        const int nn = ipNodeMap[curGaussPId];
        const int rowR = nn * totalNodes;

        // resize work arrays
        lhs.resize(totalNodes * totalNodes);
        rhs.resize(totalNodes);
        scratchIds.resize(totalNodes);
        scratchVals.resize(totalNodes);
        connected_nodes.resize(totalNodes);
        ws_c_gen_shpf.resize(curNPF);
        ws_o_gen_shpf.resize(oppNPF);
        ws_oppCoords.resize(oppNPF * nDim);
        ws_curElemCoords.resize(curNPE * nDim);
        ws_oppElemCoords.resize(oppNPE * nDim);
        cur_dndx.resize(curNPE * nDim);
        opp_dndx.resize(oppNPE * nDim);

        const int* c_face_node_ordinals = curMESCS->side_node_ordinals(curFaceOrd);
        const int* o_face_node_ordinals = oppMESCS->side_node_ordinals(oppFaceOrd);

        // zero out lhs/rhs before population
        double* p_lhs = lhs.data();
        double* p_rhs = rhs.data();
        for ( int p = 0; p < totalNodes*totalNodes; ++p )
          p_lhs[p] = 0.0;
        for ( int p = 0; p < totalNodes; ++p )
          p_rhs[p] = 0.0;

        const auto* oppFaceNodeRels = bulk.begin_nodes(oppFace);
        const int oppNumFaceNodes = bulk.num_nodes(oppFace);
        for (int ni=0; ni < oppNumFaceNodes; ni++) {
          auto& node = oppFaceNodeRels[ni];
          const double* coords = stk::mesh::field_data(*coordinates_, node);
          for(int i=0; i < nDim; i++)
            ws_oppCoords[ni * nDim + i] = coords[i];
        }

        const auto* curElemNodeRels = bulk.begin_nodes(curElem);
        const int curNumElemNodes = bulk.num_nodes(curElem);
        for (int ni=0; ni < curNumElemNodes; ni++) {
          auto& node = curElemNodeRels[ni];
          connected_nodes[ni] = node;

          const double* coords = stk::mesh::field_data(*coordinates_, node);
          const int offset = ni * nDim;
          for (int i=0; i < nDim; i++)
            ws_curElemCoords[offset + i] = coords[i];
        }

        const auto* oppElemNodeRels = bulk.begin_nodes(oppElem);
        const int oppNumElemNodes = bulk.num_nodes(oppElem);
        for (int ni=0; ni < oppNumElemNodes; ni++) {
          auto& node = oppElemNodeRels[ni];
          connected_nodes[ni + curNumElemNodes] = node;

          const double* coords = stk::mesh::field_data(*coordinates_, node);
          const int offset = ni * nDim;
          for (int i=0; i < nDim; i++)
            ws_oppElemCoords[offset + i] = coords[i];
        }

        // Compute the magnitude of the exposed area
        const double* c_areaVec = stk::mesh::field_data(*exposedAreaVec_, curFace);
        double c_amag = 0.0;
        for (int j=0; j < nDim; j++) {
          const double c_axj = c_areaVec[curGaussPId * nDim + j];
          c_amag += c_axj * c_axj;
        }
        c_amag = std::sqrt(c_amag);

        // Compute the unit normal for the current and opposing faces
        for (int i=0; i < nDim; i++)
          curNx[i] = c_areaVec[curGaussPId * nDim + i] / c_amag;

        if (useCurrentNormal_) {
          for (int i=0; i < nDim; i++)
            oppNx[i] = -curNx[i];
        } else {
          oppMEFC->general_normal(oppIsoParCrd.data(), ws_oppCoords.data(), oppNx.data());
        }

        // Convert [-1, 1] iso-parametric coords to [-0.5, 0.5]
        curMESCS->sidePcoords_to_elemPcoords(
          curFaceOrd, 1, curIsoParCrd.data(), curElemIsoParCrd.data());
        oppMESCS->sidePcoords_to_elemPcoords(
          oppFaceOrd, 1, oppIsoParCrd.data(), oppElemIsoParCrd.data());

        // Face gradient operators to compute the inverse lengths
        double scs_error = 0.0;
        curMESCS->general_face_grad_op(
          curFaceOrd, curElemIsoParCrd.data(), ws_curElemCoords.data(),
          cur_dndx.data(), cur_detj.data(), &scs_error);
        oppMESCS->general_face_grad_op(
          oppFaceOrd, oppElemIsoParCrd.data(), ws_oppElemCoords.data(),
          opp_dndx.data(), opp_detj.data(), &scs_error);

        // Inverse lengths for current and opposing faces
        double curInvLen = 0.0;
        double oppInvLen = 0.0;
        for (int ic=0; ic < curNumElemNodes; ++ic) {
          const int fnnum = c_face_node_ordinals[ic];
          const int offset = fnnum * nDim;
          for (int j=0; j < nDim; j++) {
            const double nxj = curNx[j];
            const double dndxj = cur_dndx[offset + j];
            curInvLen += dndxj * nxj;
          }
        }

        for (int ic=0; ic < oppNumElemNodes; ++ic) {
          const int fnnum = o_face_node_ordinals[ic];
          const int offset = fnnum * nDim;
          for (int j=0; j < nDim; j++) {
            const double nxj = oppNx[j];
            const double dndxj = opp_dndx[offset + j];
            oppInvLen += dndxj * nxj;
          }
        }

        double totlen = 0.5 * (curInvLen + oppInvLen);
        double lhsfac = totlen * c_amag;
        curMEFC->general_shape_fcn(1, curIsoParCrd.data(), ws_c_gen_shpf.data());
        for (int ic=0; ic < curNPF; ++ic) {
          const int icnn = c_face_node_ordinals[ic];
          const double r = ws_c_gen_shpf[ic];
          p_lhs[rowR + icnn] += r * lhsfac;
        }

        oppMEFC->general_shape_fcn(1, oppIsoParCrd.data(), ws_o_gen_shpf.data());
        for (int ic=0; ic < oppNPF; ic++) {
          const int icnn = o_face_node_ordinals[ic];
          const double r = ws_o_gen_shpf[ic];
          p_lhs[rowR + icnn + curNPE] -= r * lhsfac;
        }

        // No RHS contributions

        apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
      }
    }
  }
}

}  // nalu
}  // sierra
