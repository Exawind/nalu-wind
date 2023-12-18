// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <AssemblePNGNonConformalSolverAlgorithm.h>
#include <EquationSystem.h>
#include <DgInfo.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <Realm.h>
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
// AssemblePNGNonConformalSolverAlgorithm - PNG for NC alg
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssemblePNGNonConformalSolverAlgorithm::AssemblePNGNonConformalSolverAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  std::string independentDofName,
  std::string dofName,
  const bool includePenalty)
  : SolverAlgorithm(realm, part, eqSystem),
    scalarQ_(NULL),
    Gjq_(NULL),
    coordinates_(NULL),
    exposedAreaVec_(NULL),
    useCurrentNormal_(realm_.get_nc_alg_current_normal()),
    includePenalty_(includePenalty)
{
  // save off data
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  scalarQ_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, independentDofName);
  Gjq_ = meta_data.get_field<double>(stk::topology::NODE_RANK, dofName);
  coordinates_ = meta_data.get_field<double>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  exposedAreaVec_ =
    meta_data.get_field<double>(meta_data.side_rank(), "exposed_area_vector");

  // what do we need ghosted for this alg to work?
  ghostFieldVec_.push_back(scalarQ_);
  ghostFieldVec_.push_back(Gjq_);
  ghostFieldVec_.push_back(coordinates_);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssemblePNGNonConformalSolverAlgorithm::
  ~AssemblePNGNonConformalSolverAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssemblePNGNonConformalSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildNonConformalNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssemblePNGNonConformalSolverAlgorithm::execute()
{
  stk::mesh::BulkData& bulk_data = realm_.bulk_data();
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const double penaltyFac = includePenalty_ ? 1.0 : 0.0;

  // space for LHS/RHS; nodesPerElem*nDim*nodesPerElem*nDim and
  // nodesPerElem*nDim
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // ip values; both boundary and opposing surface
  std::vector<double> currentIsoParCoords(nDim);
  std::vector<double> opposingIsoParCoords(nDim);
  std::vector<double> cNx(nDim);
  std::vector<double> oNx(nDim);

  // mapping for -1:1 -> -0.5:0.5 volume element
  std::vector<double> currentElementIsoParCoords(nDim);
  std::vector<double> opposingElementIsoParCoords(nDim);

  // space for current and opposing Gjq
  std::vector<double> currentGjqBip(nDim);
  std::vector<double> opposingGjqBip(nDim);

  // interpolate nodal values to point-in-elem
  const int sizeOfScalarField = 1;
  const int sizeOfVectorField = nDim;

  // pointers to fixed values
  double* p_cNx = &cNx[0];
  double* p_oNx = &oNx[0];

  // nodal fields to gather; face and element
  std::vector<double> ws_c_scalarQ;
  std::vector<double> ws_o_scalarQ;
  std::vector<double> ws_c_Gjq;
  std::vector<double> ws_o_Gjq;
  std::vector<double> ws_o_coordinates;
  std::vector<double> ws_c_elem_coordinates;
  std::vector<double> ws_o_elem_coordinates;

  // master element data
  std::vector<double> ws_c_dndx;
  std::vector<double> ws_o_dndx;
  std::vector<double> ws_c_det_j;
  std::vector<double> ws_o_det_j;
  std::vector<double> ws_c_general_shape_function;
  std::vector<double> ws_o_general_shape_function;
  std::vector<int> ws_c_face_node_ordinals;
  std::vector<int> ws_o_face_node_ordinals;

  // parallel communicate ghosted entities
  if (NULL != realm_.nonConformalManager_->nonConformalGhosting_)
    stk::mesh::communicate_field_data(
      *(realm_.nonConformalManager_->nonConformalGhosting_), ghostFieldVec_);

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
        stk::mesh::Entity currentElement = dgInfo->currentElement_;
        stk::mesh::Entity opposingElement = dgInfo->opposingElement_;
        stk::topology currentElementTopo = dgInfo->currentElementTopo_;
        stk::topology opposingElementTopo = dgInfo->opposingElementTopo_;
        const int currentFaceOrdinal = dgInfo->currentFaceOrdinal_;
        const int opposingFaceOrdinal = dgInfo->opposingFaceOrdinal_;

        // master element
        MasterElement* meFCCurrent = dgInfo->meFCCurrent_;
        MasterElement* meFCOpposing = dgInfo->meFCOpposing_;
        MasterElement* meSCSCurrent = dgInfo->meSCSCurrent_;
        MasterElement* meSCSOpposing = dgInfo->meSCSOpposing_;

        // local ip, ordinals, etc
        const int currentGaussPointId = dgInfo->currentGaussPointId_;
        currentIsoParCoords = dgInfo->currentIsoParCoords_;
        opposingIsoParCoords = dgInfo->opposingIsoParCoords_;

        // mapping from ip to nodes for this ordinal
        const int* ipNodeMap = meSCSCurrent->ipNodeMap(currentFaceOrdinal);

        // extract some master element info
        const int currentNodesPerFace = meFCCurrent->nodesPerElement_;
        const int opposingNodesPerFace = meFCOpposing->nodesPerElement_;
        const int currentNodesPerElement = meSCSCurrent->nodesPerElement_;
        const int opposingNodesPerElement = meSCSOpposing->nodesPerElement_;

        // resize some things; matrix related
        const int totalNodes = currentNodesPerElement + opposingNodesPerElement;
        const int lhsSize = totalNodes * nDim * totalNodes * nDim;
        const int rhsSize = totalNodes * nDim;
        lhs.resize(lhsSize);
        rhs.resize(rhsSize);
        scratchIds.resize(rhsSize);
        scratchVals.resize(rhsSize);
        connected_nodes.resize(totalNodes);

        // algorithm related; element; dndx will be at a single gauss point...
        ws_c_elem_coordinates.resize(currentNodesPerElement * nDim);
        ws_o_elem_coordinates.resize(opposingNodesPerElement * nDim);
        ws_c_dndx.resize(nDim * currentNodesPerElement);
        ws_o_dndx.resize(nDim * opposingNodesPerElement);
        ws_c_det_j.resize(1);
        ws_o_det_j.resize(1);

        // algorithm related; face
        ws_c_scalarQ.resize(currentNodesPerFace);
        ws_o_scalarQ.resize(opposingNodesPerFace);
        ws_c_Gjq.resize(currentNodesPerFace * nDim);
        ws_o_Gjq.resize(opposingNodesPerFace * nDim);
        ws_o_coordinates.resize(opposingNodesPerFace * nDim);
        ws_c_general_shape_function.resize(currentNodesPerFace);
        ws_o_general_shape_function.resize(opposingNodesPerFace);

        // face node identification
        ws_c_face_node_ordinals.resize(currentNodesPerFace);
        ws_o_face_node_ordinals.resize(opposingNodesPerFace);

        // pointers
        double* p_lhs = &lhs[0];
        double* p_rhs = &rhs[0];

        double* p_c_scalarQ = &ws_c_scalarQ[0];
        double* p_o_scalarQ = &ws_o_scalarQ[0];
        double* p_c_Gjq = &ws_c_Gjq[0];
        double* p_o_Gjq = &ws_o_Gjq[0];
        double* p_c_elem_coordinates = &ws_c_elem_coordinates[0];
        double* p_o_elem_coordinates = &ws_o_elem_coordinates[0];
        double* p_o_coordinates = &ws_o_coordinates[0];

        // me pointers
        double* p_c_general_shape_function = &ws_c_general_shape_function[0];
        double* p_o_general_shape_function = &ws_o_general_shape_function[0];
        double* p_c_dndx = &ws_c_dndx[0];
        double* p_o_dndx = &ws_o_dndx[0];

        // populate current face_node_ordinals
        currentElementTopo.side_node_ordinals(
          currentFaceOrdinal, ws_c_face_node_ordinals.begin());

        // gather current face data
        stk::mesh::Entity const* current_face_node_rels =
          bulk_data.begin_nodes(currentFace);
        const int current_num_face_nodes = bulk_data.num_nodes(currentFace);
        for (int ni = 0; ni < current_num_face_nodes; ++ni) {
          stk::mesh::Entity node = current_face_node_rels[ni];
          // gather...
          p_c_scalarQ[ni] = *stk::mesh::field_data(*scalarQ_, node);
          // gather; vector
          const double* Gjq = stk::mesh::field_data(*Gjq_, node);
          for (int i = 0; i < nDim; ++i) {
            const int offSet = i * current_num_face_nodes + ni;
            p_c_Gjq[offSet] = Gjq[i];
          }
        }

        // populate opposing face_node_ordinals
        opposingElementTopo.side_node_ordinals(
          opposingFaceOrdinal, ws_o_face_node_ordinals.begin());

        // gather opposing face data
        stk::mesh::Entity const* opposing_face_node_rels =
          bulk_data.begin_nodes(opposingFace);
        const int opposing_num_face_nodes = bulk_data.num_nodes(opposingFace);
        for (int ni = 0; ni < opposing_num_face_nodes; ++ni) {
          stk::mesh::Entity node = opposing_face_node_rels[ni];
          // gather; scalar
          p_o_scalarQ[ni] = *stk::mesh::field_data(*scalarQ_, node);
          // gather; vector
          const double* Gjq = stk::mesh::field_data(*Gjq_, node);
          const double* coords = stk::mesh::field_data(*coordinates_, node);
          for (int i = 0; i < nDim; ++i) {
            const int offSet = i * opposing_num_face_nodes + ni;
            p_o_Gjq[offSet] = Gjq[i];
            p_o_coordinates[ni * nDim + i] = coords[i];
          }
        }

        // gather current element data; sneak in first of connected nodes
        stk::mesh::Entity const* current_elem_node_rels =
          bulk_data.begin_nodes(currentElement);
        const int current_num_elem_nodes = bulk_data.num_nodes(currentElement);
        for (int ni = 0; ni < current_num_elem_nodes; ++ni) {
          stk::mesh::Entity node = current_elem_node_rels[ni];
          // set connected nodes
          connected_nodes[ni] = node;
          // gather; vector
          const double* coords = stk::mesh::field_data(*coordinates_, node);
          const int niNdim = ni * nDim;
          for (int i = 0; i < nDim; ++i) {
            p_c_elem_coordinates[niNdim + i] = coords[i];
          }
        }

        // gather opposing element data; sneak in second connected nodes
        stk::mesh::Entity const* opposing_elem_node_rels =
          bulk_data.begin_nodes(opposingElement);
        const int opposing_num_elem_nodes =
          bulk_data.num_nodes(opposingElement);
        for (int ni = 0; ni < opposing_num_elem_nodes; ++ni) {
          stk::mesh::Entity node = opposing_elem_node_rels[ni];
          // set connected nodes
          connected_nodes[ni + current_num_elem_nodes] = node;
          // gather; vector
          const double* coords = stk::mesh::field_data(*coordinates_, node);
          const int niNdim = ni * nDim;
          for (int i = 0; i < nDim; ++i) {
            p_o_elem_coordinates[niNdim + i] = coords[i];
          }
        }

        // compute opposing normal through master element call, not using
        // oppoing exposed area
        meFCOpposing->general_normal(
          &opposingIsoParCoords[0], &p_o_coordinates[0], &p_oNx[0]);

        // pointer to face data
        const double* c_areaVec =
          stk::mesh::field_data(*exposedAreaVec_, currentFace);

        // compute area magnitide at current
        double c_amag = 0.0;
        for (int j = 0; j < nDim; ++j) {
          const double c_axj = c_areaVec[currentGaussPointId * nDim + j];
          c_amag += c_axj * c_axj;
        }
        c_amag = std::sqrt(c_amag);

        // now compute normal
        for (int i = 0; i < nDim; ++i) {
          p_cNx[i] = c_areaVec[currentGaussPointId * nDim + i] / c_amag;
        }

        // override opposing normal
        if (useCurrentNormal_) {
          for (int i = 0; i < nDim; ++i)
            p_oNx[i] = -p_cNx[i];
        }

        // project from side to element; method deals with the -1:1 isInElement
        // range to the proper underlying CVFEM range
        meSCSCurrent->sidePcoords_to_elemPcoords(
          currentFaceOrdinal, 1, &currentIsoParCoords[0],
          &currentElementIsoParCoords[0]);
        meSCSOpposing->sidePcoords_to_elemPcoords(
          opposingFaceOrdinal, 1, &opposingIsoParCoords[0],
          &opposingElementIsoParCoords[0]);

        // compute dndx
        double scs_error = 0.0;
        meSCSCurrent->general_face_grad_op(
          currentFaceOrdinal, &currentElementIsoParCoords[0],
          &p_c_elem_coordinates[0], &p_c_dndx[0], &ws_c_det_j[0], &scs_error);
        meSCSOpposing->general_face_grad_op(
          opposingFaceOrdinal, &opposingElementIsoParCoords[0],
          &p_o_elem_coordinates[0], &p_o_dndx[0], &ws_o_det_j[0], &scs_error);

        // current inverse length scale; can loop over face nodes to avoid
        // "nodesOnFace" array
        double currentInverseLength = 0.0;
        for (int ic = 0; ic < current_num_face_nodes; ++ic) {
          const int faceNodeNumber = ws_c_face_node_ordinals[ic];
          const int offSetDnDx = faceNodeNumber * nDim; // single intg. point
          for (int j = 0; j < nDim; ++j) {
            const double nxj = p_cNx[j];
            const double dndxj = p_c_dndx[offSetDnDx + j];
            currentInverseLength += dndxj * nxj;
          }
        }

        // opposing inverse length scale; can loop over face nodes to avoid
        // "nodesOnFace" array
        double opposingInverseLength = 0.0;
        for (int ic = 0; ic < opposing_num_face_nodes; ++ic) {
          const int faceNodeNumber = ws_o_face_node_ordinals[ic];
          const int offSetDnDx = faceNodeNumber * nDim; // single intg. point
          for (int j = 0; j < nDim; ++j) {
            const double nxj = p_oNx[j];
            const double dndxj = p_o_dndx[offSetDnDx + j];
            opposingInverseLength += dndxj * nxj;
          }
        }

        // interpolate face data; current and opposing...
        double currentScalarQBip = 0.0;
        meFCCurrent->interpolatePoint(
          sizeOfScalarField, &(dgInfo->currentIsoParCoords_[0]),
          &ws_c_scalarQ[0], &currentScalarQBip);

        double opposingScalarQBip = 0.0;
        meFCOpposing->interpolatePoint(
          sizeOfScalarField, &(dgInfo->opposingIsoParCoords_[0]),
          &ws_o_scalarQ[0], &opposingScalarQBip);

        // projected nodal gradient
        meFCCurrent->interpolatePoint(
          sizeOfVectorField, &(dgInfo->currentIsoParCoords_[0]), &ws_c_Gjq[0],
          &currentGjqBip[0]);

        meFCOpposing->interpolatePoint(
          sizeOfVectorField, &(dgInfo->opposingIsoParCoords_[0]), &ws_o_Gjq[0],
          &opposingGjqBip[0]);

        // zero lhs/rhs
        for (int p = 0; p < lhsSize; ++p)
          p_lhs[p] = 0.0;
        for (int p = 0; p < rhsSize; ++p)
          p_rhs[p] = 0.0;

        // form penalty and mean value of scalarQ at this bip; penalty term
        // includes an interesting recursion
        const double penaltyIp =
          0.5 * (1.0 / currentInverseLength + 1.0 / opposingInverseLength) *
          penaltyFac;
        const double ncScalarQ = 0.5 * (currentScalarQBip + opposingScalarQBip);

        // extract nearset node
        const int nn = ipNodeMap[currentGaussPointId];

        // compute general shape function at current and opposing integration
        // points
        meFCCurrent->general_shape_fcn(
          1, &currentIsoParCoords[0], &ws_c_general_shape_function[0]);
        meFCOpposing->general_shape_fcn(
          1, &opposingIsoParCoords[0], &ws_o_general_shape_function[0]);

        // loop over all components of gradient, i
        for (int i = 0; i < nDim; ++i) {

          const int indexR = nn * nDim + i;

          // form residual
          p_rhs[indexR] +=
            ncScalarQ * c_areaVec[currentGaussPointId * nDim + i] +
            penaltyIp * (currentGjqBip[i] - opposingGjqBip[i]) * c_amag;

          // set-up row for matrix
          const int rowR = indexR * totalNodes * nDim;
          double lhsFac = penaltyIp * c_amag;

          // sensitivities; current face (penalty); use general shape function
          // for this single ip
          for (int ic = 0; ic < currentNodesPerFace; ++ic) {
            const int icNdim = ws_c_face_node_ordinals[ic] * nDim;
            const double r = p_c_general_shape_function[ic];
            p_lhs[rowR + icNdim + i] -= r * lhsFac;
          }

          // sensitivities; opposing face (penalty); use general shape function
          // for this single ip
          for (int ic = 0; ic < opposingNodesPerFace; ++ic) {
            const int icNdim =
              (ws_o_face_node_ordinals[ic] + currentNodesPerElement) * nDim;
            const double r = p_o_general_shape_function[ic];
            p_lhs[rowR + icNdim + i] += r * lhsFac;
          }
        }
        apply_coeff(
          connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
