/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "overset/AssembleOversetPressureAlgorithm.h"
#include "EquationSystem.h"
#include "FieldTypeDef.h"
#include "LinearSystem.h"
#include "Realm.h"
#include "SolverAlgorithm.h"
#include "TimeIntegrator.h"
#include "master_element/MasterElement.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Part.hpp"

#include <cmath>

namespace sierra{
namespace nalu{

AssembleOversetPressureAlgorithm::AssembleOversetPressureAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  stk::mesh::FieldBase *fieldQ)
  : OversetConstraintBase(realm, part, eqSystem, fieldQ)
{
  auto& meta = realm.meta_data();
  Udiag_ = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "momentum_diag");
}

void
AssembleOversetPressureAlgorithm::execute()
{
  // first thing to do is to zero out the row (lhs and rhs)
  prepare_constraints();

  // extract the rank
  const int theRank = NaluEnv::self().parallel_rank();

  stk::mesh::BulkData & bulkData = realm_.bulk_data();

  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double tauScale = gamma1 / dt;

  // space for LHS/RHS (nodesPerElem+1)*numDof*(nodesPerElem+1)*numDof; (nodesPerElem+1)*numDof
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // master element data
  std::vector <double > ws_general_shape_function;

  // interpolate nodal values to point-in-elem
  const int sizeOfDof = eqSystem_->linsys_->numDof();

  // size interpolated value
  std::vector<double> qNp1Orphan(sizeOfDof, 0.0);

  // Parallel communication of ghosted entities has been already handled in
  // EquationSystems::pre_iter_work

  // iterate oversetInfoVec_
  std::vector<OversetInfo *>::iterator ii;
  for( ii=realm_.oversetManager_->oversetInfoVec_.begin();
       ii!=realm_.oversetManager_->oversetInfoVec_.end(); ++ii ) {

    // overset info object of interest
    OversetInfo * infoObject = (*ii);

    // extract element and node mesh object
    stk::mesh::Entity owningElement = infoObject->owningElement_;
    stk::mesh::Entity orphanNode = infoObject->orphanNode_;

    // extract the owning rank for this node
    const int nodeRank = bulkData.parallel_owner_rank(orphanNode);

    // check to see if this node is locally owned by this rank; we only want to
    // process locally owned nodes, not shared
    if (theRank != nodeRank)
      continue;

    const double* dVol = stk::mesh::field_data(*dualNodalVolume_, orphanNode);
    const double* udiag = stk::mesh::field_data(*Udiag_, orphanNode);
    const double multFac = tauScale * std::cbrt(dVol[0]) / udiag[0];

    // get master element type for this contactInfo
    MasterElement *meSCS  = infoObject->meSCS_;
    const int nodesPerElement = meSCS->nodesPerElement_;
    std::vector <double > elemNodalQ(nodesPerElement*sizeOfDof);
    std::vector <double > shpfc(nodesPerElement);

    // resize some things; matrix related
    const int npePlusOne = nodesPerElement+1;
    const int lhsSize = npePlusOne*sizeOfDof*npePlusOne*sizeOfDof;
    const int rhsSize = npePlusOne*sizeOfDof;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(npePlusOne);

    // algorithm related; element
    ws_general_shape_function.resize(nodesPerElement);

    // pointer to lhs/rhs
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];

    // zeroing of lhs/rhs
    for ( int k = 0; k < lhsSize; ++k ) {
      p_lhs[k] = 0.0;
    }
    for ( int k = 0; k < rhsSize; ++k ) {
      p_rhs[k] = 0.0;
    }

    // extract nodal value for scalarQ
    const double *qNp1Nodal = (double *)stk::mesh::field_data(*fieldQ_, orphanNode);

    stk::mesh::Entity const* elem_node_rels = bulkData.begin_nodes(owningElement);
    const int num_nodes = bulkData.num_nodes(owningElement);

    // now load the elemental values for future interpolation; fill in connected
    // nodes; first connected node is orhpan
    connected_nodes[0] = orphanNode;
    for (int ni = 0; ni < num_nodes; ++ni) {
      stk::mesh::Entity node = elem_node_rels[ni];
      connected_nodes[ni + 1] = node;

      const double* qNp1 = (double*)stk::mesh::field_data(*fieldQ_, node);
      for (int i = 0; i < sizeOfDof; ++i) {
        elemNodalQ[i * nodesPerElement + ni] = qNp1[i];
      }
    }

    // interpolate dof to elemental ips (assigns qNp1)
    meSCS->interpolatePoint(
      sizeOfDof, &(infoObject->isoParCoords_[0]), &elemNodalQ[0],
      &qNp1Orphan[0]);

    // lhs; extract general shape function
    meSCS->general_shape_fcn(
      1, &(infoObject->isoParCoords_[0]), &ws_general_shape_function[0]);

    // rhs; orphan node is defined to be the zeroth connected node
    for (int i = 0; i < sizeOfDof; ++i) {
      const int rowOi = i * npePlusOne * sizeOfDof;
      const double residual = qNp1Nodal[i] - qNp1Orphan[i];
      p_rhs[i] = -residual * multFac;

      // row is zero by design (first connected node is the orphan node); assign
      // it fully
      p_lhs[rowOi + i] = multFac;

      for (int ic = 0; ic < nodesPerElement; ++ic) {
        const int indexR = i + sizeOfDof * (ic + 1);
        const int rOiR = rowOi + indexR;
        p_lhs[rOiR] -= ws_general_shape_function[ic] * multFac;
      }
    }

    // apply to linear system
    // don't use apply_coeff here as it checks for overset logic
    eqSystem_->linsys_->sumInto(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
  }
}

} // namespace nalu
} // namespace Sierra
