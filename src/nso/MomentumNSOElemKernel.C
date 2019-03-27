/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "nso/MomentumNSOElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
MomentumNSOElemKernel<AlgTraits>::MomentumNSOElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  VectorFieldType* velocity,
  GenericFieldType* Gju,
  ScalarFieldType* viscosity,
  const double fourthFac,
  const double altResFac,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    viscosity_(viscosity),
    Gju_(Gju),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    fourthFac_(fourthFac),
    altResFac_(altResFac),
    om_altResFac_(1.0 - altResFac),
    includeDivU_(solnOpts.includeDivU_),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(velocity->name()))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  ScalarFieldType *density = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");

  velocityN_ = &(velocity->field_of_state(stk::mesh::StateN));
  velocityNp1_ = &(velocity->field_of_state(stk::mesh::StateNP1));
  if (velocity->number_of_states() == 2)
    velocityNm1_ = velocityN_;
  else
    velocityNm1_ = &(velocity->field_of_state(stk::mesh::StateNM1));

  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  if (density->number_of_states() == 2)
    densityNm1_ = densityN_;
  else
    densityNm1_ = &(density->field_of_state(stk::mesh::StateNM1));

  if (solnOpts.does_mesh_move())
    velocityRTM_ = metaData.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = metaData.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity");

  pressure_ = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "pressure");

  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);

  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields
  dataPreReqs.add_gathered_nodal_field(*Gju_, AlgTraits::nDim_, AlgTraits::nDim_);
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*velocityNm1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityN_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*velocityRTM_, AlgTraits::nDim_);

  dataPreReqs.add_gathered_nodal_field(*densityNm1_,1);
  dataPreReqs.add_gathered_nodal_field(*densityN_,1);
  dataPreReqs.add_gathered_nodal_field(*densityNp1_,1);
  dataPreReqs.add_gathered_nodal_field(*viscosity_,1);
  dataPreReqs.add_gathered_nodal_field(*pressure_,1);

  // master element data
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);

  dataPreReqs.add_master_element_call(SCS_GIJ, CURRENT_COORDINATES);

  // initialize kd
  for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
    for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
      v_kd_(i,j) = (i == j) ? 1.0 : 0.0;
    }
  }
}

template<typename AlgTraits>
void
MomentumNSOElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  dt_ = timeIntegrator.get_time_step();
  gamma1_ = timeIntegrator.get_gamma1();
  gamma2_ = timeIntegrator.get_gamma2();
  gamma3_ = timeIntegrator.get_gamma3(); // gamma3 may be zero
}

template<typename AlgTraits>
void
MomentumNSOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_dukdxScs   [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_rhoVrtmScs [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dpdxScs    [AlgTraits::nDim_];

  SharedMemView<DoubleType***>& v_Gju = scratchViews.get_scratch_view_3D(*Gju_);
  SharedMemView<DoubleType**>& v_uNm1 = scratchViews.get_scratch_view_2D(*velocityNm1_);
  SharedMemView<DoubleType**>& v_uN = scratchViews.get_scratch_view_2D(*velocityN_);
  SharedMemView<DoubleType**>& v_uNp1 = scratchViews.get_scratch_view_2D(*velocityNp1_);
  SharedMemView<DoubleType**>& v_velocityRTM = scratchViews.get_scratch_view_2D(*velocityRTM_);
  SharedMemView<DoubleType*>& v_rhoNm1 = scratchViews.get_scratch_view_1D(*densityNm1_);
  SharedMemView<DoubleType*>& v_rhoN = scratchViews.get_scratch_view_1D(*densityN_);
  SharedMemView<DoubleType*>& v_rhoNp1 = scratchViews.get_scratch_view_1D(*densityNp1_);
  SharedMemView<DoubleType*>& v_viscosity = scratchViews.get_scratch_view_1D(*viscosity_);
  SharedMemView<DoubleType*>& v_pressure = scratchViews.get_scratch_view_1D(*pressure_);

  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted
    : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;
  SharedMemView<DoubleType***>& v_gijUpper = scratchViews.get_me_views(CURRENT_COORDINATES).gijUpper;
  SharedMemView<DoubleType***>& v_gijLower = scratchViews.get_me_views(CURRENT_COORDINATES).gijLower;

  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // save off some offsets
    const int ilNdim = il*AlgTraits::nDim_;
    const int irNdim = ir*AlgTraits::nDim_;

    // zero out; scalars that prevail over all components
    DoubleType rhoNm1Scs = 0.0;
    DoubleType rhoNScs = 0.0;
    DoubleType rhoNp1Scs = 0.0;
    DoubleType dFdxCont = 0.0;
    DoubleType divU = 0.0;

    // zero out vectors that prevail over all components of k
    for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
      w_rhoVrtmScs[i] = 0.0;
      w_dpdxScs[i] = 0.0;
    }

    // determine scs values of interest
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      // save off shape function
      const DoubleType r = v_shape_function_(ip,ic);

      // time term, density
      rhoNm1Scs += r*v_rhoNm1(ic);
      rhoNScs += r*v_rhoN(ic);
      rhoNp1Scs += r*v_rhoNp1(ic);

      // compute scs derivatives and flux derivative
      const DoubleType pIC = v_pressure(ic);
      const DoubleType rhoIC = v_rhoNp1(ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType dnj = v_dndx(ip,ic,j);
        const DoubleType vrtmj = v_velocityRTM(ic,j);
        w_rhoVrtmScs[j] += r*rhoIC*vrtmj;
        divU += r*v_Gju(ic,j,j);
        dFdxCont += rhoIC*vrtmj*dnj;
        w_dpdxScs[j] += pIC*dnj;
      }
    }

    // full continuity residual (constant for all component k)
    const DoubleType contRes = (gamma1_*rhoNp1Scs + gamma2_*rhoNScs + gamma3_*rhoNm1Scs)/dt_ + dFdxCont;

    const double twoThirds = 2.0/3.0;
    // assemble each component
    for ( int k = 0; k < AlgTraits::nDim_; ++k ) {

      const int indexL = ilNdim + k;
      const int indexR = irNdim + k;

      // zero out residual_k and interpolated velocity_k to scs
      DoubleType dFdxkAdv = 0.0;
      DoubleType dFdxkDiff = 0.0;
      DoubleType ukNm1Scs = 0.0;
      DoubleType ukNScs = 0.0;
      DoubleType ukNp1Scs = 0.0;

      // zero out vector of local derivatives (duk/dxj)
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        w_dukdxScs[j] = 0.0;
      }

      // determine scs values of interest
      for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

        // save off shape function
        const DoubleType r = v_shape_function_(ip,ic);

        // save off velocityUnp1 for component k
        const DoubleType& ukNp1 = v_uNp1(ic,k);

        // interpolate all velocity states
        ukNm1Scs += r*v_uNm1(ic,k);
        ukNScs += r*v_uN(ic,k);
        ukNp1Scs += r*ukNp1;

        // compute scs derivatives and flux derivative (adv/diff)
        const DoubleType rhoIC = v_rhoNp1(ic);
        const DoubleType viscIC = v_viscosity(ic);
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
          const DoubleType dnj = v_dndx(ip,ic,j);
          w_dukdxScs[j] += ukNp1*dnj;
          dFdxkAdv += rhoIC*v_velocityRTM(ic,j)*ukNp1*dnj;
          dFdxkDiff += viscIC*(v_Gju(ic,k,j) + v_Gju(ic,j,k) - twoThirds*divU*v_kd_(k,j)*includeDivU_)*dnj;
        }
      }

      // compute residual for NSO; linearized first
      DoubleType residualAlt = dFdxkAdv - ukNp1Scs*dFdxCont;
      for ( int j = 0; j < AlgTraits::nDim_; ++j )
        residualAlt -= w_rhoVrtmScs[j]*w_dukdxScs[j];

      // compute residual for NSO; pde-based second
      const DoubleType time = (gamma1_*rhoNp1Scs*ukNp1Scs + gamma2_*rhoNScs*ukNScs + gamma3_*rhoNm1Scs*ukNm1Scs)/dt_;
      const DoubleType residualPde = time + dFdxkAdv - dFdxkDiff + w_dpdxScs[k] - contRes*ukNp1Scs*nonConservedForm_;

      // final form
      const DoubleType residual = residualAlt*altResFac_ + residualPde*om_altResFac_;

      // denominator for nu as well as terms for "upwind" nu
      DoubleType gUpperMagGradQ = 0.0;
      DoubleType rhoVrtmiGLowerRhoVrtmj = 0.0;
      for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
        const DoubleType duidxScs = w_dukdxScs[i];
        const DoubleType rhoVrtmi = w_rhoVrtmScs[i];
        for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
          gUpperMagGradQ += duidxScs*v_gijUpper(ip,i,j)*w_dukdxScs[j];
          rhoVrtmiGLowerRhoVrtmj += rhoVrtmi*v_gijLower(ip,i,j)*w_rhoVrtmScs[j];
        }
      }

      // construct nu from residual
      const DoubleType nuResidual = stk::math::sqrt((residual*residual)/(gUpperMagGradQ+small_));

      // construct nu from first-order-like approach; SNL-internal write-up (eq 209)
      // for now, only include advection as full set of terms is too diffuse
      const DoubleType nuFirstOrder = stk::math::sqrt(rhoVrtmiGLowerRhoVrtmj);

      // limit based on first order; Cupw_ is a fudge factor similar to Guermond's approach
      const DoubleType nu = stk::math::min(Cupw_*nuFirstOrder, nuResidual);

      DoubleType gijFac = 0.0;
      for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

        // save off shape function
        const DoubleType r = v_shape_function_(ip,ic);

        // find the row
        const int icNdim = ic*AlgTraits::nDim_;

        // save off some variables
        const DoubleType ukNp1 = v_uNp1(ic,k);

        // NSO diffusion-like term; -nu*gUpper*dQ/dxj*ai (residual below)
        DoubleType lhsfac = 0.0;
        for ( int i = 0; i < AlgTraits::nDim_; ++i ) {
          const DoubleType axi = v_scs_areav(ip,i);
          for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
            const DoubleType fac = v_gijUpper(ip,i,j)*v_dndx(ip,ic,j)*axi;
            const DoubleType facGj = r*v_gijUpper(ip,i,j)*v_Gju(ic,k,j)*axi;
            gijFac = stk::math::fmadd(fac,ukNp1,gijFac) - facGj*fourthFac_;
            lhsfac -= fac;
          }
        }

        // no coupling between components
        lhs(indexL,icNdim+k) = stk::math::fmadd(nu,lhsfac,lhs(indexL,icNdim+k));
        lhs(indexR,icNdim+k) -= nu*lhsfac;
      }

      // residual; left and right
      const DoubleType residualNSO = -nu*gijFac;
      rhs(indexL) -= residualNSO;
      rhs(indexR) += residualNSO;
    }
  }
}

INSTANTIATE_KERNEL(MomentumNSOElemKernel)

}  // nalu
}  // sierra
