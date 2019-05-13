/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "pmr/RadTransAdvectionSUCVElemKernel.h"
#include "pmr/RadiativeTransportEquationSystem.h"
#include "AlgTraits.h"
#include "Enums.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"

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
RadTransAdvectionSUCVElemKernel<AlgTraits>::RadTransAdvectionSUCVElemKernel(
  const stk::mesh::BulkData& bulkData,
  RadiativeTransportEquationSystem *radEqSystem,
  const double sucvFac,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    radEqSystem_(radEqSystem),
    sucvFac_(sucvFac),
    useEdgeH_(true),
    invFourPi_(1.0/std::acos(-1.0)/4.0),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes())
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  intensity_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "intensity");
  absorption_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "absorption_coefficient");
  scattering_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scattering_coefficient");
  scalarFlux_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_flux");
  radiationSource_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "radiation_source");
  dualNodalVolume_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  
  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  // compute and save shape function
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS);
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // required fields
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(*intensity_, 1);
  dataPreReqs.add_gathered_nodal_field(*absorption_, 1);
  dataPreReqs.add_gathered_nodal_field(*scattering_, 1);
  dataPreReqs.add_gathered_nodal_field(*scalarFlux_, 1);
  dataPreReqs.add_gathered_nodal_field(*radiationSource_, 1);
  dataPreReqs.add_gathered_nodal_field(*dualNodalVolume_, 1);

  // master element 
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);

  NaluEnv::self().naluOutputP0() << "sucvFac_ is: " << sucvFac_ << std::endl;
}

template<typename AlgTraits>
RadTransAdvectionSUCVElemKernel<AlgTraits>::~RadTransAdvectionSUCVElemKernel()
{}


template<typename AlgTraits>
void
RadTransAdvectionSUCVElemKernel<AlgTraits>::setup(const TimeIntegrator& /*timeIntegrator*/)
{
  // extract ordinate direction and copy to Double type
  std::vector<double> Sk(AlgTraits::nDim_,0.0);
  radEqSystem_->get_current_ordinate(&Sk[0]);
  for ( int j = 0; j < AlgTraits::nDim_; ++j )
    v_Sk_(j) = Sk[j];
}

template<typename AlgTraits>
void
RadTransAdvectionSUCVElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>&lhs,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType**>& v_coordinates = scratchViews.get_scratch_view_2D(*coordinates_);
  SharedMemView<DoubleType*>& v_intensity = scratchViews.get_scratch_view_1D(
    *intensity_);
  SharedMemView<DoubleType*>& v_absorption = scratchViews.get_scratch_view_1D(
    *absorption_);
  SharedMemView<DoubleType*>& v_scattering = scratchViews.get_scratch_view_1D(
    *scattering_);
  SharedMemView<DoubleType*>& v_scalarFlux = scratchViews.get_scratch_view_1D(
    *scalarFlux_);
  SharedMemView<DoubleType*>& v_radiationSource = scratchViews.get_scratch_view_1D(
    *radiationSource_);
  SharedMemView<DoubleType*>& v_dualNodalVolume = scratchViews.get_scratch_view_1D(
    *dualNodalVolume_);

  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = scratchViews.get_me_views(CURRENT_COORDINATES).dndx;

  for (int ip=0; ip < AlgTraits::numScsIp_; ++ip) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // form sj*njdS (part of the lhs for central term; I*sj*njdS)
    DoubleType sjaj = 0.0;
    DoubleType asq = 0.0;
    for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
      const DoubleType aj = v_scs_areav(ip,j);
      sjaj += v_Sk_(j)*aj;
      asq += aj*aj;
    }
    const DoubleType aMag = stk::math::sqrt(asq);

    // integration point interpolation
    DoubleType Iscs = 0.0;
    DoubleType extCoeffscs = 0.0;
    DoubleType ePscs = 0.0;
    DoubleType isotropicScatterscs = 0.0;
    DoubleType dualNodalVscs = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);
          
      // save of some variables
      const DoubleType I = v_intensity(ic);
      const DoubleType mua = v_absorption(ic);
      const DoubleType mus = v_scattering(ic);
      
      // interpolation to scs
      Iscs += r*I;
      extCoeffscs += r*(mua+mus);
      ePscs += r*v_radiationSource(ic);
      isotropicScatterscs += r*mus*v_scalarFlux(ic)*invFourPi_;
      dualNodalVscs += r*v_dualNodalVolume(ic);
      
      // assemble I*sj*njdS to lhs; left/right
      lhs(il,ic) += sjaj*r;
      lhs(ir,ic) -= sjaj*r;
    }

    // rhs residual for I*sj*njdS
    rhs(il) -= Iscs*sjaj;
    rhs(ir) += Iscs*sjaj;

    // now work on SUCV stabilization terms; needed tau, hence second ic loop
    DoubleType h_edge = 0.0;
    for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
      const DoubleType nj = v_scs_areav(ip,j)/aMag;
      const DoubleType dxj = v_coordinates(ir,j) - v_coordinates(il,j);
      h_edge += nj*dxj;
    }

    // alternative h
    const DoubleType h_vol = (AlgTraits::nDim_ == 2 ) ? stk::math::sqrt(dualNodalVscs) : stk::math::cbrt(dualNodalVscs);

    // form tau
    const DoubleType h = useEdgeH_ ? h_edge : h_vol;
    const DoubleType tau = stk::math::sqrt(1.0/((2.0/h)*(2.0/h) + extCoeffscs*extCoeffscs))*sucvFac_;
	
    DoubleType sidIdxi = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);

      // save of some variables
      const DoubleType I = v_intensity(ic);
          
      // SUCV -tau*sj*aj*(mua+mus)*I term; left/right (residual below)
      lhs(il,ic) += -tau*sjaj*r*extCoeffscs;
      lhs(ir,ic) -= -tau*sjaj*r*extCoeffscs;
	  
      // SUCV diffusion-like term; -tau*si*dI/dxi*sjaj (residual below)
      DoubleType lhsfac = 0.0;
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        const DoubleType sjdNj = v_Sk_(j)*v_dndx(ip,ic,j);
        sidIdxi += sjdNj*I;
        lhsfac += -sjdNj;
      }
      lhs(il,ic) += tau*sjaj*lhsfac;
      lhs(ir,ic) -= tau*sjaj*lhsfac;	  
    }
    
    // full sucv residual
    const DoubleType residual = -tau*sjaj*(sidIdxi + extCoeffscs*Iscs - ePscs - isotropicScatterscs);
    
    // residual; left and right
    rhs(il) -= residual;
    rhs(ir) += residual; 
  }
}

INSTANTIATE_KERNEL(RadTransAdvectionSUCVElemKernel)

}  // nalu
}  // sierra
