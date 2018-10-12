/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#if defined (NALU_USES_PERCEPT)

#include <Adapter.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <OutputInfo.h>
#include <SolutionOptions.h>

#include <stk_util/diag/Timer.hpp>

// adapt
#include <adapt/ElementRefinePredicate.hpp>
#include <adapt/TransitionElementAdapter.hpp>
#include <adapt/UniformRefinerPattern.hpp>
#include <adapt/UniformRefiner.hpp>
#include <adapt/IAdapter.hpp>
#include <adapt/AdaptedMeshVerifier.hpp>

// percept
#include <percept/PerceptMesh.hpp>

#include <ErrorIndicatorAlgorithmDriver.h>
#include <Simulation.h>

#define USE_NALU_PERFORMANCE_TESTING_CALLGRIND 0
#if USE_NALU_PERFORMANCE_TESTING_CALLGRIND
#include "/usr/netpub/valgrind-3.8.1/include/valgrind/callgrind.h"
#endif

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// Adapter - runs adapt
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Adapter::Adapter(
                 const Realm &realm)
  : realm_(realm), uniformRefinementPattern_(NULL), refinementPattern_(NULL), perceptMesh_(NULL), uniformBreaker_(NULL), breaker_(NULL),
    elementRefinePredicate_(NULL), selector_(NULL), adaptedMeshVerifier_(NULL)
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData * bulk_data = 0;
  if (meta_data.is_commit())
    throw std::logic_error("Adapter called with already-committed meta data");
  bool isCommitted = false;
  perceptMesh_ = new percept::PerceptMesh(&meta_data, bulk_data, isCommitted);

  // only output the active part, no parents... except, this only currently
  // works for volume elements, side sets are unaffected
  perceptMesh_->output_active_children_only(true);

  uniformRefinementPattern_ = new percept::URP_Heterogeneous_3D(*perceptMesh_);

  if (3 == meta_data.spatial_dimension()) {

    {
      TransitionElementType& transition_element       = perceptMesh_->get_fem_meta_data()->declare_field<TransitionElementType>(stk::topology::ELEMENT_RANK, "transition_element_3");
      stk::mesh::put_field_on_mesh( transition_element , perceptMesh_->get_fem_meta_data()->universal_part(), nullptr);
      stk::io::set_field_role(transition_element, Ioss::Field::TRANSIENT);
    }
    {
      TransitionElementType& transition_element       = perceptMesh_->get_fem_meta_data()->declare_field<TransitionElementType>(stk::topology::FACE_RANK, "transition_element");
      stk::mesh::put_field_on_mesh( transition_element , perceptMesh_->get_fem_meta_data()->universal_part(), nullptr);
      stk::io::set_field_role(transition_element, Ioss::Field::TRANSIENT);
    }

    refinementPattern_ = new percept::Local_Tet4_Tet4_N_HangingNode (*perceptMesh_);
  }
  else {
    
    {
      percept::TransitionElementType& transition_element = perceptMesh_->get_fem_meta_data()->declare_field<percept::TransitionElementType>(stk::topology::ELEMENT_RANK, "transition_element");
      stk::mesh::put_field_on_mesh( transition_element , perceptMesh_->get_fem_meta_data()->universal_part(), nullptr);
      stk::io::set_field_role(transition_element, Ioss::Field::TRANSIENT);
    }
    
    bool has_tri = false, has_quad = false;
    stk::mesh::PartVector pv = meta_data.get_parts();
    for (unsigned ii=0; ii < pv.size(); ++ii) {
      if (stk::mesh::is_auto_declared_part(*pv[ii]))
        continue;
      if (pv[ii]->topology().name() == "TRIANGLE_3_2D")
        has_tri = true;
      else if (pv[ii]->topology().name() == "QUADRILATERAL_4_2D")
        has_quad = true;
    }
    
    if (has_quad && !has_tri)
      refinementPattern_ = new percept::Local_Quad4_Quad4_N_Transition(*perceptMesh_);
    else if (has_tri && !has_quad)
      refinementPattern_ = new percept::Local_Tri3_Tri3_N_HangingNode (*perceptMesh_);
    else
      throw std::runtime_error("sorry, hybrid initial mesh of tri and quad not yet supported");
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Adapter::~Adapter()
{
  delete uniformRefinementPattern_;
  delete refinementPattern_;
  delete perceptMesh_;
  delete uniformBreaker_;
  delete breaker_;
  delete elementRefinePredicate_;
  delete selector_;
  if ( NULL != adaptedMeshVerifier_)
    delete adaptedMeshVerifier_;
}

void
Adapter::do_uniform_refine()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  if (NULL == perceptMesh_->get_bulk_data()) {
    perceptMesh_->set_bulk_data(&bulk_data);
  }

  if (uniformBreaker_ == NULL) {
    uniformBreaker_ = new percept::UniformRefiner(*perceptMesh_, *uniformRefinementPattern_, 0);
  }

  uniformBreaker_->doBreak();

  // this int field is not prolonged so need to set here
  setNaluGlobalId();

  if (realm_.solutionOptions_->uniformRefineSaveAfter_) {
    static int fileid = 0;
    std::ostringstream fileid_ss;
    fileid_ss << std::setfill('0') << std::setw(4) << (fileid);

    std::string oname = realm_.outputInfo_->outputDBName_ +"-after-uniform-refine.e";
    if (fileid > 0) oname += "-s" + fileid_ss.str();
    perceptMesh_->save_as(oname);
  }
}

//--------------------------------------------------------------------------
//-------- do_adapt --------------------------------------------------------
//--------------------------------------------------------------------------
void
Adapter::do_adapt(int what_to_do)
{
#if USE_NALU_PERFORMANCE_TESTING_CALLGRIND
  CALLGRIND_START_INSTRUMENTATION;
  CALLGRIND_TOGGLE_COLLECT;
#endif

  bool do_dump_seq = false;

  static stk::diag::Timer timerAdapt_("Adapt", Simulation::rootTimer());
  static stk::diag::Timer timerRefine_("Refine", timerAdapt_);
  static stk::diag::Timer timerUnrefine_("Unrefine", timerAdapt_);
  static stk::diag::Timer timerVerify_("AdaptedMeshVerifier", timerAdapt_);

  stk::diag::TimeBlock tbTimerAdapt_(timerAdapt_);

  stk::mesh::MetaData & meta_data = realm_.meta_data();
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();

  if (NULL == perceptMesh_->get_bulk_data()) {
    perceptMesh_->set_bulk_data(&bulk_data);
  }

  bool doMeshVerification = realm_.debug();
  if (doMeshVerification) {
    const int timeStepCount = realm_.get_time_step_count();
    NaluEnv::self().naluOutputP0() << "Adapt: verifier, timeStepCount= " << timeStepCount << std::endl;

    if (NULL == adaptedMeshVerifier_) {
      stk::diag::TimeBlock tbTimerVerify_(timerVerify_);
      adaptedMeshVerifier_ = new percept::AdaptedMeshVerifier(0);
      if (!adaptedMeshVerifier_->isValid(*perceptMesh_, true))
        throw std::runtime_error("Invalid initial mesh");
    }
  }

  RefineFieldType *refine_field = meta_data.get_field<RefineFieldType>(stk::topology::ELEMENT_RANK, "refine_field");
  if (!refine_field)
    throw std::logic_error("refine field not present but adaptivity was requested");

  if (selector_ == NULL) {
    selector_ = new stk::mesh::Selector();
    *selector_ = stk::mesh::selectField(*refine_field);
  }

  double tolerance = 0.0; // not used
  if (elementRefinePredicate_ == NULL) {
    elementRefinePredicate_ = new percept::ElementRefinePredicate(*perceptMesh_, selector_, refine_field, tolerance);
  }

  stk::mesh::FieldBase* proc_rank_field = 0;  //not used
  if (breaker_ == NULL) {
    {
      breaker_ = new percept::TransitionElementAdapter<percept::ElementRefinePredicate>
        (*elementRefinePredicate_, *perceptMesh_, *refinementPattern_, proc_rank_field);
    }
    breaker_->setRemoveOldElements(false);
    breaker_->setAlwaysInitializeNodeRegistry(false);
    breaker_->setAlternateRootTimer(&timerAdapt_);
    if (realm_.realmUsesEdges_) {
      stk::mesh::PartVector excludeParts;
      excludeParts.push_back(realm_.edgesPart_);
      breaker_->setExcludeParts(excludeParts);
    }
  }

  const bool extra_output = realm_.solutionOptions_->adapterExtraOutput_;
  static int fileid = 0;
  std::ostringstream fileid_ss;
  static int fid = 0;
  fileid_ss << std::setfill('0') << std::setw(4) << (fileid+1);

  if (what_to_do & ADAPT_REFINE) {
    stk::diag::TimeBlock tbTimerRefine_(timerRefine_);

    if (1 && extra_output) {
      std::string oname = realm_.outputInfo_->outputDBName_ +"-before-adapt.e";
      if (fileid > 0) oname += "-s" + fileid_ss.str();
      perceptMesh_->save_as(oname);
    }

    if (do_dump_seq && fid == 0 && extra_output) {
      std::string oname = realm_.outputInfo_->outputDBName_ +"-seq.e";
      std::ostringstream fid_ss;
      fid_ss << std::setfill('0') << std::setw(4) << (fid+1);
      if (fid > 0) oname += "-s" + fid_ss.str();
      perceptMesh_->save_as(oname);
      ++fid;
    }

    {
      stk::diag::TimeBlock tbTimerVerify_(timerVerify_);
      if (adaptedMeshVerifier_ && !adaptedMeshVerifier_->isValid(*perceptMesh_, false))
        throw std::runtime_error("Invalid mesh before refine");
    }

    {
      // {node-, edge-, face-neighors}
      bool enforce_what[3] = {false, true, false};
      breaker_->refine( enforce_what);
    }

    if (do_dump_seq &&  extra_output) {
      std::string oname = realm_.outputInfo_->outputDBName_ +"-seq.e";
      std::ostringstream fid_ss;
      fid_ss << std::setfill('0') << std::setw(4) << (fid+1);
      if (fid > 0) oname += "-s" + fid_ss.str();
      perceptMesh_->save_as(oname);
      fid++;
    }

    {
      stk::diag::TimeBlock tbTimerVerify_(timerVerify_);
      if (adaptedMeshVerifier_ && !adaptedMeshVerifier_->isValid(*perceptMesh_, false))
        throw std::runtime_error("Invalid mesh after refine");
    }
  }

  if (what_to_do & ADAPT_UNREFINE) {
    stk::diag::TimeBlock tbTimerUnrefine_(timerUnrefine_);

    if (1 && extra_output) {
      std::string oname = realm_.outputInfo_->outputDBName_ +"-before-unref.e";
      if (fileid > 0) oname += "-s" + fileid_ss.str();
      perceptMesh_->save_as(oname);
    }

    if (do_dump_seq &&  extra_output) {
      std::string oname = realm_.outputInfo_->outputDBName_ +"-seq.e";
      std::ostringstream fid_ss;
      fid_ss << std::setfill('0') << std::setw(4) << (fid+1);
      if (fid > 0) oname += "-s" + fid_ss.str();
      perceptMesh_->save_as(oname);
      fid++;
    }
    
    {
      bool enforce_what[3] = {false, true, false};
      breaker_->unrefine( enforce_what);
    }
    
    if (do_dump_seq &&  extra_output) {
      std::string oname = realm_.outputInfo_->outputDBName_ +"-seq.e";
      std::ostringstream fid_ss;
      fid_ss << std::setfill('0') << std::setw(4) << (fid+1);
      if (fid > 0) oname += "-s" + fid_ss.str();
      perceptMesh_->save_as(oname);
      fid++;
    }
    
    if (1 && extra_output) {
      std::string oname = realm_.outputInfo_->outputDBName_ +"-after-adapt.e";
      if (fileid > 0) oname += "-s" + fileid_ss.str();
      perceptMesh_->save_as(oname);
      ++fileid;
    }

    {
      stk::diag::TimeBlock tbTimerVerify_(timerVerify_);
      if (adaptedMeshVerifier_ && !adaptedMeshVerifier_->isValid(*perceptMesh_, false))
        throw std::runtime_error("Invalid mesh after unrefine");
    }

  }


  // this int field is not prolonged so need to set here
  setNaluGlobalId();

#if USE_NALU_PERFORMANCE_TESTING_CALLGRIND
  CALLGRIND_TOGGLE_COLLECT;
  CALLGRIND_STOP_INSTRUMENTATION;
#endif


}

void 
Adapter::setNaluGlobalId()
{
  // set integer field naluGlobalId_ (adapt doesn't prolongate integer fields,
  //   it really isn't a well-defined process, so the app code needs to do it)

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::BucketVector buckets = bulk_data.buckets(stk::topology::NODE_RANK);
  
  for ( std::vector<stk::mesh::Bucket*>::const_iterator ib = buckets.begin() ;
        ib != buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::EntityId *id = stk::mesh::field_data(*realm_.naluGlobalId_, b[k]);
      stk::mesh::EntityId id0 = bulk_data.identifier(b[k]);
      *id = id0;
    }
  }
}

} // namespace nalu
} // namespace Sierra

#endif // NALU_USES_PERCEPT
