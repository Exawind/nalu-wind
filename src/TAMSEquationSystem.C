/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <AlgorithmDriver.h>
#include <ComputeTAMSAvgMdotEdgeAlgorithm.h>
#include <ComputeTAMSAvgMdotElemAlgorithm.h>
#include <ComputeMetricTensorNodeAlgorithm.h>
#include <ComputeSSTTAMSAveragesNodeAlgorithm.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <FieldFunctions.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <TAMSEquationSystem.h>
#include <TimeIntegrator.h>
#include <TurbViscSSTTAMSAlgorithm.h>

#include <SolverAlgorithmDriver.h>

// template for supp algs
#include <AlgTraits.h>
#include <kernel/KernelBuilder.h>
#include <kernel/KernelBuilderLog.h>

// stk_util
#include <stk_util/parallel/Parallel.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
//#include <stk_mesh/base/FieldParallel.hpp>
//#include <stk_mesh/base/GetBuckets.hpp>
//#include <stk_mesh/base/GetEntities.hpp>
//#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra {
namespace nalu {

TAMSEquationSystem::TAMSEquationSystem(EquationSystems& eqSystems)
  : EquationSystem(eqSystems, "TAMSEQS", "time_averaged_model_split"),
    managePNG_(realm_.get_consistent_mass_matrix_png("adaptivity_parameter")),
    avgVelocity_(NULL),
    avgDensity_(NULL),
    avgTkeResolved_(NULL),
    avgDudx_(NULL),
    metric_(NULL),
    alpha_(NULL),
    resAdequacy_(NULL),
    avgResAdequacy_(NULL),
    avgProduction_(NULL),
    avgTime_(NULL),
    avgMdotScs_(NULL),
    avgMdot_(NULL),
    metricTensorAlgDriver_(new AlgorithmDriver(realm_)),
    averagingAlgDriver_(new AlgorithmDriver(realm_)),
    avgMdotAlgDriver_(new AlgorithmDriver(realm_)),
    tviscAlgDriver_(new AlgorithmDriver(realm_)),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    resetTAMSAverages_(realm_.solutionOptions_->resetTAMSAverages_)
{
  // push back EQ to manager
  realm_.push_equation_to_systems(this);

  if (turbulenceModel_ != SST_TAMS) {
    throw std::runtime_error(
      "User has requested TAMSEqs, however, turbulence model has not been set "
      "to sst_tams, the only one supported by this equation system currently.");
  }
}

TAMSEquationSystem::~TAMSEquationSystem()
{
  if (NULL != metricTensorAlgDriver_)
    delete metricTensorAlgDriver_;
  if (NULL != averagingAlgDriver_)
    delete averagingAlgDriver_;
  if (NULL != avgMdotAlgDriver_)
    delete avgMdotAlgDriver_;
  if (NULL != tviscAlgDriver_)
    delete tviscAlgDriver_;
}

void
TAMSEquationSystem::register_nodal_fields(stk::mesh::Part* part)
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // register dof; set it as a restart variable
  alpha_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "k_ratio"));
  stk::mesh::put_field_on_mesh(*alpha_, *part, nullptr);

  avgVelocity_ = &(meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "average_velocity"));
  stk::mesh::put_field_on_mesh(*avgVelocity_, *part, nDim, nullptr);
  realm_.augment_restart_variable_list("average_velocity");

  avgDensity_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_density"));
  stk::mesh::put_field_on_mesh(*avgDensity_, *part, nullptr);
  realm_.augment_restart_variable_list("average_density");

  avgProduction_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_production"));
  stk::mesh::put_field_on_mesh(*avgProduction_, *part, nullptr);
  realm_.augment_restart_variable_list("average_production");

  avgDudx_ = &(meta_data.declare_field<GenericFieldType>(
    stk::topology::NODE_RANK, "average_dudx"));
  stk::mesh::put_field_on_mesh(*avgDudx_, *part, nDim * nDim, nullptr);
  realm_.augment_restart_variable_list("average_dudx");

  avgTkeResolved_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_tke_resolved"));
  stk::mesh::put_field_on_mesh(*avgTkeResolved_, *part, nullptr);
  realm_.augment_restart_variable_list("average_tke_resolved");

  avgTime_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_time"));
  stk::mesh::put_field_on_mesh(*avgTime_, *part, nullptr);

  metric_ = &(meta_data.declare_field<GenericFieldType>(
    stk::topology::NODE_RANK, "metric_tensor"));
  stk::mesh::put_field_on_mesh(*metric_, *part, nDim * nDim, nullptr);

  resAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*resAdequacy_, *part, nullptr);

  avgResAdequacy_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "avg_res_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*avgResAdequacy_, *part, nullptr);
  realm_.augment_restart_variable_list("avg_res_adequacy_parameter");
}

void
TAMSEquationSystem::register_element_fields(
  stk::mesh::Part* part, const stk::topology& theTopo)
{
  NaluEnv::self().naluOutputP0() << "Elemental Mdot average added in TAMS " << std::endl;

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  MasterElement* meSCS =
    sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
  const int numScsIp = meSCS->num_integration_points();

  avgMdotScs_ = &(meta_data.declare_field<GenericFieldType>(
    stk::topology::ELEMENT_RANK, "average_mass_flow_rate_scs"));
  stk::mesh::put_field_on_mesh(*avgMdotScs_, *part, numScsIp, nullptr);
  realm_.augment_restart_variable_list("average_mass_flow_rate_scs");
}

void
TAMSEquationSystem::register_edge_fields(stk::mesh::Part* part)
{
  NaluEnv::self().naluOutputP0() << "Edge Mdot average added in TAMS " << std::endl;

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  avgMdot_ = &(meta_data.declare_field<ScalarFieldType>(
    stk::topology::EDGE_RANK, "average_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*avgMdot_, *part, nullptr);
  realm_.augment_restart_variable_list("average_mass_flow_rate");
}

void
TAMSEquationSystem::register_interior_algorithm(stk::mesh::Part* part)
{

  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  // metric tensor algorithm
  if (NULL == metricTensorAlgDriver_)
    metricTensorAlgDriver_ = new AlgorithmDriver(realm_);

  std::map<AlgorithmType, Algorithm*>::iterator itmt =
    metricTensorAlgDriver_->algMap_.find(algType);

  if (itmt == metricTensorAlgDriver_->algMap_.end()) {
    ComputeMetricTensorNodeAlgorithm* metricTensorAlg =
      new ComputeMetricTensorNodeAlgorithm(realm_, part);
    metricTensorAlgDriver_->algMap_[algType] = metricTensorAlg;
  } else {
    itmt->second->partVec_.push_back(part);
  }

  // averaging algorithm
  if (NULL == averagingAlgDriver_)
    averagingAlgDriver_ = new AlgorithmDriver(realm_);

  std::map<AlgorithmType, Algorithm*>::iterator itav =
    averagingAlgDriver_->algMap_.find(algType);

  if (itav == averagingAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = NULL;
    switch (turbulenceModel_) {
    case SST_TAMS:
      theAlg = new ComputeSSTTAMSAveragesNodeAlgorithm(realm_, part);
      break;
    default:
      throw std::runtime_error("TAMSEquationSystem: non-supported turb model");
    }
    averagingAlgDriver_->algMap_[algType] = theAlg;
  } else {
    itav->second->partVec_.push_back(part);
  }

  // avgMdot algorithm
  if (NULL == avgMdotAlgDriver_)
    avgMdotAlgDriver_ = new AlgorithmDriver(realm_);

  if (realm_.realmUsesEdges_) {
    std::map<AlgorithmType, Algorithm*>::iterator itmd =
      avgMdotAlgDriver_->algMap_.find(algType);

    if (itmd == avgMdotAlgDriver_->algMap_.end()) {
      ComputeTAMSAvgMdotEdgeAlgorithm* avgMdotEdgeAlg =
        new ComputeTAMSAvgMdotEdgeAlgorithm(realm_, part);
      avgMdotAlgDriver_->algMap_[algType] = avgMdotEdgeAlg;
    } else {
      itmd->second->partVec_.push_back(part);
    }
  } else {
    std::map<AlgorithmType, Algorithm*>::iterator itmd =
      avgMdotAlgDriver_->algMap_.find(algType);

    if (itmd == avgMdotAlgDriver_->algMap_.end()) {
      ComputeTAMSAvgMdotElemAlgorithm* avgMdotAlg =
        new ComputeTAMSAvgMdotElemAlgorithm(realm_, part);
      avgMdotAlgDriver_->algMap_[algType] = avgMdotAlg;
    } else {
      itmd->second->partVec_.push_back(part);
    }
  }

  // FIXME: tvisc needed for TAMS update, but is updated in LowMach...
  //        Perhaps there is a way to call tvisc from LowMach here?
  std::map<AlgorithmType, Algorithm*>::iterator it_tv =
    tviscAlgDriver_->algMap_.find(algType);
  if (it_tv == tviscAlgDriver_->algMap_.end()) {
    Algorithm* theAlg = NULL;
    switch (realm_.solutionOptions_->turbulenceModel_) {
      case SST_TAMS:
        theAlg = new TurbViscSSTTAMSAlgorithm(realm_, part);
        break;
      default:
        throw std::runtime_error("non-supported turb model in TAMS Eq Sys");
    }
    tviscAlgDriver_->algMap_[algType] = theAlg;
  } else {
    it_tv->second->partVec_.push_back(part);
  }
}

void
TAMSEquationSystem::initial_work()
{
  compute_metric_tensor();

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // Initialize average_velocity, avg_dudx and avg_Prod
  // We don't want to do this on restart where TAMS fields are present
  if (resetTAMSAverages_) {
    const int nDim = meta_data.spatial_dimension();

    // Copy velocity to average velocity
    VectorFieldType &avgU = avgVelocity_->field_of_state(stk::mesh::StateNP1);
    const auto& U = *meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
    field_copy(realm_.meta_data(), realm_.bulk_data(), U, avgU, realm_.get_activate_aura());

    // Copy dudx to average dudx
    GenericFieldType &avgDudx = avgDudx_->field_of_state(stk::mesh::StateNP1);
    const auto& dudx = *meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
    field_copy(realm_.meta_data(), realm_.bulk_data(), dudx, avgDudx, realm_.get_activate_aura());

    ScalarFieldType* turbKinEne_ = meta_data.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "turbulent_ke");
    ScalarFieldType* tvisc_ = meta_data.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "turbulent_viscosity");
    ScalarFieldType& tkeNp1 = turbKinEne_->field_of_state(stk::mesh::StateNP1);

    // define some common selectors
    stk::mesh::Selector s_all_nodes =
      (meta_data.locally_owned_part() | meta_data.globally_shared_part()) &
      stk::mesh::selectField(*avgDudx_);

    stk::mesh::BucketVector const& buckets =
      realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
    for (stk::mesh::BucketVector::const_iterator ib = buckets.begin();
         ib != buckets.end(); ++ib) {
      stk::mesh::Bucket& b = **ib;
      const stk::mesh::Bucket::size_type length = b.size();

      double* tke = stk::mesh::field_data(tkeNp1, b);
      double* tvisc = stk::mesh::field_data(*tvisc_, b);
      double* avgProd = stk::mesh::field_data(*avgProduction_, b);
      double* rho = stk::mesh::field_data(*avgDensity_, b);

      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
        // Initialize average production to mean production
        const double* avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);
        std::vector<double> tij(nDim * nDim, 0.0);
        for (int i = 0; i < nDim; ++i) {
          for (int j = 0; j < nDim; ++j) {
            const double avgSij =
              0.5 * (avgDudx[i * nDim + j] + avgDudx[j * nDim + i]);
            tij[i * nDim + j] = 2.0 * tvisc[k] * avgSij;
          }
        }

        std::vector<double> Pij(nDim * nDim, 0.0);
        for (int i = 0; i < nDim; ++i) {
          for (int j = 0; j < nDim; ++j) {
            Pij[i * nDim + j] = 0.0;
            for (int m = 0; m < nDim; ++m) {
              Pij[i * nDim + j] += avgDudx[i * nDim + m] * tij[j * nDim + m] +
                                   avgDudx[j * nDim + m] * tij[i * nDim + m];
            }
            Pij[i * nDim + j] *= 0.5;
          }
        }

        double instProd = 0.0;
        for (int i = 0; i < nDim; ++i)
          instProd += Pij[i * nDim + i];

        avgProd[k] = instProd;
      }
    }
  }

  compute_averages();

  // FIXME: Moved this to SST Eqn Systems for now since mdot has not 
  //        been calculated during intial_work phase...
  //        Is that the best approach? Or would it be better to keep TAMS self-contained?
  //initialize_average_mdot();
  //compute_avgMdot();
}

void
TAMSEquationSystem::post_converged_work()
{
  // Compute TAMS terms here, since we only want to do so once per timestep

  // Need to update tvisc for use in computing averages
  tviscAlgDriver_->execute();

  // TODO: Assess consistency of this order of operations...
  compute_averages();

  compute_avgMdot();
}

void
TAMSEquationSystem::compute_metric_tensor()
{
  metricTensorAlgDriver_->execute();
}

void
TAMSEquationSystem::compute_averages()
{
  averagingAlgDriver_->execute();
}

void
TAMSEquationSystem::compute_avgMdot()
{
  avgMdotAlgDriver_->execute();
}

} // namespace nalu
} // namespace sierra
