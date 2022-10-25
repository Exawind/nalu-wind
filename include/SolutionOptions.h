// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SolutionOptions_h
#define SolutionOptions_h

#include <Enums.h>

// standard c++
#include <string>
#include <map>
#include <utility>
#include <memory>
#include <vector>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

struct FixPressureAtNodeInfo;

enum ErrorIndicatorType {
  EIT_NONE = 0,
  EIT_PSTAB = 1 << 1,
  EIT_LIMITER = 1 << 2,
  EIT_SIMPLE_BASE = 1 << 3,
  EIT_SIMPLE_VORTICITY = EIT_SIMPLE_BASE + (1 << 4),
  EIT_SIMPLE_VORTICITY_DX = EIT_SIMPLE_BASE + (1 << 5),
  EIT_SIMPLE_DUDX2 = EIT_SIMPLE_BASE + (1 << 6)
};

/** Form of projection timescale used in pressure equation
 *
 */
enum ProjTScaleType {
  TSCALE_DEFAULT = 0,  //!< Original Nalu implementation
  TSCALE_UDIAGINV = 1, //!< 1/diag(A_p) implementation
  NUM_TSCALE_TYPES
};

class SolutionOptions
{
public:
  SolutionOptions();
  ~SolutionOptions();

  void load(const YAML::Node& node);
  void initialize_turbulence_constants();

  inline bool has_mesh_motion() const { return meshMotion_; }

  inline bool has_mesh_deformation() const { return externalMeshDeformation_; }

  inline bool does_mesh_move() const
  {
    return has_mesh_motion() | has_mesh_deformation();
  }

  inline std::string get_coordinates_name() const
  {
    return does_mesh_move() ? "current_coordinates" : "coordinates";
  }

  inline double get_mdot_interp() const
  {
    return mdotInterpRhoUTogether_ ? 1.0 : 0.0;
  }

  double get_alpha_factor(const std::string&) const;

  double get_alpha_upw_factor(const std::string&) const;

  double get_upw_factor(const std::string&) const;

  double get_relaxation_factor(const std::string&) const;

  bool primitive_uses_limiter(const std::string&) const;

  bool get_shifted_grad_op(const std::string&) const;

  bool get_skew_symmetric(const std::string&) const;

  std::vector<double> get_gravity_vector(const unsigned nDim) const;

  double get_turb_model_constant(TurbulenceModelConstant turbModelEnum) const;

  bool get_noc_usage(const std::string& dofName) const;

  bool has_set_boussinesq_time_scale();

  double hybridDefault_;
  double alphaDefault_;
  double alphaUpwDefault_;
  double upwDefault_;
  // Relaxation factors for equations
  double relaxFactorDefault_{1.0};
  double lamScDefault_;
  double turbScDefault_;
  double turbPrDefault_;
  bool nocDefault_;
  bool shiftedGradOpDefault_;
  bool skewSymmetricDefault_;
  std::string tanhFormDefault_;
  double tanhTransDefault_;
  double tanhWidthDefault_;
  double referenceDensity_;
  double referenceTemperature_;
  double thermalExpansionCoeff_;
  double stefanBoltzmann_;
  double nearestFaceEntrain_;
  double includeDivU_;
  bool mdotInterpRhoUTogether_;
  bool isTurbulent_;
  TurbulenceModel turbulenceModel_;
  bool meshMotion_;
  bool meshTransformation_;
  bool externalMeshDeformation_;
  bool ncAlgGaussLabatto_;
  bool ncAlgUpwindAdvection_;
  bool ncAlgIncludePstab_;
  bool ncAlgDetailedOutput_;
  bool ncAlgCoincidentNodesErrorCheck_;
  bool ncAlgCurrentNormal_;
  bool ncAlgPngPenalty_;
  bool cvfemShiftMdot_;
  bool cvfemReducedSensPoisson_;
  double inputVariablesRestorationTime_;
  bool inputVariablesInterpolateInTime_;
  double inputVariablesPeriodicTime_;
  bool consistentMMPngDefault_;
  bool useConsolidatedSolverAlg_;
  bool useConsolidatedBcSolverAlg_;
  bool eigenvaluePerturb_;
  double eigenvaluePerturbDelta_;
  int eigenvaluePerturbBiasTowards_;
  double eigenvaluePerturbTurbKe_;
  double earthAngularVelocity_;
  double latitude_;
  double raBoussinesqTimeScale_;
  double symmetryBcPenaltyFactor_;
  bool useStreletsUpwinding_;

  // global mdot correction alg
  bool activateOpenMdotCorrection_;
  double mdotAlgOpenCorrection_;
  bool explicitlyZeroOpenPressureGradient_;

  bool resetAMSAverages_;
  bool transition_model_;
  bool gammaEqActive_;
  bool lengthScaleLimiter_;
  double referenceVelocity_;
  double roughnessHeight_;
  bool RANSBelowKs_;

  // turbulence model coeffs
  std::map<TurbulenceModelConstant, double> turbModelConstantMap_;

  // numerics related
  std::map<std::string, double> hybridMap_;
  std::map<std::string, double> alphaMap_;
  std::map<std::string, double> alphaUpwMap_;
  std::map<std::string, double> upwMap_;
  std::map<std::string, double> relaxFactorMap_;
  std::map<std::string, bool> limiterMap_;
  std::map<std::string, std::string> tanhFormMap_;
  std::map<std::string, double> tanhTransMap_;
  std::map<std::string, double> tanhWidthMap_;
  std::map<std::string, bool> consistentMassMatrixPngMap_;
  std::map<std::string, bool> skewSymmetricMap_;

  // property related
  std::map<std::string, double> lamScMap_;
  std::map<std::string, double> lamPrMap_;
  std::map<std::string, double> turbScMap_;
  std::map<std::string, double> turbPrMap_;

  // source; nodal and fully integrated
  std::map<std::string, std::vector<std::string>> srcTermsMap_;
  std::map<std::string, std::vector<double>> srcTermParamMap_;
  std::map<std::string, std::vector<std::string>> elemSrcTermsMap_;
  std::map<std::string, std::vector<double>> elemSrcTermParamMap_;

  // nodal gradient
  std::map<std::string, std::string> nodalGradMap_;

  // non-orthogonal correction
  std::map<std::string, bool> nocMap_;

  // shifting of Laplace operator for the element-based grad_op
  std::map<std::string, bool> shiftedGradOpMap_;

  // read any fields from input files
  std::map<std::string, std::string> inputVarFromFileMap_;

  std::vector<double> gravity_;
  std::vector<double> bodyForce_;

  // Coriolis source term
  std::vector<double> eastVector_;
  std::vector<double> northVector_;

  //! Flag indicating whether the user has requested pressure referencing
  bool needPressureReference_{false};

  std::unique_ptr<FixPressureAtNodeInfo> fixPressureInfo_;

  ProjTScaleType tscaleType_{TSCALE_DEFAULT};

  // dynamic forcing parameters
  int dynamicBodyForceDir_{0};
  double dynamicBodyForceVelReference_{0.0};
  double dynamicBodyForceDenReference_{1.0};
  std::string dynamicBodyForceVelTarget_;
  std::vector<std::string> dynamicBodyForceDragTarget_;
  std::string dynamicBodyForceOutFile_;
  bool dynamicBodyForceBox_{false};

  std::string name_;

  bool newHO_;
};

} // namespace nalu
} // namespace sierra

#endif
