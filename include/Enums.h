// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Enums_h
#define Enums_h

#include <string>
#include <map>

namespace sierra {
namespace nalu {

enum AlgorithmType {
  INTERIOR = 0,
  BOUNDARY,
  INFLOW,
  WALL,
  WALL_FCN,
  OPEN,
  MASS,
  SRC,
  SYMMETRY,
  WALL_HF,
  WALL_CHT,
  WALL_RAD,
  NON_CONFORMAL,
  ELEM_SOURCE,
  OVERSET,
  WALL_ABL,

  TOP_ABL,

  /** Set the reference pressure at a node.
   *
   *  Used only for continuity equation system. This needs to be the last
   *  algorithm applied to the linear system because it resets the row and
   *  overwrites contributions from other algorithms at this node.
   *
   * \sa FixPressureAtNodeAlgorithm
   */

  REF_PRESSURE,
  X_SYM_STRONG,
  Y_SYM_STRONG,
  Z_SYM_STRONG
};

enum BoundaryConditionType {
  INFLOW_BC = 1,
  OPEN_BC = 2,
  WALL_BC = 3,
  SYMMETRY_BC = 4,
  PERIODIC_BC = 5,
  NON_CONFORMAL_BC = 6,
  OVERSET_BC = 7,
  ABLTOP_BC = 8
};

enum EquationType {
  EQ_MOMENTUM = 0,
  EQ_CONTINUITY = 1,
  EQ_MIXTURE_FRACTION = 2,
  EQ_TURBULENT_KE = 3,
  EQ_TEMPERATURE = 4,
  EQ_INTENSITY = 5,
  EQ_ENTHALPY = 6,
  EQ_MESH_DISPLACEMENT = 7,
  EQ_SPEC_DISS_RATE = 8,
  EQ_TOT_DISS_RATE = 9,
  EQ_MASS_FRACTION = 10,
  EQ_PNG = 11,
  EQ_PNG_P = 12,
  EQ_PNG_Z = 13,
  EQ_PNG_H = 14,
  EQ_PNG_U = 15,
  EQ_PNG_TKE = 16, // FIXME... Last PNG managed like this..
  EQ_WALL_DISTANCE = 17,
  EQ_GAMMA_TRANS = 18,
  EQ_VOLUME_OF_FLUID = 19,
  EquationSystemType_END
};

static const std::string EquationTypeMap[] = {
  "Momentum",
  "Continuity",
  "Mixture_Fraction",
  "Turbulent_KE",
  "Temperature",
  "Intensity",
  "Enthalpy",
  "MeshVelocity",
  "Specific_Dissipation_Rate",
  "Total_Dissipation_Rate",
  "Mass_Fraction",
  "PNG",
  "PNG_P",
  "PNG_Z",
  "PNG_H",
  "PNG_U",
  "PNG_TKE",
  "Wall_Distance",
  "Volume_of_Fluid"};

enum UserDataType {
  CONSTANT_UD = 0,
  FUNCTION_UD = 1,
  USER_SUB_UD = 2,
  STRING_FUNCTION_UD = 3,
  UserDataType_END
};

// prop enum and name below
enum PropertyIdentifier {
  DENSITY_ID = 0,
  VISCOSITY_ID = 1,
  SPEC_HEAT_ID = 2,
  THERMAL_COND_ID = 3,
  ABSORBTION_COEFF_ID = 4,
  ENTHALPY_ID = 5,
  LAME_MU_ID = 6,
  LAME_LAMBDA_ID = 7,
  SCATTERING_COEFF_ID = 8,
  PropertyIdentifier_END
};

static const std::string PropertyIdentifierNames[] = {
  "density",
  "viscosity",
  "specific_heat",
  "thermal_conductivity",
  "absorption_coefficient",
  "enthalpy",
  "lame_mu",
  "lame_lambda",
  "scattering_coefficient"};

// prop enum and name below
enum MaterialPropertyType {
  CONSTANT_MAT = 0,
  MIXFRAC_MAT = 1,
  POLYNOMIAL_MAT = 2,
  IDEAL_GAS_T_MAT = 3,
  GEOMETRIC_MAT = 4,
  IDEAL_GAS_T_P_MAT = 5,
  HDF5_TABLE_MAT = 6,
  IDEAL_GAS_YK_MAT = 7,
  GENERIC = 8,
  VOF_MAT = 9,
  MaterialPropertyType_END
};

enum NaluState { NALU_STATE_N = 0, NALU_STATE_NM1 = 1 };

enum class TurbulenceModel {
  LAMINAR,
  KSGS,
  SMAGORINSKY,
  WALE,
  SST,
  SSTLR,
  SST_DES,
  SST_AMS,
  SST_IDDES,
  KE,
  KO,
  TurbulenceModel_END
};

// matching string name index into above enums (must match PERFECTLY)
static const std::string TurbulenceModelNames[] = {
  "laminar", "ksgs",    "smagorinsky", "wale", "sst", "sstlr",
  "sst_des", "sst_ams", "sst_iddes",   "ke",   "ko"};

enum TurbulenceModelConstant {
  TM_cMu = 0,
  TM_kappa = 1,
  TM_cDESke = 2,
  TM_cDESkw = 3,
  TM_tkeProdLimitRatio = 4,
  TM_cmuEps = 5,
  TM_cEps = 6,
  TM_betaStar = 7,
  TM_aOne = 8,
  TM_betaOne = 9,
  TM_betaTwo = 10,
  TM_gammaOne = 11,
  TM_gammaTwo = 12,
  TM_sigmaKOne = 13,
  TM_sigmaKTwo = 14,
  TM_sigmaWOne = 15,
  TM_sigmaWTwo = 16,
  TM_cmuCs = 17,
  TM_Cw = 18,
  TM_CbTwo = 19,
  TM_SDRWallFactor = 20,
  TM_zCV = 21,
  TM_ci = 22,
  TM_elog = 23,
  TM_yplus_crit = 24,
  TM_CMdeg = 25,
  TM_forCl = 26,
  TM_forCeta = 27,
  TM_forCt = 28,
  TM_forBlT = 29,
  TM_forBlKol = 30,
  TM_forFac = 31,
  TM_v2cMu = 32,
  TM_aspRatSwitch = 33,
  TM_periodicForcingLengthX = 34,
  TM_periodicForcingLengthY = 35,
  TM_periodicForcingLengthZ = 36,
  TM_sigmaMax = 37,
  TM_ch1 = 38,
  TM_ch2 = 39,
  TM_ch3 = 40,
  TM_tau_des = 41,
  TM_iddes_Cw = 42,
  TM_iddes_Cdt1 = 43,
  TM_iddes_Cdt2 = 44,
  TM_iddes_Cl = 45,
  TM_iddes_Ct = 46,
  TM_abl_bndtw = 47,
  TM_abl_deltandtw = 48,
  TM_abl_sigma = 49,
  TM_ams_peclet_offset = 50,
  TM_ams_peclet_slope = 51,
  TM_ams_peclet_scale = 52,
  TM_fMuExp = 53,
  TM_utau = 54,
  TM_cEpsOne = 55,
  TM_cEpsTwo = 56,
  TM_fOne = 57,
  TM_sigmaK = 58,
  TM_sigmaEps = 59,
  TM_sstLRDestruct = 60,
  TM_sstLRProd = 61,
  TM_tkeAmb = 62,
  TM_sdrAmb = 63,
  TM_avgTimeCoeff = 64,
  TM_alphaInf = 65,
  TM_caOne = 66,
  TM_caTwo = 67,
  TM_ceOne = 68,
  TM_ceTwo = 69,
  TM_c0t = 70,
  TM_END = 71
};

static const std::string TurbulenceModelConstantNames[] = {
  "cMu",
  "kappa",
  "cDESke",
  "cDESkw",
  "tkeProdLimitRatio",
  "cmuEps",
  "cEps",
  "betaStar",
  "aOne",
  "betaOne",
  "betaTwo",
  "gammaOne",
  "gammaTwo",
  "sigmaKOne",
  "sigmaKTwo",
  "sigmaWOne",
  "sigmaWTwo",
  "cmuCs",
  "Cw",
  "Cb2",
  "SDRWallFactor",
  "Z_CV",
  "ci",
  "Elog",
  "yplus_crit",
  "CMdeg",
  "forcingCl",
  "forcingCeta",
  "forcingCt",
  "forcingBlT",
  "forcingBlKol",
  "forcingFactor",
  "v2cMu",
  "aspectRatioSwitch",
  "periodicForcingLengthX",
  "periodicForcingLengthY",
  "periodicForcingLengthZ",
  "sigmaMax",
  "ch1",
  "ch2",
  "ch3",
  "tau_des",
  "iddes_Cw",
  "iddes_Cdt1",
  "iddes_Cdt2",
  "iddes_Cl",
  "iddes_Ct",
  "abl_bndtw",
  "abl_deltandtw",
  "abl_sigma",
  "ams_peclet_offset",
  "ams_peclet_slope",
  "ams_peclet_scale",
  "fMuExp",
  "utau",
  "cEpsOne",
  "cEpsTwo",
  "fOne",
  "sigmaK",
  "sigmaEps",
  "sstLRDestruct",
  "sstLRProd",
  "tke_amb",
  "sdr_amb",
  "avgTimeCoeff",
  "alphaInf",
  "caOne",
  "caTwo",
  "ceOne",
  "ceTwo",
  "c0t",
  "END"};

enum ActuatorType {
  ActLinePointDrag = 0,
  ActLineFAST = 1,
  ActLineFASTNGP = 2,
  AdvActLineFASTNGP = 3,
  ActDiskFAST = 4,
  ActDiskFASTNGP = 5,
  ActLineSimple = 6,
  ActLineSimpleNGP = 7,
  ActuatorType_END
};

static std::map<std::string, ActuatorType> ActuatorTypeMap = {
  {"ActLinePointDrag", ActuatorType::ActLinePointDrag},
  {"ActLineFAST", ActuatorType::ActLineFAST},
  {"ActDiskFAST", ActuatorType::ActDiskFAST},
  {"ActLineSimple", ActuatorType::ActLineSimple},
  {"ActLineSimpleNGP", ActuatorType::ActLineSimpleNGP},
  {"ActLineFASTNGP", ActuatorType::ActLineFASTNGP},
  {"AdvActLineFASTNGP", ActuatorType::AdvActLineFASTNGP},
  {"ActDiskFASTNGP", ActuatorType::ActDiskFASTNGP}};

enum class EntrainmentMethod { SPECIFIED = 0, COMPUTED = 1, CLASSIC = 2 };

static std::map<std::string, EntrainmentMethod> EntrainmentMethodMap{
  {"computed", EntrainmentMethod::COMPUTED},
  {"specified", EntrainmentMethod::SPECIFIED}};

} // namespace nalu
} // namespace sierra

#endif
