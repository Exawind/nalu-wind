/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Enums_h
#define Enums_h

#include <string>
#include <map>

namespace sierra {
namespace nalu {

enum AlgorithmType{
  INTERIOR  = 0,
  INFLOW    = 1,
  WALL      = 2,
  WALL_FCN  = 3,
  OPEN      = 4,
  MASS      = 5,
  SRC       = 6,
  SYMMETRY  = 7,
  WALL_HF   = 8,
  WALL_CHT  = 9,
  WALL_RAD  = 10,
  NON_CONFORMAL = 11,
  ELEM_SOURCE = 12,
  OVERSET = 13,
  WALL_ABL = 14,

  /** Set the reference pressure at a node.
   *
   *  Used only for continuity equation system. This needs to be the last
   *  algorithm applied to the linear system because it resets the row and
   *  overwrites contributions from other algorithms at this node.
   *
   * \sa FixPressureAtNodeAlgorithm
   */
  REF_PRESSURE = 15, 
 
  TOP_ABL = 16
};

enum BoundaryConditionType{
  INFLOW_BC    = 1,
  OPEN_BC      = 2,
  WALL_BC      = 3,
  SYMMETRY_BC  = 4,
  PERIODIC_BC  = 5,
  NON_CONFORMAL_BC = 6,
  OVERSET_BC = 7,
  ABLTOP_BC  = 8
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
  EQ_MASS_FRACTION = 9,
  EQ_PNG   = 10,
  EQ_PNG_P = 11,
  EQ_PNG_Z = 12,
  EQ_PNG_H = 13,
  EQ_PNG_U = 14,
  EQ_PNG_TKE = 15, // FIXME... Last PNG managed like this..
  EQ_WALL_DISTANCE = 16,
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
  "Mass_Fraction",
  "PNG",
  "PNG_P",
  "PNG_Z",
  "PNG_H",
  "PNG_U",
  "PNG_TKE",
  "Wall_Distance"
};

enum UserDataType {
  CONSTANT_UD = 0,
  FUNCTION_UD = 1,
  USER_SUB_UD = 2,
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
enum  MaterialPropertyType {
  CONSTANT_MAT = 0,
  MIXFRAC_MAT = 1,
  POLYNOMIAL_MAT = 2,
  IDEAL_GAS_T_MAT = 3,
  GEOMETRIC_MAT = 4,
  IDEAL_GAS_T_P_MAT = 5,
  HDF5_TABLE_MAT = 6,
  IDEAL_GAS_YK_MAT = 7,
  GENERIC = 8,
  MaterialPropertyType_END
};

enum NaluState {
  NALU_STATE_N = 0,
  NALU_STATE_NM1 = 1
};

enum TurbulenceModel {
  LAMINAR = 0,
  KSGS = 1,
  SMAGORINSKY = 2,
  WALE = 3,
  SST = 4,
  SST_DES = 5,
  TurbulenceModel_END
};  

// matching string name index into above enums (must match PERFECTLY)
static const std::string TurbulenceModelNames[] = {
  "laminar",
  "ksgs",
  "smagorinsky",
  "wale",
  "sst",
  "sst_des"};

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
  TM_END = 25
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
  "END"};

enum ActuatorType {
  ActLinePointDrag = 0,
  ActLineFAST = 1,
  ActDiskFAST = 2,
  ActuatorType_END
};

 static std::map<std::string, ActuatorType> ActuatorTypeMap = {
     {"ActLinePointDrag",ActuatorType::ActLinePointDrag},
     {"ActLineFAST",ActuatorType::ActLineFAST},
     {"ActDiskFAST",ActuatorType::ActDiskFAST}
 };

} // namespace nalu
} // namespace Sierra

#endif
