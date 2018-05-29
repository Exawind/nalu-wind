/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleScalarEigenEdgeSolverAlgorithm_h
#define AssembleScalarEigenEdgeSolverAlgorithm_h

#include<SolverAlgorithm.h>
#include<FieldTypeDef.h>
#include<EigenDecomposition.h>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class Realm;
template <typename T> class PecletFunction;

class AssembleScalarEigenEdgeSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleScalarEigenEdgeSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    ScalarFieldType *scalarQ,
    VectorFieldType *dqdx,
    ScalarFieldType *thermalCond,
    ScalarFieldType *specHeat,
    ScalarFieldType *turbViscosity,
    const double turbSigma);

  virtual ~AssembleScalarEigenEdgeSolverAlgorithm();
  virtual void initialize_connectivity();
  virtual void execute();

  // eignenvalue helpers
  void perturb(double (&D)[3][3]);
  void sort(const double (&D)[3][3]);

  double van_leer(
    const double &dqm,
    const double &dqp,
    const double &small);

  const bool meshMotion_;
  const double includeDivU_;
  const double turbSigma_;

  ScalarFieldType *scalarQ_;
  VectorFieldType *dqdx_;
  ScalarFieldType *thermalCond_;
  ScalarFieldType *specHeat_;
  ScalarFieldType *turbViscosity_;
  VectorFieldType *velocityRTM_;
  VectorFieldType *coordinates_;
  ScalarFieldType *density_;
  ScalarFieldType *massFlowRate_;
  VectorFieldType *edgeAreaVec_;

  // extra for GGDH
  ScalarFieldType *turbKe_;
  VectorFieldType *velocity_;
  GenericFieldType *dudx_;

  // peclect function specifics
  PecletFunction<double>* pecletFunction_;

  // constants and perturbation from user
  const double cGGDH_;
  const double deltaB_;
  const double perturbTurbKe_;
  double BinvXt_[3];

  // fixed size
  double duidxj_[3][3];
  double dqdxj_[3];
  double R_[3][3];
  double b_[3][3];
  double D_[3][3];
  double Q_[3][3];
  int rowMap_[3];
};

} // namespace nalu
} // namespace Sierra

#endif
