/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleMomentumEdgeABLTopBC_h
#define AssembleMomentumEdgeABLTopBC_h

#include<SolverAlgorithm.h>
#include<FieldTypeDef.h>
#include<complex.h> // Must proceed fftw3.h in order to get native c complex
#include<fftw3.h>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class Realm;

class AssembleMomentumEdgeABLTopBC : public SolverAlgorithm
{
public:

  AssembleMomentumEdgeABLTopBC(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem, std::vector<int>& grid_dims);
  virtual ~AssembleMomentumEdgeABLTopBC() {}
  virtual void initialize_connectivity();
  virtual void execute();
  virtual void initialize(
    int imax_,
    int jmax_,
    int kmax_,
    double zSample_,
    double *xL_,
    double *yL_,
    double *deltaZ_,
    stk::mesh::Entity *nodeMapSamp_,
    stk::mesh::Entity *nodeMapBC_,
    stk::mesh::Entity *nodeMapM1_,
    int *indexMapSamp_,
    int *indexMapBC_,
    int *indexMapSampGlobal_,
    int *sampleDistrib_,
    int *displ_,
    int *nSamp_,
    int *nBC_);
  virtual void potentialBCPeriodicPeriodic(
    double *wSamp_,
    double xL_,
    double yL_,
    double deltaZ_,
    double *uAvg,
    int imax_,
    int jmax_,
    double *uBC_,
    double *vBC_,
    double *wBC_,
    std::complex<double> *uCoef_,
    std::complex<double> *vCoef_,
    std::complex<double> *wCoef_ );

  VectorFieldType *velocity_;
  VectorFieldType *bcVelocity_;
  ScalarFieldType *density_;
  GenericFieldType *exposedAreaVec_;
  int imax_, jmax_, kmax_;
  std::vector<double> wSamp_, uBC_, vBC_, wBC_;
  std::vector< std::complex<double> > uCoef_, vCoef_, wCoef_;
  std::vector<stk::mesh::Entity> nodeMapSamp_, nodeMapBC_, nodeMapM1_;
  std::vector<int> indexMapSamp_, indexMapBC_, indexMapSampGlobal_,
                   sampleDistrib_, displ_;
  double xL_, yL_, deltaZ_, zSample_;
  int nSamp_, nBC_;
  bool needToInitialize_;
};

} // namespace nalu
} // namespace Sierra

#endif
