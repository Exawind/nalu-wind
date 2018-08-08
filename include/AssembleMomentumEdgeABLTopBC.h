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
  virtual void potentialBCPeriodicPeriodic(
    double *wSamp_,
    fftw_complex *uCoef_,
    fftw_complex *vCoef_,
    fftw_complex *wCoef_,
    double *uBC_,
    double *vBC_,
    double *wBC_,
    double xL,
    double yL,
    double deltaZ, 
    int nx,
    int ny );

  VectorFieldType *velocity_;
  VectorFieldType *bcVelocity_;
  ScalarFieldType *density_;
  GenericFieldType *exposedAreaVec_;
  int imax_, jmax_, kmax_;
  double *wSamp_, *uBC_, *vBC_, *wBC_;
  fftw_complex *uCoef_, *vCoef_, *wCoef_;
//  std::vector<fftw_complex> uCoef_, vCoef_, wCoef_;
//  fftw_complex uCoef_, wCoef_;
  int counter;
};

} // namespace nalu
} // namespace Sierra

#endif
