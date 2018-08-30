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
    EquationSystem *eqSystem, std::vector<int>& grid_dims_,
    std::vector<int>& horiz_bcs_, double z_sample_);
  virtual ~AssembleMomentumEdgeABLTopBC() {}
  virtual void initialize_connectivity();
  virtual void execute();
  virtual void initialize();
  virtual void potentialBCPeriodicPeriodic(
    std::vector<double>& wSamp,
    std::vector<double>& uAvg,
    std::vector<double>& uBC,
    std::vector<double>& vBC,
    std::vector<double>& wBC );
  virtual void potentialBCInflowPeriodic(
    std::vector<double>& wSamp,
    std::vector<double>& uAvg,
    std::vector<double>& uBC,
    std::vector<double>& vBC,
    std::vector<double>& wBC );
  VectorFieldType *velocity_;
  VectorFieldType *bcVelocity_;
  ScalarFieldType *density_;
  GenericFieldType *exposedAreaVec_;
  int imax_, jmax_, kmax_;
  std::vector<double> weight_;
  std::vector<stk::mesh::Entity> nodeMapSamp_, nodeMapBC_, nodeMapM1_,
                                 nodeMapX0_;
  std::vector<int> indexMapSampGlobal_, indexMapBC_, sampleDistrib_, displ_,
                   horizBC_;
  double xL_, yL_, deltaZ_, zSample_;
  int nBC_, nX0_, horizBCType_;
  bool needToInitialize_;
  fftw_plan planFourier2dF_, planFourier2dB_, planSinx_, planCosx_,
            planFourierxF_, planFourierxB_,   planSiny_, planCosy_,
            planFourieryF_, planFourieryB_, planSinxSiny_, planCosxSiny_,
            planSinxCosy_;
};

} // namespace nalu
} // namespace Sierra

#endif
