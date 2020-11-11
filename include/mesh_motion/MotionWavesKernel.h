#ifndef MOTIONWAVES_H
#define MOTIONWAVES_H


#include <string>
#include <cmath>
#include "NgpMotion.h"

namespace stk {
namespace mesh {
class MetaData;
}
}

namespace sierra{
namespace nalu{
class MotionWavesKernel : public NgpMotionKernel<MotionWavesKernel>
{
public:
  MotionWavesKernel(
    stk::mesh::MetaData&,
    const YAML::Node&);

  KOKKOS_FUNCTION
  MotionWavesKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MotionWavesKernel() = default;

  KOKKOS_FUNCTION
  virtual void build_transformation(const DblType, const DblType*);

  /** Function to compute motion-specific velocity
   *
   * @param[in]  time       Current time
   * @param[in]  compTrans  Transformation matrix
   *                        for points other than xyz
   * @param[in]  mxyz       Model coordinates
   * @param[in]  mxyz       Transformed coordinates
   * @param[out] vel        Velocity associated with coordinates
   */
  KOKKOS_FUNCTION
  virtual void compute_velocity(
    const DblType time,
    const TransMatType& compTrans,
    const DblType* mxyz,
    const DblType* cxyz,
    ThreeDVecType& vel);

    struct StokesCoeff{
        DblType k;
        DblType d;
        DblType a11;
        DblType a22;
        DblType a31;
        DblType a33;
        DblType a42;
        DblType a44;
        DblType a51;
        DblType a53;
        DblType a55;
        DblType b22;
        DblType b31;
        DblType b42;
        DblType b44;
        DblType b53;
        DblType b55;
        DblType c0;
        DblType c2;
        DblType c4;
        DblType d2;
        DblType d4;
        DblType e2;
        DblType e4;
    };

    void get_StokesCoeff(StokesCoeff *stokes);

private:
  void load(const YAML::Node&);
  void translation_mat(const ThreeDVecType&);

  void Stokes_coefficients();
  void Stokes_parameters();
  
  DblType my_sinh_sin(int i, int j,DblType phase);
  DblType my_cosh_cos(int i, int j,DblType phase);

  const DblType g_{9.81};

  std::string waveModel_{"Airy"};   
  // General parameters for waves
  DblType height_{0.1}; // Wave height
  DblType period_{1.0}; // Wave period
  DblType length_{1.0}; // Wave length
  DblType waterdepth_{100}; // Water depth
  DblType omega_{2.*M_PI}; // Angular frequency omega=2*pi/tau (tau being the period)
  DblType k_{2.*M_PI}; // Angular wavenumber k=2*pi/lambda (lambda being the wavenumber)
  DblType sealevelz_{0.0}; // Sea level assumed to be at z=0
  DblType c_{1.};   // wave phase velocity c


  // Stokes waves parameters
  int StokesOrder_{2}; // Stokes order - it defaults to 2
  DblType a11_{0.};
  DblType a22_{0.};
  DblType a31_{0.};
  DblType a33_{0.};
  DblType a42_{0.};
  DblType a44_{0.};
  DblType a51_{0.};
  DblType a53_{0.};
  DblType a55_{0.};
  DblType b22_{0.};
  DblType b31_{0.};
  DblType b42_{0.};
  DblType b44_{0.};
  DblType b53_{0.};
  DblType b55_{0.};
  DblType c0_{0.};
  DblType c2_{0.};
  DblType c4_{0.};
  DblType d2_{0.};
  DblType d4_{0.};
  DblType e2_{0.};
  DblType e4_{0.};
  DblType eps_{0.1};
  DblType Q_{0.};
  DblType cs_{0.2}; //Mean Stokes drift speed

    // Deformation damping function
  DblType meshdampinglength_{1000};
  int meshdampingcoeff_{3};
    
};

} // nalu
} // sierra

#endif /* MOTIONWAVES_H */
