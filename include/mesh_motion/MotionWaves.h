#ifndef MOTIONWAVES_H
#define MOTIONWAVES_H


#include <string>
#include <cmath>
#include "MotionBase.h"

namespace stk {
namespace mesh {
class MetaData;
}
}

namespace sierra{
namespace nalu{
class MotionWaves : public MotionBase
{
public:
  MotionWaves(
    stk::mesh::MetaData&,
    const YAML::Node&);

  virtual ~MotionWaves()
  {
  }

virtual void build_transformation(const double, const double*);

  /** Function to compute motion-specific velocity
   *
   * @param[in] time           Current time
   * @param[in] compTrans      Transformation matrix
   *                           for points other than xyz
   * @param[in] mxyz           Model coordinates
   * @param[in] mxyz           Transformed coordinates
   */
  virtual ThreeDVecType compute_velocity(
    const double time,
    const TransMatType& compTrans,
    const double* mxyz,
    const double* cxyz );

  /** perform post compute geometry work for this motion
   *
   * @param[in] computedMeshVelDiv flag to denote if divergence of
   *                               mesh velocity already computed
   */
	void post_compute_geometry(
		stk::mesh::BulkData&,
        stk::mesh::PartVector&,
        stk::mesh::PartVector&,
        bool& computedMeshVelDiv );

    struct StokesCoeff{
        double k;
        double d;
        double a11;  
        double a22;
        double a31;
        double a33;
        double a42;
        double a44;
        double a51;
        double a53;
        double a55;
        double b22;
        double b31;
        double b42;
        double b44;
        double b53;
        double b55;
        double c0; 
        double c2;
        double c4;
        double d2;
        double d4;
        double e2;
        double e4;  
    };

    void get_StokesCoeff(StokesCoeff *stokes);

private:
  MotionWaves() = delete;
  MotionWaves(const MotionWaves&) = delete;

  void load(const YAML::Node&);
  void translation_mat(const ThreeDVecType&);

  void Stokes_coefficients();
  void Stokes_parameters();
  
  double my_sinh_sin(int i, int j,double phase);
  double my_cosh_cos(int i, int j,double phase);

  const double g_{9.81};

  std::string waveModel_{"Airy"};   
  // General parameters for waves
  double height_{0.1}; // Wave height
  double period_{1.0}; // Wave period
  double length_{1.0}; // Wave length
  double waterdepth_{100}; // Water depth
  double omega_{2.*M_PI}; // Angular frequency omega=2*pi/tau (tau being the period)
  double k_{2.*M_PI}; // Angular wavenumber k=2*pi/lambda (lambda being the wavenumber)
  double sealevelz_{0.0}; // Sea level assumed to be at z=0
  double c_{1.};   // wave phase velocity c


  // Stokes waves parameters
  int StokesOrder_{2}; // Stokes order - it defaults to 2
  double a11_{0.};
  double a22_{0.};
  double a31_{0.};
  double a33_{0.};
  double a42_{0.};
  double a44_{0.};
  double a51_{0.};
  double a53_{0.};
  double a55_{0.};
  double b22_{0.};
  double b31_{0.};
  double b42_{0.};
  double b44_{0.};
  double b53_{0.};
  double b55_{0.};
  double c0_{0.};
  double c2_{0.};
  double c4_{0.};
  double d2_{0.};
  double d4_{0.};
  double e2_{0.};
  double e4_{0.};
  double eps_{0.1};
  double Q_{0.};
  double cs_{0.2}; //Mean Stokes drift speed

    // Deformation damping function
  double meshdampinglength_{1000};
  int meshdampingcoeff_{3};
    
};

} // nalu
} // sierra

#endif /* MOTIONWAVES_H */