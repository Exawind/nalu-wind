
#include "wind_energy/ABLForcingAlgorithm.h"
#include "Realm.h"
#include "xfer/Transfer.h"
#include "xfer/Transfers.h"
#include "utils/LinearInterpolation.h"
#include "wind_energy/BdyLayerStatistics.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

#include <stk_io/IossBridge.hpp>

#include <stk_util/parallel/ParallelReduce.hpp>

#include <boost/format.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace sierra {
namespace nalu {

ABLForcingAlgorithm::ABLForcingAlgorithm(Realm& realm, const YAML::Node& node)
  : realm_(realm),
    momSrcType_(ABLForcingAlgorithm::OFF),
    tempSrcType_(ABLForcingAlgorithm::OFF),
    alphaMomentum_(1.0),
    alphaTemperature_(1.0),
    velHeights_(0),
    tempHeights_(0),
    velXTimes_(0),
    velYTimes_(0),
    velZTimes_(0),
    tempTimes_(0),
    velX_(0),
    velY_(0),
    velZ_(0),
    temp_(0),
    UmeanCalc_(0),
    rhoMeanCalc_(0),
    USource_(0),
    TmeanCalc_(0),
    TSource_(0)
{
  if (realm_.bdyLayerStats_ == nullptr)
    throw std::runtime_error("ABL Forcing requires ABL Boundary Layer statistics");
  load(node);
}

ABLForcingAlgorithm::ABLForcingAlgorithm(Realm& realm)
  : realm_(realm)
{}

ABLForcingAlgorithm::~ABLForcingAlgorithm()
{}

void
ABLForcingAlgorithm::load(const YAML::Node& node)
{
  get_if_present(node, "output_frequency", outputFreq_, outputFreq_);
  get_if_present(node, "output_format", outFileFmt_, outFileFmt_);

  if (node["momentum"])
    load_momentum_info(node["momentum"]);

  if (node["temperature"])
    load_temperature_info(node["temperature"]);
}

void
ABLForcingAlgorithm::load_momentum_info(const YAML::Node& node)
{
  std::string mom_type = node["type"].as<std::string>();
  if (mom_type == "user_defined") {
    momSrcType_ = ABLForcingAlgorithm::USER_DEFINED;
  } else if (mom_type == "computed") {
    momSrcType_ = ABLForcingAlgorithm::COMPUTED;
  } else {
    throw std::runtime_error(
      "ABLForcingAlgorithm: Invalid type specification for momentum. "
      "Valid types are: [user_defined, computed]");
  }
  get_if_present(node, "relaxation_factor", alphaMomentum_, alphaMomentum_);
  get_required<std::vector<double>>(node, "heights", velHeights_);
  auto nHeights = velHeights_.size();

  // Load momentum source time histories in temporary data structures, test
  // consistency of data with heights and then recast them into 2-D lookup
  // arrays.
  Array2D<double> vxtmp, vytmp, vztmp;
  get_required<Array2D<double>>(node, "velocity_x", vxtmp);
  get_required<Array2D<double>>(node, "velocity_y", vytmp);
  get_required<Array2D<double>>(node, "velocity_z", vztmp);

  create_interp_arrays(nHeights, vxtmp, velXTimes_, velX_);
  create_interp_arrays(nHeights, vytmp, velYTimes_, velY_);
  create_interp_arrays(nHeights, vztmp, velZTimes_, velZ_);

  const int ndim = realm_.spatialDimension_;
  if (momSrcType_ == COMPUTED) {
    UmeanCalc_.resize(nHeights);
    rhoMeanCalc_.resize(nHeights);

    for (size_t i = 0; i < nHeights; i++) {
      UmeanCalc_[i].resize(ndim);
    }
  }

  USource_.resize(ndim);
  for (int i = 0; i < ndim; i++) {
    USource_[i].resize(nHeights);
  }
}

void
ABLForcingAlgorithm::load_temperature_info(const YAML::Node& node)
{
  std::string temp_type = node["type"].as<std::string>();
  if (temp_type == "user_defined") {
    tempSrcType_ = ABLForcingAlgorithm::USER_DEFINED;
  } else if (temp_type == "computed") {
    tempSrcType_ = ABLForcingAlgorithm::COMPUTED;
  } else {
    throw std::runtime_error(
      "ABLForcingAlgorithm: Invalid type specification for temperature. "
      "Valid types are: [user_defined, computed]");
  }
  get_if_present(node, "relaxation_factor", alphaTemperature_, alphaTemperature_);
  get_required<std::vector<double>>(node, "heights", tempHeights_);
  auto nHeights = tempHeights_.size();

  // Load temperature source time histories, check data consistency and create
  // interpolation lookup tables.
  Array2D<double> temp;
  get_required<Array2D<double>>(node, "temperature", temp);

  create_interp_arrays(nHeights, temp, tempTimes_, temp_);

  TSource_.resize(nHeights);
  if (tempSrcType_ == COMPUTED)
    TmeanCalc_.resize(nHeights);
}

void
ABLForcingAlgorithm::create_interp_arrays(
  const std::vector<double>::size_type nHeights,
  const Array2D<double>& inpArr,
  std::vector<double>& outTimes,
  Array2D<double>& outValues)
{
  /* The input vector is shaped [nTimes, nHeights+1]. We transform it to two
   * arrays:
   *    time[nTimes] = inp[nTimes,0], and
   *    value[nHeights, nTimes] -> swap rows/cols from input
   */

  // Check that all timesteps contain values for all the heights
  for (auto vx : inpArr) {
    ThrowAssert((vx.size() == (nHeights + 1)));
  }

  auto nTimes = inpArr.size();
  outTimes.resize(nTimes);
  outValues.resize(nHeights);
  for (std::vector<double>::size_type i = 0; i < nHeights; i++) {
    outValues[i].resize(nTimes);
  }
  for (std::vector<double>::size_type i = 0; i < nTimes; i++) {
    outTimes[i] = inpArr[i][0];
    for (std::vector<double>::size_type j = 0; j < nHeights; j++) {
      outValues[j][i] = inpArr[i][j + 1];
    }
  }
}

void
ABLForcingAlgorithm::initialize()
{
  if (momSrcType_ != OFF) {
    NaluEnv::self().naluOutputP0()
      << "ABL Forcing active for Momentum Equations\n"
      << "\t Number of planes: " << velHeights_.size()
      << "\n\t Number of time steps: " << velXTimes_.size() << std::endl;
  }
  if (tempSrcType_ != OFF) {
    NaluEnv::self().naluOutputP0()
      << "ABL Forcing active for Temperature Equation\n"
      << "\t Number of planes: " << tempHeights_.size()
      << "\n\t Number of time steps: " << tempTimes_.size() << std::endl
      << std::endl;
  }

  // Prepare output files to dump sources when computed during precursor phase
  if (( NaluEnv::self().parallel_rank() == 0 ) &&
      ( momSrcType_ == COMPUTED )) {
    std::string uxname((boost::format(outFileFmt_)%"Ux").str());
    std::string uyname((boost::format(outFileFmt_)%"Uy").str());
    std::string uzname((boost::format(outFileFmt_)%"Uz").str());
    std::fstream uxFile, uyFile, uzFile;
    uxFile.open(uxname.c_str(), std::fstream::out);
    uyFile.open(uyname.c_str(), std::fstream::out);
    uzFile.open(uzname.c_str(), std::fstream::out);

    uxFile << "# Time, " ;
    uyFile << "# Time, " ;
    uzFile << "# Time, " ;
    for (size_t ih = 0; ih < velHeights_.size(); ih++) {
      uxFile << velHeights_[ih] << " ";
      uyFile << velHeights_[ih] << " ";
      uzFile << velHeights_[ih] << " ";
    }
    uxFile << std::endl ;
    uyFile << std::endl ;
    uzFile << std::endl ;
    uxFile.close();
    uyFile.close();
    uzFile.close();
  }
}

void
ABLForcingAlgorithm::execute()
{
  if (momentumForcingOn())
    compute_momentum_sources();

  if (temperatureForcingOn())
    compute_temperature_sources();
}

void
ABLForcingAlgorithm::compute_momentum_sources()
{
  const double dt = realm_.get_time_step();
  const double currTime = realm_.get_current_time();

  if (momSrcType_ == COMPUTED) {
    auto* bdyLayerStats = realm_.bdyLayerStats_;
    for (size_t ih=0; ih < velHeights_.size(); ih++) {
      bdyLayerStats->velocity(velHeights_[ih], UmeanCalc_[ih].data());
      bdyLayerStats->density(velHeights_[ih], &rhoMeanCalc_[ih]);
    }
    for (size_t ih = 0; ih < velHeights_.size(); ih++) {
      double xval, yval;

      // Interpolate the velocities from the table to the current time
      utils::linear_interp(velXTimes_, velX_[ih], currTime, xval);
      utils::linear_interp(velYTimes_, velY_[ih], currTime, yval);

      // Compute the momentum source
      // Momentum source in the x direction
      USource_[0][ih] = rhoMeanCalc_[ih] * (alphaMomentum_ / dt) *
                          (xval - UmeanCalc_[ih][0]);
      // Momentum source in the y direction
      USource_[1][ih] = rhoMeanCalc_[ih] * (alphaMomentum_ / dt) *
                          (yval - UmeanCalc_[ih][1]);

      // No momentum source in z-direction
      USource_[2][ih] = 0.0;

    }

  } else {
    for (size_t ih = 0; ih < velHeights_.size(); ih++) {
      utils::linear_interp(velXTimes_, velX_[ih], currTime, USource_[0][ih]);
      utils::linear_interp(velYTimes_, velY_[ih], currTime, USource_[1][ih]);
      utils::linear_interp(velZTimes_, velZ_[ih], currTime, USource_[2][ih]);
    }
  }

  const int tcount = realm_.get_time_step_count();
  if (( NaluEnv::self().parallel_rank() == 0 ) &&
      ( momSrcType_ == COMPUTED ) &&
      ( tcount % outputFreq_ == 0)) {
    std::string uxname((boost::format(outFileFmt_)%"Ux").str());
    std::string uyname((boost::format(outFileFmt_)%"Uy").str());
    std::string uzname((boost::format(outFileFmt_)%"Uz").str());
    std::fstream uxFile, uyFile, uzFile;
    uxFile.open(uxname.c_str(), std::fstream::app);
    uyFile.open(uyname.c_str(), std::fstream::app);
    uzFile.open(uzname.c_str(), std::fstream::app);

    uxFile << std::setw(12) << currTime << " ";
    uyFile << std::setw(12) << currTime << " ";
    uzFile << std::setw(12) << currTime << " ";
    for (size_t ih = 0; ih < velHeights_.size(); ih++) {
      uxFile << std::setprecision(6)
             << std::setw(15)
             << USource_[0][ih] << " ";
      uyFile << std::setprecision(6)
             << std::setw(15)
             << USource_[1][ih] << " ";
      uzFile << std::setprecision(6)
             << std::setw(15)
             << USource_[2][ih] << " ";
    }
    uxFile << std::endl;
    uyFile << std::endl;
    uzFile << std::endl;
    uxFile.close();
    uyFile.close();
    uzFile.close();
  }
}

void
ABLForcingAlgorithm::compute_temperature_sources()
{
  const double dt = realm_.get_time_step();
  const double currTime = realm_.get_current_time();

  if (tempSrcType_ == COMPUTED) {
    auto* bdyLayerStats = realm_.bdyLayerStats_;
    for (size_t ih=0; ih < tempHeights_.size(); ih++) {
      bdyLayerStats->temperature(tempHeights_[ih], &TmeanCalc_[ih]);
    }
    for (size_t ih = 0; ih < tempHeights_.size(); ih++) {
      double tval;
      utils::linear_interp(tempTimes_, temp_[ih], currTime, tval);
      TSource_[ih] = (alphaTemperature_ / dt) * (tval - TmeanCalc_[ih]);
    }
  } else {
    for (size_t ih = 0; ih < tempHeights_.size(); ih++) {
      utils::linear_interp(tempTimes_, temp_[ih], currTime, TSource_[ih]);
    }
  }

  const int tcount = realm_.get_time_step_count();
  if (( NaluEnv::self().parallel_rank() == 0 ) &&
      ( tempSrcType_ == COMPUTED ) &&
      ( tcount % outputFreq_ == 0)) {
    std::string fname((boost::format(outFileFmt_)%"T").str());
    std::fstream tFile;
    tFile.open(fname.c_str(), std::fstream::app);

    tFile << currTime << " ";
    for (size_t ih = 0; ih < tempHeights_.size(); ih++) {
      tFile << std::setprecision(6)
            << std::setw(15)
            << TSource_[ih] << " ";
    }
    tFile << std::endl;
    tFile.close();
  }
}

void
ABLForcingAlgorithm::eval_momentum_source(
  const double zp, std::vector<double>& momSrc)
{
  const int nDim = realm_.spatialDimension_;
  if (velHeights_.size() == 1) {
    // Constant source term throughout the domain
    for (int i = 0; i < nDim; i++) {
      momSrc[i] = USource_[i][0];
    }
  } else {
    // Linearly interpolate source term within the planes, maintain constant
    // source term above and below the heights provided
    for (int i = 0; i < nDim; i++) {
      utils::linear_interp(velHeights_, USource_[i], zp, momSrc[i]);
    }
  }
}

void
ABLForcingAlgorithm::eval_temperature_source(const double zp, double& tempSrc)
{
  if (tempHeights_.size() == 1) {
    tempSrc = TSource_[0];
  } else {
    utils::linear_interp(tempHeights_, TSource_, zp, tempSrc);
  }
}

} // namespace nalu
} // namespace sierra
