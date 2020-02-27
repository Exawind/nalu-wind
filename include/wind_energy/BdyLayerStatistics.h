// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef BDYLAYERSTATISTICS_H
#define BDYLAYERSTATISTICS_H

#include "KokkosInterface.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Part.hpp"

#include <memory>

namespace YAML { class Node; }

namespace sierra {
namespace nalu {

class Realm;
class TurbulenceAveragingPostProcessing;
class AveragingInfo;
class BdyHeightAlgorithm;

/** Boundary layer statistics post-processing utility
 *
 *  A post-processing utility to compute statistics of flow quantities that are
 *  temporally and spatially averaged. Primarily used to evaluate atmospheric
 *  boundary layer characteristics during precursor simulations, this utility
 *  can also be used for channel flows.
 *
 *  The temporal averaging is perfomed via
 *  sierra::nalu::TurbulenceAveragingPostProcessing class.
 */
class BdyLayerStatistics
{
public:
  using ArrayType = Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>;
  using HostArrayType = typename ArrayType::HostMirror;

  BdyLayerStatistics(
    Realm&,
    const YAML::Node&);

  virtual ~BdyLayerStatistics();

  /** Check for parts and initialize indexing field
   */
  void setup();

  /** Determine the number of height levels and node association with height levels
   */
  void initialize();

  /** Compute statistics of interest and output them if necessary
   */
  void execute();

  /** Return the spatial average of the instantaneous velocity field at a given height
   *
   *  The method interpolates the spatially averaged velocity from available
   *  heights to return the average velocity at an arbitrary height.
   *
   *  @param[in] height The height where velocity is desired
   *  @param[out] velArray A pointer to an array of nDim which contains the components of velocity
   */
  void velocity(double, double*);

  /** Return the spatially and temporally averaged velocity at a given height
   *
   *  The method interpolates the spatially averaged velocity from available
   *  heights to return the average velocity at an arbitrary height.
   *
   *  @param[in] height The height where velocity is desired
   *  @param[out] velArray A pointer to an array of nDim which contains the components of velocity
   */
  void time_averaged_velocity(double, double*);

  //! Return the spatial average of the instantaneous density field at a given height
  void density(double, double*);

  //! Return the spatial average of the instantaneous temperature field at a given height
  void temperature(double, double*);

  void set_utau_avg(double utau)
  { uTauAvg_ = utau; }

  //! Number of vertical levels on this ABL mesh
  int abl_num_levels() const { return heights_.size(); }

  //! Return the reference to the heights vector
  const HostArrayType& abl_heights() const { return heights_; }

  //! Return the index in height array
  //!
  //! Returns index into the height array such that
  //!     \f$ heights[i] \leq ht \leq heights[i+1]\$
  //!
  int abl_height_index(const double) const;

  //! Process the velocity data and compute averages
  void impl_compute_velocity_stats();

  //! Process the temperature field and compute averages
  void impl_compute_temperature_stats();

private:
  BdyLayerStatistics() = delete;
  BdyLayerStatistics(const BdyLayerStatistics&) = delete;

  //! Process the user inputs and initialize class data
  void load(const YAML::Node&);

  //! Initialize necessary parameters in sierra::nalu::TurbulenceAveragingPostProcessing
  void setup_turbulence_averaging(const double);

  //! Output averaged velocity and stress profiles as a function of height
  void output_velocity_averages();

  //! Output averaged temperature profiles as a function of height
  void output_temperature_averages();

  /** Helper method to perform interpolations across data with multiple components
   *
   *  @param[in] nComp The number of components: scalar=1, vector=3, tensor=6 and so on
   *  @param[in] varArray Reference to the array that contains averaged data at all heights
   *  @param[in] height The height where data is to be interpolated
   *  @param[out] interpArry Pointer to an array of size nComp where interpolated data is populated
   */
  void interpolate_variable(
    int,
    HostArrayType&,
    double,
    double*);

  /** Prepare the NetCDF statstics file with the necessary metadata
   */
  void prepare_nc_file();

  //! Write out statistics for the current time step to the NetCDF file
  void write_time_hist_file();

  //! Reference to Realm object
  Realm& realm_;

  //! Spatially averaged instantaneous velocity at desired heights [nHeights, nDim]
  ArrayType d_velAvg_;

  //! Spatially and temporally averaged velocity at desired heights [nHeights, nDim]
  ArrayType d_velBarAvg_;

  //! Spatially averaged resolved stress [nHeights, nDim]
  ArrayType d_uiujAvg_;

  //! Spatially averaged sfs stress [nHeights, nDim]
  ArrayType d_sfsAvg_;

  //! Spatially and temporally averaged resolved stress field at desired heights [nHeights, nDim * 2]
  ArrayType d_uiujBarAvg_;

  //! Spatially and temporally averaged SFS field at desired heights [nHeights, nDim * 2]
  ArrayType d_sfsBarAvg_;

  //! Spatially averaged instantaneous temperature field [nHeights]
  ArrayType d_thetaAvg_;

  //! Spatially and temporally averaged temperature field [nHeights]
  ArrayType d_thetaBarAvg_;

  //! Spatially averaged instantaneous temperature SFS field
  ArrayType d_thetaSFSAvg_;

  //! Spatially averaged instantaneous temperature variance
  ArrayType d_thetaUjAvg_;

  //! Spatially and temporally averaged Temperature SFS field
  ArrayType d_thetaSFSBarAvg_;

  ArrayType d_thetaUjBarAvg_;

  //! Spatially averaged temperature variance [nHeights]
  ArrayType d_thetaVarAvg_;

  //! Spatially and temporally averaged temperature variance [nHeights]
  ArrayType d_thetaBarVarAvg_;

  //! Total nodal volume at each height level used for volumetric averaging
  ArrayType d_sumVol_;

  //! Average density at each height level
  ArrayType d_rhoAvg_;

  //! Height from the wall
  ArrayType d_heights_;

  //! Spatially averaged instantaneous velocity at desired heights [nHeights, nDim]
  HostArrayType velAvg_;

  //! Spatially and temporally averaged velocity at desired heights [nHeights, nDim]
  HostArrayType velBarAvg_;

  //! Spatially averaged resolved stress [nHeights, nDim]
  HostArrayType uiujAvg_;

  //! Spatially averaged sfs stress [nHeights, nDim]
  HostArrayType sfsAvg_;

  //! Spatially and temporally averaged resolved stress field at desired heights [nHeights, nDim * 2]
  HostArrayType uiujBarAvg_;

  //! Spatially and temporally averaged SFS field at desired heights [nHeights, nDim * 2]
  HostArrayType sfsBarAvg_;

  //! Spatially averaged instantaneous temperature field [nHeights]
  HostArrayType thetaAvg_;

  //! Spatially and temporally averaged temperature field [nHeights]
  HostArrayType thetaBarAvg_;

  //! Spatially averaged instantaneous temperature SFS field
  HostArrayType thetaSFSAvg_;

  //! Spatially averaged instantaneous temperature variance
  HostArrayType thetaUjAvg_;

  //! Spatially and temporally averaged Temperature SFS field
  HostArrayType thetaSFSBarAvg_;

  HostArrayType thetaUjBarAvg_;

  //! Spatially averaged temperature variance [nHeights]
  HostArrayType thetaVarAvg_;

  //! Spatially and temporally averaged temperature variance [nHeights]
  HostArrayType thetaBarVarAvg_;

  //! Total nodal volume at each height level used for volumetric averaging
  HostArrayType sumVol_;

  //! Average density at each height level
  HostArrayType rhoAvg_;

  //! Height from the wall
  HostArrayType heights_;

  //! Part names for post-processing
  std::vector<std::string> partNames_;

  //! Parts of the fluid mesh where velocity/temperature averaging is performed
  stk::mesh::PartVector fluidParts_;

  //! Map of `{variableName : netCDF_ID}` obtained from the NetCDF C interface
  std::unordered_map<std::string, int> ncVarIDs_;

  //! Name of the NetCDF file where statistics are output
  std::string bdyStatsFile_{"abl_statistics.nc"};

  //! Friction velocity average
  double uTauAvg_{0.0};

  //! Dimensionality of the mesh
  int nDim_{3};

  //! Output frequency for averaged statistics
  int outputFrequency_{10};

  //! Output frequency for time histories
  int timeHistOutFrequency_{10};

  //! Starting time step (offset for NetCDF with restarts)
  int startStep_{0};

  //! Height index field
  ScalarIntFieldType* heightIndex_;

  std::unique_ptr<BdyHeightAlgorithm> bdyHeightAlg_;

  //! Calculate temperature statistics
  bool calcTemperatureStats_{true};

  //! Flag indicating whether uTau history is processed
  bool hasUTau_{true};

  //! Flag indicating whether initialization must be performed
  bool doInit_{true};
};

}  // nalu
}  // sierra


#endif /* BDYLAYERSTATISTICS_H */
