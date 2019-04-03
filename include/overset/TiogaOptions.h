/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TIOGAOPTIONS_H
#define TIOGAOPTIONS_H

#include "yaml-cpp/yaml.h"

namespace TIOGA {
class tioga;
}

namespace tioga_nalu {

// Notes:
//
// Creating a class to hold the options instead of adding them as members in
// TiogaSTKIface so that in future we can have the ability to set these
// per-block (TiogaBlock).

/** Options to control TIOGA overset hole-cut behavior
 *
 *  This class is a simple structure that holds the different options that can
 *  be used to control the behavior of TIOGA during holecuts. These options can
 *  be set from the Nalu-Wind input file within the overset boundary condition
 *  section and are passed to TIOGA during runtime.
 */
class TiogaOptions
{
public:
  /** Initialize options from the input file
   */
  void load(const YAML::Node&);

  /** Control TIOGA behavior by calling the appropriate API calls.
   */
  void set_options(TIOGA::tioga&);

  bool set_resolutions() const { return setResolutions_; }

  bool reduce_fringes() const { return reduceFringes_; }

private:
  //! Symmetry plane direction [1 = x; 2 = y; 3 = z]; default = 3
  int symmetryDir_{3};

  /** Number of fringe layers on the overset BC sideset
   *
   *  This parameter indicates the number of layers that will be set as
   *  mandatory fringes on an overset boundary. This default for this parameter
   *  is set in TIOGA API, so an additional flag is used to determine if the
   *  user has provided this variable in the input file.
   */
  int nFringe_{1};

  /** Number of cells from outer boundary that are excluded from being donors
   *
   *  Like nfringe_ option, this option is only used if the user explicity sets
   *  this in the input file, the default value isn't used.
   */
  int mExclude_{3};

  //! Set the node and cell resolutions from Nalu-Wind instead of letting TIOGA
  //! compute it for P=1 cells. Default is true
  bool setResolutions_{true};

  //! Option to let TIOGA attempt to reduce fringes.
  bool reduceFringes_{false};

  //! Indicates whether the user has set the number of mandatory fringe points
  //! in the input file.
  bool hasNumFringe_{false};

  //! Flag indicating whether user has set the mexclude variable
  bool hasMexclude_{false};
};

}

#endif /* TIOGAOPTIONS_H */
