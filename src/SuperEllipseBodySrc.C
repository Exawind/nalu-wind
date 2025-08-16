
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <SuperEllipseBodySrc.h>
#include <Realm.h>
#include <SolutionOptions.h>

#include <master_element/TensorOps.h>

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// SuperEllipseBodySrc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SuperEllipseBodySrc::SuperEllipseBodySrc(const SolutionOptions& solnOpts)
{
  read_from_file();
}

SuperEllipseBodySrc::read_from_file()
{
  // Implementation for reading location, orientation, and dimensions from file

    std::ifstream fin("nacelle.dat");
    if (!fin.good()) {
        std::cerr << "Input file 'nacelle.dat' does not exist: user "
                     "specified (or default) name= "
                  << std::endl;
      return 0;
    }

    fin >> seb_loc_[0] >> seb_loc_[1] >> seb_loc_[2];
    fin >> seb_orient_[0] >> seb_orient_[1] >> seb_orient_[2];
    fin >> seb_dim_[0] >> seb_dim_[1] >> seb_dim_[2];

    fin.close();
}
} // namespace nalu
} // namespace sierra
