// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <SuperEllipseBodySrc.h>
#include <SolutionOptions.h>
#include <iostream>
#include <fstream>

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
SuperEllipseBodySrc::SuperEllipseBodySrc(const SolutionOptions& /* solnOpts */)
{
  read_from_file();
}

void SuperEllipseBodySrc::read_from_file()
{
    // Implementation for reading location, orientation, and dimensions from file

    std::ifstream fin("nacelle.dat");
    if (!fin.good()) {
        std::cerr << "Input file 'nacelle.dat' does not exist: user "
                     "specified (or default) name= "
                  << std::endl;
    }

    fin >> seb_loc_.x() >> seb_loc_.y() >> seb_loc_.z();
    fin >> seb_orient_.x() >> seb_orient_.y() >> seb_orient_.z();
    fin >> seb_dim_.x() >> seb_dim_.y() >> seb_dim_.z();

    fin.close();
}
} // namespace nalu
} // namespace sierra
