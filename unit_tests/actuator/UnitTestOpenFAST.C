// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <OpenFAST.H>
#include <gtest/gtest.h>
#include <NaluEnv.h>

namespace sierra{
namespace nalu{

namespace{

TEST(OpenFAST_API, initializeOpenFAST){
  try{
  fast::fastInputs fi;
  fi.comm=NaluEnv::self().parallel_comm();
  fi.globTurbineData.resize(1);
  fi.debug = true;
  fi.dryRun = false;
  fi.nTurbinesGlob = 1;
  fi.tStart = 0.0;
  fi.simStart = fast::init;
  fi.nEveryCheckPoint = 1;
  fi.dtFAST = 0.00625;
  fi.tMax = 0.0625;

  fi.globTurbineData[0].FASTInputFileName = "reg_tests/test_files/nrel5MWactuatorLine/nrel5mw.fst";
  fi.globTurbineData[0].FASTRestartFileName ="blah";
  fi.globTurbineData[0].TurbID=0;
  fi.globTurbineData[0].TurbineBasePos = {0,0,0};
  fi.globTurbineData[0].TurbineHubPos = {0,0,60.0};
  fi.globTurbineData[0].numForcePtsBlade = 10;
  fi.globTurbineData[0].numForcePtsTwr=10;
  fi.globTurbineData[0].air_density=1.0;
  fi.globTurbineData[0].nacelle_area=1.0;
  fi.globTurbineData[0].nacelle_cd=1.0;

  fast::OpenFAST fast;

  fast.setInputs(fi);
  fast.setTurbineProcNo(0,0);
  fast.init();

  int nTurb = fast.get_nTurbinesGlob();

  for (int iTurb = 0; iTurb < nTurb; iTurb++) {
  if (fast.get_procNo(iTurb) == NaluEnv::self().parallel_rank()) {
    if (!fast.isDryRun()) {
      const int numForcePts = fast.get_numForcePts(iTurb);

      try{
      for (int np = 0; np < numForcePts; np++) {


        switch (fast.getForceNodeType(iTurb, np)) {
        case fast::HUB: {
          float nac_cd = fast.get_nacelleCd(iTurb);
          break;
        }
        case fast::BLADE: {
          double chord = fast.getChord(np, iTurb);
          break;
        }
        case fast::TOWER: {
          break;
        }
        default:
          throw std::runtime_error("Actuator line model node type not valid");
          break;
        }
      }
      }catch(std::exception const& err){
        throw std::runtime_error(err.what());
      }
    }
  }
  }

  fast.end();
  SUCCEED();
  }catch(std::exception const& err){
    FAIL()<<err.what();
  }
}

}

}
}
