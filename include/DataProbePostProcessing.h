// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef DataProbePostProcessing_h
#define DataProbePostProcessing_h

#include "NaluParsedTypes.h"

#include <string>
#include <vector>
#include <utility>
#include <memory>

// stk_mesh/base/fem
#include <stk_mesh/base/Selector.hpp>

#include <stk_io/StkMeshIoBroker.hpp>

namespace YAML { class Node; }

// stk forwards
namespace stk {
  namespace mesh {
    class BulkData;
    class FieldBase;
    class MetaData;
    class Part;
    //class Selector; ? why is this?
    struct Entity;
    typedef std::vector< Part * > PartVector;
  }
}

namespace sierra{
namespace nalu{

class Realm;
class Transfer;
class Transfers;

enum class DataProbeSampleType{
  STEPCOUNT,
  APRXFREQUENCY
};

// Defines the different kinds of probe geometries
enum class DataProbeGeomType{
  LINEOFSITE,
  PLANE
};

class DataProbeInfo {
public:
  DataProbeInfo() { }
  ~DataProbeInfo() {}

  // for each type of probe, e.g., line of site, hold some stuff
  bool isLineOfSite_;
  int numProbes_;
  std::vector<std::string> partName_;
  std::vector<int> processorId_;
  std::vector<int> numPoints_;
  std::vector<int> generateNewIds_;
  std::vector<Coordinates> tipCoordinates_;
  std::vector<Coordinates> tailCoordinates_;
  std::vector<std::vector<stk::mesh::Entity> > nodeVector_;
  std::vector<stk::mesh::Part *> part_;

  // variables for sample planes
  bool isSamplePlane_;   
  std::vector<DataProbeGeomType> geomType_;
  std::vector<Coordinates> cornerCoordinates_;
  std::vector<Coordinates> edge1Vector_;
  std::vector<Coordinates> edge2Vector_;
  std::vector<int>         edge1NumPoints_;
  std::vector<int>         edge2NumPoints_;
  std::vector<Coordinates> offsetDir_;
  std::vector<std::vector<double>>  offsetSpacings_;


};

class DataProbeSpecInfo {
public:
  DataProbeSpecInfo();
  ~DataProbeSpecInfo();

  std::string xferName_;
  std::vector<std::string> fromTargetNames_;
  
  // vector of averaging information
  std::vector<DataProbeInfo *> dataProbeInfo_;
 
  // homegeneous collection of fields over each specification
  std::vector<std::pair<std::string, std::string> > fromToName_;
  std::vector<std::pair<std::string, int> > fieldInfo_;
};

class DataProbePostProcessing
{
public:
  
  DataProbePostProcessing(
    Realm &realm,
    const YAML::Node &node);
  ~DataProbePostProcessing();
  
  // load all of the options
  void load(
    const YAML::Node & node);

  void add_external_data_probe_spec_info(DataProbeSpecInfo* dpsInfo);

  // setup part creation and nodal field registration (before populate_mesh())
  void setup();

  // setup part creation and nodal field registration (after populate_mesh())
  void initialize();

  void register_field(
    const std::string fieldName,
    const int fieldSize,
    stk::mesh::MetaData &metaData,
    stk::mesh::Part *part);

  void review( 
    const DataProbeInfo *probeInfo);

  // we want these nodes to be excluded from anything of importance
  void create_inactive_selector();

  // create the transfer and hold the vector in the DataProbePostProcessing class
  void create_transfer();

  // optionally create an exodus database
  void create_exodus();

  // populate nodal field and output norms (if appropriate)
  void execute();

  // output to a file
  void provide_output_txt(const double currentTime);
  void provide_output_exodus(const double currentTime);

  
  // provide the inactive selector
  stk::mesh::Selector &get_inactive_selector();

  // hold the realm
  Realm &realm_;

  // frequency of output
  double outputFreq_;

  // width for output
  int w_;

  // xfer specifications
  std::string searchMethodName_;
  double searchTolerance_;
  double searchExpansionFactor_;

  // vector of specifications
  std::vector<DataProbeSpecInfo *> dataProbeSpecInfo_;

  // hold all the parts; provide a selector
  stk::mesh::PartVector allTheParts_;
  stk::mesh::Selector inactiveSelector_;

  // hold the transfers
  Transfers *transfers_;


  DataProbeSampleType probeType_;

private:
  std::unique_ptr<stk::io::StkMeshIoBroker> io;

  double previousTime_;
  bool useExo_{false};
  bool useText_{false};
  std::string exoName_;
  size_t fileIndex_;
  size_t precisionvar_;
};

} // namespace nalu
} // namespace Sierra

#endif
