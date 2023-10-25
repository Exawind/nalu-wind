// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <DataProbePostProcessing.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>

#include <stk_io/StkMeshIoBroker.hpp>

// xfer
#include <xfer/Transfer.h>
#include <xfer/Transfers.h>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>

// basic c++
#include <stdexcept>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <iostream>

// boost
#ifdef NALU_USES_BOOST
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#endif

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// DataProbeSpecInfo - holds DataProbeInfo
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
DataProbeSpecInfo::DataProbeSpecInfo()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
DataProbeSpecInfo::~DataProbeSpecInfo()
{
  // delete the probe info
  for (size_t k = 0; k < dataProbeInfo_.size(); ++k)
    delete dataProbeInfo_[k];
}

//==========================================================================
// Class Definition
//==========================================================================
// DataProbePostProcessing - post process
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
DataProbePostProcessing::DataProbePostProcessing(
  Realm& realm, const YAML::Node& node)
  : realm_(realm),
    outputFreq_(10),
    writeCoords_(true),
    gzLevel_(0),
    w_(26),
    searchMethodName_("none"),
    searchTolerance_(1.0e-4),
    searchExpansionFactor_(1.5),
    probeType_(DataProbeSampleType::STEPCOUNT),
    previousTime_(0.0),
    exoName_("data_probes.exo"),
    precisionvar_(8)
{
  // load the data
  load(node);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
DataProbePostProcessing::~DataProbePostProcessing()
{
  // delete xfer(s)
  if (NULL != transfers_)
    delete transfers_;

  // delete data probes specifications vector
  for (size_t k = 0; k < dataProbeSpecInfo_.size(); ++k)
    delete dataProbeSpecInfo_[k];
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::load(const YAML::Node& y_node)
{
  // check for any data probes
  const YAML::Node y_dataProbe = y_node["data_probes"];
  if (y_dataProbe) {
    NaluEnv::self().naluOutputP0()
      << "DataProbePostProcessing::load" << std::endl;

    // Set the output format for probes
    std::vector<std::string> formatList;
    const YAML::Node y_formats = y_dataProbe["output_format"];
    if (y_formats) {
      if (y_formats.Type() == YAML::NodeType::Sequence) { // Provided as as
                                                          // sequence
        for (size_t ioutput = 0; ioutput < y_formats.size(); ++ioutput) {
          const YAML::Node y_format = y_formats[ioutput];
          std::string formatName = y_format.as<std::string>();
          formatList.push_back(formatName);
        }
      } else { // Not provided as a sequence, just one string
        std::string formatName = y_formats.as<std::string>();
        formatList.push_back(formatName);
      }
    } else { // output_format not given at all, add the default ("text")
      std::string formatName("text");
      formatList.push_back(formatName);
    }
    // Go through and parse each format in formatList
    for (size_t iformat = 0; iformat < formatList.size(); iformat++) {
      std::string formatName = formatList[iformat];
      if (case_insensitive_compare(formatName, "exodus")) {
        useExo_ = true;
      } else if (case_insensitive_compare(formatName, "text")) {
        useText_ = true;
      } else {
        throw std::runtime_error("output_format has unrecognized format");
      }
      NaluEnv::self().naluOutputP0()
        << "DataProbePostProcessing::Adding " << formatName
        << " output format..." << std::endl;
    }
    // Enable performance timings of output
    get_if_present(
      y_dataProbe, "time_performance", enablePerfTiming_, enablePerfTiming_);

    // Optional speed-up parameters
    get_if_present(y_dataProbe, "write_coords", writeCoords_, writeCoords_);
    get_if_present(y_dataProbe, "gzip_level", gzLevel_, gzLevel_);

    // extract the frequency of output

    get_if_present(y_dataProbe, "exodus_name", exoName_, exoName_);

    get_if_present(y_dataProbe, "output_frequency", outputFreq_, outputFreq_);

    get_if_present(
      y_dataProbe, "begin_sampling_after", previousTime_, previousTime_);

    bool sampleInTime = false;
    get_if_present(
      y_dataProbe, "sample_based_on_time", sampleInTime, sampleInTime);
    probeType_ = sampleInTime ? DataProbeSampleType::APRXFREQUENCY
                              : DataProbeSampleType::STEPCOUNT;

    if (
      outputFreq_ != static_cast<int>(outputFreq_) &&
      probeType_ == DataProbeSampleType::STEPCOUNT) {
      throw std::runtime_error(
        "output_frequency must be an integer unless sample_based_on_time: on");
    }

    // transfer specifications
    get_if_present(
      y_dataProbe, "search_method", searchMethodName_, searchMethodName_);
    get_if_present(
      y_dataProbe, "search_tolerance", searchTolerance_, searchTolerance_);
    get_if_present(
      y_dataProbe, "search_expansion_factor", searchExpansionFactor_,
      searchExpansionFactor_);

    const YAML::Node y_specs =
      expect_sequence(y_dataProbe, "specifications", true);
    if (y_specs) {

      // each specification can have multiple probes
      for (size_t ispec = 0; ispec < y_specs.size(); ++ispec) {
        const YAML::Node y_spec = y_specs[ispec];

        DataProbeSpecInfo* probeSpec = new DataProbeSpecInfo();
        dataProbeSpecInfo_.push_back(probeSpec);

        DataProbeInfo* probeInfo = new DataProbeInfo();
        probeSpec->dataProbeInfo_.push_back(probeInfo);

        // name; will serve as the transfer name
        const YAML::Node theName = y_spec["name"];
        if (theName)
          probeSpec->xferName_ = theName.as<std::string>();
        else
          throw std::runtime_error("DataProbePostProcessing: no name provided");

        // extract the set of from target names; each spec is homogeneous in
        // this respect
        const YAML::Node& fromTargets = y_spec["from_target_part"];
        if (fromTargets.Type() == YAML::NodeType::Scalar) {
          probeSpec->fromTargetNames_.resize(1);
          probeSpec->fromTargetNames_[0] = fromTargets.as<std::string>();
        } else {
          probeSpec->fromTargetNames_.resize(fromTargets.size());
          for (size_t i = 0; i < fromTargets.size(); ++i) {
            probeSpec->fromTargetNames_[i] = fromTargets[i].as<std::string>();
          }
        }

        // extract the type of probe, e.g., line of site, plane, etc
        const YAML::Node y_loss =
          expect_sequence(y_spec, "line_of_site_specifications", true);
        const YAML::Node y_plane =
          expect_sequence(y_spec, "plane_specifications", true);
        probeInfo->numProbes_ = 0;
        if (y_loss) {

          // l-o-s is active..
          probeInfo->isLineOfSite_ = true;

          // extract and save number of probes
          const int numProbes = probeInfo->numProbes_ + y_loss.size();
          probeInfo->numProbes_ = numProbes;

          // resize everything...
          probeInfo->partName_.resize(numProbes);
          probeInfo->processorId_.resize(numProbes);
          probeInfo->numPoints_.resize(numProbes);
          probeInfo->generateNewIds_.resize(numProbes);
          probeInfo->tipCoordinates_.resize(numProbes);
          probeInfo->tailCoordinates_.resize(numProbes);
          probeInfo->nodeVector_.resize(numProbes);
          probeInfo->part_.resize(numProbes);
          // more resizing
          probeInfo->geomType_.resize(numProbes);
          probeInfo->cornerCoordinates_.resize(numProbes);
          probeInfo->edge1Vector_.resize(numProbes);
          probeInfo->edge2Vector_.resize(numProbes);
          probeInfo->edge1NumPoints_.resize(numProbes);
          probeInfo->edge2NumPoints_.resize(numProbes);
          probeInfo->offsetDir_.resize(numProbes);
          probeInfo->offsetSpacings_.resize(numProbes);
          probeInfo->onlyOutputField_.resize(numProbes);

          // deal with processors... Distribute each probe over subsequent procs
          const int numProcs = NaluEnv::self().parallel_size();
          const int divProcProbe = std::max(numProcs / numProbes, numProcs);

          for (size_t ilos = 0; ilos < y_loss.size(); ilos++) {
            const YAML::Node y_los = y_loss[ilos];

            // Set the geometry type
            probeInfo->geomType_[ilos] = DataProbeGeomType::LINEOFSITE;

            // processor id; distribute los equally over the number of
            // processors
            probeInfo->processorId_[ilos] =
              divProcProbe > 0 ? ilos % divProcProbe : 0;

            // name; which is the part name of choice
            const YAML::Node nameNode = y_los["name"];
            if (nameNode)
              probeInfo->partName_[ilos] = nameNode.as<std::string>();
            else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking the name");

            // number of points
            const YAML::Node numPoints = y_los["number_of_points"];
            if (numPoints)
              probeInfo->numPoints_[ilos] = numPoints.as<int>();
            else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking number of points");

            // coordinates; tip
            const YAML::Node tipCoord = y_los["tip_coordinates"];
            if (tipCoord)
              probeInfo->tipCoordinates_[ilos] =
                tipCoord.as<sierra::nalu::Coordinates>();
            else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking tip coordinates");

            // coordinates; tail
            const YAML::Node tailCoord = y_los["tail_coordinates"];
            if (tailCoord)
              probeInfo->tailCoordinates_[ilos] =
                tailCoord.as<sierra::nalu::Coordinates>();
            else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking tail coordinates");
          }
        }
        if (y_plane) {
          // Get the specifications for defining a sample plane

          // plane is active..
          probeInfo->isSamplePlane_ = true;
          probeInfo->isLineOfSite_ = true;

          // extract and save number of probes
          const int offset = probeInfo->numProbes_;
          const int numProbes = probeInfo->numProbes_ + y_plane.size();
          probeInfo->numProbes_ = numProbes;

          // resize everything...
          probeInfo->partName_.resize(numProbes);
          probeInfo->processorId_.resize(numProbes);
          probeInfo->numPoints_.resize(numProbes);
          probeInfo->generateNewIds_.resize(numProbes);
          probeInfo->tipCoordinates_.resize(numProbes);
          probeInfo->tailCoordinates_.resize(numProbes);
          probeInfo->nodeVector_.resize(numProbes);
          probeInfo->part_.resize(numProbes);
          // more resizing
          probeInfo->geomType_.resize(numProbes);
          probeInfo->cornerCoordinates_.resize(numProbes);
          probeInfo->edge1Vector_.resize(numProbes);
          probeInfo->edge2Vector_.resize(numProbes);
          probeInfo->edge1NumPoints_.resize(numProbes);
          probeInfo->edge2NumPoints_.resize(numProbes);
          probeInfo->offsetDir_.resize(numProbes);
          probeInfo->offsetSpacings_.resize(numProbes);
          probeInfo->onlyOutputField_.resize(numProbes);

          // deal with processors... Distribute each probe over subsequent procs
          const int numProcs = NaluEnv::self().parallel_size();
          const int divProcProbe = std::max(
            numProcs / numProbes,
            numProcs); // unnecessary, divProcProbe = numProcs

          for (size_t iplane = 0; iplane < y_plane.size(); iplane++) {
            const YAML::Node y_planenode = y_plane[iplane];

            // Set the geometry type
            probeInfo->geomType_[iplane + offset] = DataProbeGeomType::PLANE;

            // processor id; distribute los equally over the number of
            // processors
            probeInfo->processorId_[iplane + offset] =
              divProcProbe > 0 ? iplane % divProcProbe : 0;

            // name; which is the part name of choice
            const YAML::Node nameNode = y_planenode["name"];
            if (nameNode) {
              probeInfo->partName_[iplane + offset] =
                nameNode.as<std::string>();
            } else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking the name");

            // number of edge1 points
            const YAML::Node edge1NumPoints = y_planenode["edge1_numPoints"];
            if (edge1NumPoints) {
              probeInfo->edge1NumPoints_[iplane + offset] =
                edge1NumPoints.as<int>();
            } else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking edge 1 number of points");

            // number of edge2 points
            const YAML::Node edge2NumPoints = y_planenode["edge2_numPoints"];
            if (edge2NumPoints) {
              probeInfo->edge2NumPoints_[iplane + offset] =
                edge2NumPoints.as<int>();
            } else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking edge 2 number of points");

            // coordinates; corner
            const YAML::Node cornerCoord = y_planenode["corner_coordinates"];
            if (cornerCoord)
              probeInfo->cornerCoordinates_[iplane + offset] =
                cornerCoord.as<sierra::nalu::Coordinates>();
            else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking corner coordinates");

            // coordinates; edge1
            const YAML::Node edge1Vector = y_planenode["edge1_vector"];
            if (edge1Vector)
              probeInfo->edge1Vector_[iplane + offset] =
                edge1Vector.as<sierra::nalu::Coordinates>();
            else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking edge 1 vector");

            // coordinates; edge2
            const YAML::Node edge2Vector = y_planenode["edge2_vector"];
            if (edge2Vector)
              probeInfo->edge2Vector_[iplane + offset] =
                edge2Vector.as<sierra::nalu::Coordinates>();
            else
              throw std::runtime_error(
                "DataProbePostProcessing: lacking edge 2 vector");

            // coordinates; offsetDir
            const YAML::Node offsetDir = y_planenode["offset_vector"];
            if (offsetDir)
              probeInfo->offsetDir_[iplane + offset] =
                offsetDir.as<sierra::nalu::Coordinates>();
            else {
              probeInfo->offsetDir_[iplane + offset].x_ = 0.0;
              probeInfo->offsetDir_[iplane + offset].y_ = 0.0;
              probeInfo->offsetDir_[iplane + offset].z_ = 0.0;
            }

            // coordinates; offset_spacings
            const YAML::Node offsetSpacings = y_planenode["offset_spacings"];
            if (offsetSpacings)
              probeInfo->offsetSpacings_[iplane + offset] =
                offsetSpacings.as<std::vector<double>>();
            else
              probeInfo->offsetSpacings_[iplane + offset].push_back(0.0);

            // string: onlyOutputField
            const YAML::Node onlyOutputField = y_planenode["only_output_field"];
            if (onlyOutputField)
              probeInfo->onlyOutputField_[iplane + offset] =
                onlyOutputField.as<std::string>() + "_probe";
            else
              probeInfo->onlyOutputField_[iplane + offset] = "";

            // Set the total number of points
            const int numPlanes =
              probeInfo->offsetSpacings_[iplane + offset].size();
            probeInfo->numPoints_[iplane + offset] =
              probeInfo->edge1NumPoints_[iplane + offset] *
              probeInfo->edge2NumPoints_[iplane + offset] * numPlanes;
          }

          // throw std::runtime_error("DataProbePostProcessing: done sample
          // plane setup");
        }

        if (probeInfo->numProbes_ < 1) {
          throw std::runtime_error("DataProbePostProcessing: Need to have some "
                                   "specification included");
        }

        // extract the output variables
        const YAML::Node y_outputs =
          expect_sequence(y_spec, "output_variables", false);
        if (y_outputs) {
          for (size_t ioutput = 0; ioutput < y_outputs.size(); ++ioutput) {
            const YAML::Node y_output = y_outputs[ioutput];

            // find the name, size and type
            const YAML::Node fieldNameNode = y_output["field_name"];
            const YAML::Node fieldSizeNode = y_output["field_size"];

            if (!fieldNameNode)
              throw std::runtime_error("DataProbePostProcessing::load() Sorry, "
                                       "field name must be provided");

            if (!fieldSizeNode)
              throw std::runtime_error("DataProbePostProcessing::load() Sorry, "
                                       "field size must be provided");

            // extract data
            std::string fieldName;
            int fieldSize;
            fieldName = fieldNameNode.as<std::string>();
            fieldSize = fieldSizeNode.as<int>();

            // push to fromToName
            std::string fromName = fieldName;
            std::string toName = fieldName + "_probe";
            probeSpec->fromToName_.push_back(std::make_pair(fromName, toName));

            // push to probeInfo
            std::pair<std::string, int> fieldInfoPair =
              std::make_pair(toName, fieldSize);
            probeSpec->fieldInfo_.push_back(fieldInfoPair);
          }
        }
      }
    }
  }
}

void
DataProbePostProcessing::add_external_data_probe_spec_info(
  DataProbeSpecInfo* dpsInfo)
{
  dataProbeSpecInfo_.push_back(dpsInfo);
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::setup()
{
  // objective: declare the part, register the fields; must be before
  // populate_mesh()

  stk::mesh::MetaData& metaData = realm_.meta_data();

  // first, declare the part
  for (size_t idps = 0; idps < dataProbeSpecInfo_.size(); ++idps) {

    DataProbeSpecInfo* probeSpec = dataProbeSpecInfo_[idps];

    for (size_t k = 0; k < probeSpec->dataProbeInfo_.size(); ++k) {

      DataProbeInfo* probeInfo = probeSpec->dataProbeInfo_[k];

      // loop over probes... one part per probe
      for (int j = 0; j < probeInfo->numProbes_; ++j) {
        // extract name
        std::string partName = probeInfo->partName_[j];

        // declare the part and push it to info; make the part available as a
        // nodeset; check for existance
        probeInfo->part_[j] = metaData.get_part(partName);
        if (NULL == probeInfo->part_[j]) {
          probeInfo->part_[j] =
            &metaData.declare_part(partName, stk::topology::NODE_RANK);
          stk::io::put_io_part_attribute(*probeInfo->part_[j]);
          // part was null, signal for generation of ids
          probeInfo->generateNewIds_[j] = 1;
        } else {
          // part was not null, no ids to be generated
          probeInfo->generateNewIds_[j] = 0;
        }
      }
    }
  }

  // second, always register the fields
  const int nDim = metaData.spatial_dimension();
  for (size_t idps = 0; idps < dataProbeSpecInfo_.size(); ++idps) {

    DataProbeSpecInfo* probeSpec = dataProbeSpecInfo_[idps];

    for (size_t k = 0; k < probeSpec->dataProbeInfo_.size(); ++k) {

      DataProbeInfo* probeInfo = probeSpec->dataProbeInfo_[k];

      // loop over probes... register all fields within the ProbInfo on each
      // part
      for (int p = 0; p < probeInfo->numProbes_; ++p) {

        // extract the part
        stk::mesh::Part* probePart = probeInfo->part_[p];
        // everyone needs coordinates to be registered
        VectorFieldType* coordinates = &(metaData.declare_field<double>(
          stk::topology::NODE_RANK, "coordinates"));
        stk::mesh::put_field_on_mesh(*coordinates, *probePart, nDim, nullptr);
        stk::io::set_field_output_type(
          *coordinates, stk::io::FieldOutputType::VECTOR_3D);
        // now the general set of fields for this probe
        for (size_t j = 0; j < probeSpec->fieldInfo_.size(); ++j) {

          register_field(
            probeSpec->fieldInfo_[j].first, probeSpec->fieldInfo_[j].second,
            metaData, probePart);
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::initialize()
{
  // objective: generate the ids, declare the entity(s) and register the fields;
  // *** must be after populate_mesh() ***
  stk::mesh::BulkData& bulkData = realm_.bulk_data();
  stk::mesh::MetaData& metaData = realm_.meta_data();

  std::vector<std::string> toPartNameVec;
  std::vector<std::string> fromPartNameVec;

  // the call to declare entities requires a high level mesh modification,
  // however, not one per part
  bulkData.modification_begin();

  for (size_t idps = 0; idps < dataProbeSpecInfo_.size(); ++idps) {

    DataProbeSpecInfo* probeSpec = dataProbeSpecInfo_[idps];

    for (size_t k = 0; k < probeSpec->dataProbeInfo_.size(); ++k) {

      DataProbeInfo* probeInfo = probeSpec->dataProbeInfo_[k];

      for (int j = 0; j < probeInfo->numProbes_; ++j) {

        // extract some things off of the probeInfo
        stk::mesh::Part* probePart = probeInfo->part_[j];
        const int numPoints = probeInfo->numPoints_[j];
        const int processorId = probeInfo->processorId_[j];
        const bool generateNewIds = probeInfo->generateNewIds_[j];

        // generate new ids; only if the part was
        std::vector<stk::mesh::EntityId> availableNodeIds(numPoints);
        if (generateNewIds > 0)
          bulkData.generate_new_ids(
            stk::topology::NODE_RANK, numPoints, availableNodeIds);

        // check to see if part has nodes on it already
        if (processorId == NaluEnv::self().parallel_rank()) {

          // set some data
          int checkNumPoints = 0;
          bool nodesExist = false;
          std::vector<stk::mesh::Entity>& nodeVec = probeInfo->nodeVector_[j];

          stk::mesh::Selector s_local_nodes =
            metaData.locally_owned_part() & stk::mesh::Selector(*probePart);

          stk::mesh::BucketVector const& node_buckets =
            bulkData.get_buckets(stk::topology::NODE_RANK, s_local_nodes);
          for (stk::mesh::BucketVector::const_iterator ib =
                 node_buckets.begin();
               ib != node_buckets.end(); ++ib) {
            stk::mesh::Bucket& b = **ib;
            for (stk::mesh::Entity node : b) {
              checkNumPoints++;
              nodeVec.push_back(node);
              nodesExist = true;
            }
          }

          // check if nodes exists. If they do, did the number of points match?
          if (nodesExist) {
            if (checkNumPoints != numPoints) {
              NaluEnv::self().naluOutput()
                << "Number of points specified within input file does not "
                   "match nodes that exists: "
                << probePart->name() << std::endl;
              NaluEnv::self().naluOutput()
                << "The old and new node count is as follows: " << numPoints
                << " " << checkNumPoints << std::endl;
              probeInfo->numPoints_[j] = checkNumPoints;
            }
          } else {
            // only declare entities on which these nodes parallel rank resides
            nodeVec.resize(numPoints);

            // declare the entity on this rank (rank is determined by calling
            // declare_entity on this rank)
            for (int i = 0; i < numPoints; ++i) {
              stk::mesh::Entity theNode = bulkData.declare_entity(
                stk::topology::NODE_RANK, availableNodeIds[i], *probePart);
              nodeVec[i] = theNode;
            }
          }
        }
      }
    }
  }

  bulkData.modification_end();

  // populate values for coord; probe stays the same place
  // FIXME: worry about mesh motion (if the probe moves around?)
  VectorFieldType* coordinates =
    metaData.get_field<double>(stk::topology::NODE_RANK, "coordinates");

  const int nDim = metaData.spatial_dimension();
  for (size_t idps = 0; idps < dataProbeSpecInfo_.size(); ++idps) {

    DataProbeSpecInfo* probeSpec = dataProbeSpecInfo_[idps];

    for (size_t k = 0; k < probeSpec->dataProbeInfo_.size(); ++k) {

      DataProbeInfo* probeInfo = probeSpec->dataProbeInfo_[k];

      for (int j = 0; j < probeInfo->numProbes_; ++j) {

        // reference to the nodeVector
        std::vector<stk::mesh::Entity>& nodeVec = probeInfo->nodeVector_[j];

        // create line-of-site geometry
        if (probeInfo->geomType_[j] == DataProbeGeomType::LINEOFSITE) {
          // populate the coordinates
          double dx[3] = {};

          std::vector<double> tipC(nDim);
          tipC[0] = probeInfo->tipCoordinates_[j].x_;
          tipC[1] = probeInfo->tipCoordinates_[j].y_;

          std::vector<double> tailC(nDim);
          tailC[0] = probeInfo->tailCoordinates_[j].x_;
          tailC[1] = probeInfo->tailCoordinates_[j].y_;
          if (nDim > 2) {
            tipC[2] = probeInfo->tipCoordinates_[j].z_;
            tailC[2] = probeInfo->tailCoordinates_[j].z_;
          }

          const int numPoints = probeInfo->numPoints_[j];
          for (int p = 0; p < nDim; ++p)
            dx[p] = (tipC[p] - tailC[p]) / (double)(std::max(numPoints - 1, 1));

          // now populate the coordinates; can use a simple loop rather than
          // buckets
          for (size_t n = 0; n < nodeVec.size(); ++n) {
            stk::mesh::Entity node = nodeVec[n];
            double* coords = stk::mesh::field_data(*coordinates, node);
            for (int i = 0; i < nDim; ++i)
              coords[i] = tailC[i] + n * dx[i];
          }
          // create sample plane geometry
        } else if (probeInfo->geomType_[j] == DataProbeGeomType::PLANE) {
          double dx[3] = {};
          double dy[3] = {};
          std::vector<double> corner(nDim);
          std::vector<double> edge1(nDim);
          std::vector<double> edge2(nDim);
          std::vector<double> OSdir(nDim);
          corner[0] = probeInfo->cornerCoordinates_[j].x_;
          corner[1] = probeInfo->cornerCoordinates_[j].y_;
          edge1[0] = probeInfo->edge1Vector_[j].x_;
          edge1[1] = probeInfo->edge1Vector_[j].y_;
          edge2[0] = probeInfo->edge2Vector_[j].x_;
          edge2[1] = probeInfo->edge2Vector_[j].y_;
          OSdir[0] = probeInfo->offsetDir_[j].x_;
          OSdir[1] = probeInfo->offsetDir_[j].y_;
          if (nDim > 2) {
            corner[2] = probeInfo->cornerCoordinates_[j].z_;
            edge1[2] = probeInfo->edge1Vector_[j].z_;
            edge2[2] = probeInfo->edge2Vector_[j].z_;
            OSdir[2] = probeInfo->offsetDir_[j].z_;
          }
          const int N1 = probeInfo->edge1NumPoints_[j];
          const int N2 = probeInfo->edge2NumPoints_[j];
          for (int p = 0; p < nDim; ++p) {
            dx[p] = edge1[p] / (double)(std::max(N1 - 1, 1));
            dy[p] = edge2[p] / (double)(std::max(N2 - 1, 1));
          }
          const int pointsPerPlane = N1 * N2;
          const int numPlanes = probeInfo->offsetSpacings_[j].size();
          std::vector<double> OSspacing(numPlanes);
          for (int i = 0; i < numPlanes; i++)
            OSspacing[i] = probeInfo->offsetSpacings_[j][i];

          // now populate the coordinates; can use a simple loop rather than
          // buckets
          for (size_t n = 0; n < nodeVec.size(); ++n) {
            stk::mesh::Entity node = nodeVec[n];
            double* coords = stk::mesh::field_data(*coordinates, node);
            const int planei = n / pointsPerPlane;
            const int localn = n - planei * pointsPerPlane;
            const int indexj = localn / N1;
            const int indexi = localn - indexj * N1;
            for (int i = 0; i < nDim; ++i) {
              coords[i] = corner[i] + indexi * dx[i] + indexj * dy[i] +
                          OSspacing[planei] * OSdir[i];
            }
          }
        }
      }
    }
  }

  create_inactive_selector();
  create_transfer();

  if (useExo_) {
    create_exodus();
  }
}

void
DataProbePostProcessing::create_exodus()
{
  io =
    std::make_unique<stk::io::StkMeshIoBroker>(realm_.bulk_data().parallel());
  io->set_bulk_data(realm_.bulk_data());
  fileIndex_ = io->create_output_mesh(exoName_, stk::io::WRITE_RESULTS);

  for (const auto* probeSpec : dataProbeSpecInfo_) {
    for (const auto& fieldInfo : probeSpec->fieldInfo_) {
      const auto& meta = realm_.meta_data();
      ThrowRequireMsg(
        meta.get_field(stk::topology::NODE_RANK, fieldInfo.first) != nullptr,
        "No field named `" + fieldInfo.first + "' of node rank");
      io->add_field(
        fileIndex_, *meta.get_field(stk::topology::NODE_RANK, fieldInfo.first));
    }
  }
  io->set_subset_selector(fileIndex_, inactiveSelector_);
}

//--------------------------------------------------------------------------
//-------- register_field --------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::register_field(
  const std::string fieldName,
  const int fieldSize,
  stk::mesh::MetaData& metaData,
  stk::mesh::Part* part)
{
  stk::mesh::FieldBase* toField =
    &(metaData.declare_field<double>(stk::topology::NODE_RANK, fieldName));
  stk::mesh::put_field_on_mesh(*toField, *part, fieldSize, nullptr);
}

//--------------------------------------------------------------------------
//-------- create_inactive_selector ----------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::create_inactive_selector()
{
  for (size_t idps = 0; idps < dataProbeSpecInfo_.size(); ++idps) {

    DataProbeSpecInfo* probeSpec = dataProbeSpecInfo_[idps];

    for (size_t k = 0; k < probeSpec->dataProbeInfo_.size(); ++k) {

      DataProbeInfo* probeInfo = probeSpec->dataProbeInfo_[k];

      // loop over probes... one part per probe
      for (int j = 0; j < probeInfo->numProbes_; ++j) {
        allTheParts_.push_back(probeInfo->part_[j]);
      }
    }
  }
  inactiveSelector_ = stk::mesh::selectUnion(allTheParts_);
}

//--------------------------------------------------------------------------
//-------- create_transfer -------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::create_transfer()
{
  stk::mesh::MetaData& metaData = realm_.meta_data();

  // create a [dummy] transfers
  transfers_ = new Transfers(*realm_.root());

  for (size_t idps = 0; idps < dataProbeSpecInfo_.size(); ++idps) {

    DataProbeSpecInfo* probeSpec = dataProbeSpecInfo_[idps];

    // new a transfer and push back
    Transfer* theTransfer = new Transfer(*transfers_);
    transfers_->transferVector_.push_back(theTransfer);

    // set some data on the transfer
    theTransfer->name_ = probeSpec->xferName_;
    theTransfer->fromRealm_ = &realm_;
    theTransfer->toRealm_ = &realm_;
    theTransfer->searchMethodName_ = searchMethodName_;
    theTransfer->searchTolerance_ = searchTolerance_;
    theTransfer->searchExpansionFactor_ = searchExpansionFactor_;

    // provide from/to parts
    for (size_t k = 0; k < probeSpec->dataProbeInfo_.size(); ++k) {

      DataProbeInfo* probeInfo = probeSpec->dataProbeInfo_[k];

      // extract field names (homegeneous over all probes)
      for (size_t j = 0; j < probeSpec->fromToName_.size(); ++j)
        theTransfer->transferVariablesPairName_.push_back(std::make_pair(
          probeSpec->fromToName_[j].first, probeSpec->fromToName_[j].second));

      // accumulate all of the From parts for this Specification
      for (size_t j = 0; j < probeSpec->fromTargetNames_.size(); ++j) {
        std::string fromTargetName = probeSpec->fromTargetNames_[j];
        stk::mesh::Part* fromTargetPart = metaData.get_part(fromTargetName);
        if (NULL == fromTargetPart) {
          throw std::runtime_error(
            "DataProbePostProcessing::create_transfer() Trouble with part, " +
            fromTargetName);
        } else {
          theTransfer->fromPartVec_.push_back(fromTargetPart);
        }
      }

      // accumulate all of the To parts for this Specification (sum over all
      // probes)
      for (int j = 0; j < probeInfo->numProbes_; ++j) {
        theTransfer->toPartVec_.push_back(probeInfo->part_[j]);
      }
    }
  }

  // okay, ready to call through Transfers to do the real work
  transfers_->initialize();
}
//--------------------------------------------------------------------------
//-------- review ----------------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::review(const DataProbeInfo* /* probeInfo */)
{
  // may or may not want this
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::execute()
{
  // only do work if this is an output step
  const double currentTime = realm_.get_current_time();
  const int timeStepCount = realm_.get_time_step_count();
  bool isOutput = false;
  switch (probeType_) {
  case DataProbeSampleType::STEPCOUNT:
    isOutput = timeStepCount % static_cast<int>(outputFreq_) == 0;
    break;
  case DataProbeSampleType::APRXFREQUENCY:
    isOutput = currentTime >= previousTime_ + outputFreq_;
    previousTime_ = isOutput ? currentTime : previousTime_;
    break;
  default:
    std::runtime_error("A DataProbe was not assigned a type.");
    break;
  }

  if (isOutput) {
    const double t1 = enablePerfTiming_ ? NaluEnv::self().nalu_time() : 0.0;
    // execute and provide results...
    transfers_->execute();
    const double t2 = enablePerfTiming_ ? NaluEnv::self().nalu_time() : 0.0;
    if (useExo_) {
      provide_output_exodus(currentTime);
    }
    if (useText_) {
      provide_output_txt(currentTime);
    }
    const double t3 = enablePerfTiming_ ? NaluEnv::self().nalu_time() : 0.0;
    if (enablePerfTiming_)
      NaluEnv::self().naluOutputP0()
        << "DataProbePostProcessing::execute "
        << " transfer_time: " << t2 - t1 << " output_time: " << t3 - t2
        << " total_time: " << t3 - t1 << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- provide_output --------------------------------------------------
//--------------------------------------------------------------------------
void
DataProbePostProcessing::provide_output_txt(const double currentTime)
{
  NaluEnv::self().naluOutputP0()
    << "DataProbePostProcessing::Writing dataprobes..." << std::endl;

  stk::mesh::MetaData& metaData = realm_.meta_data();
  VectorFieldType* coordinates =
    metaData.get_field<double>(stk::topology::NODE_RANK, "coordinates");

  const int nDim = metaData.spatial_dimension();

  for (size_t idps = 0; idps < dataProbeSpecInfo_.size(); ++idps) {

    DataProbeSpecInfo* probeSpec = dataProbeSpecInfo_[idps];

    for (size_t k = 0; k < probeSpec->dataProbeInfo_.size(); ++k) {

      DataProbeInfo* probeInfo = probeSpec->dataProbeInfo_[k];

      for (int inp = 0; inp < probeInfo->numProbes_; ++inp) {

        if (probeInfo->geomType_[inp] == DataProbeGeomType::LINEOFSITE) {

          // open the file for this probe
          const int processorId = probeInfo->processorId_[inp];
          std::ostringstream ss;
          ss << processorId;
          const std::string fileName =
            probeInfo->partName_[inp] + "_" + ss.str() + ".dat";
          std::ofstream myfile;
          if (processorId == NaluEnv::self().parallel_rank()) {

            // Get the path to the file name, and create any directories
            // necessary
#ifdef NALU_USES_BOOST
            boost::filesystem::path pathdir{fileName};
            if (pathdir.has_parent_path()) {
              if (!boost::filesystem::exists(pathdir.parent_path().string())) {
                try {
                  boost::filesystem::create_directories(
                    pathdir.parent_path().string());
                } catch (const boost::filesystem::filesystem_error& e) {
                  NaluEnv::self().naluOutputP0()
                    << "Error creating " << pathdir.parent_path().string()
                    << std::endl;
                  NaluEnv::self().naluOutputP0()
                    << e.code().message() << std::endl;
                  throw std::runtime_error(e.code().message());
                }
              }
            }
#endif

            // one banner per file
            const bool addBanner =
              std::ifstream(fileName.c_str()) ? false : true;

            myfile.open(fileName.c_str(), std::ios_base::app);

            // provide banner for current time, coordinates, field 1, field 2,
            // etc
            if (addBanner) {
              myfile << "Time" << std::setw(w_);

              for (int jj = 0; jj < nDim; ++jj)
                myfile << "coordinates[" << jj << "]" << std::setw(w_);

              for (size_t ifi = 0; ifi < probeSpec->fieldInfo_.size(); ++ifi) {
                const std::string fieldName = probeSpec->fieldInfo_[ifi].first;
                const int fieldSize = probeSpec->fieldInfo_[ifi].second;

                for (int jj = 0; jj < fieldSize; ++jj) {
                  myfile << fieldName << "[" << jj << "]" << std::setw(w_);
                }
              }

              // banner complete
              myfile << std::endl;
            }

            // reference to the nodeVector
            std::vector<stk::mesh::Entity>& nodeVec =
              probeInfo->nodeVector_[inp];

            // output in a single row
            for (size_t inv = 0; inv < nodeVec.size(); ++inv) {
              stk::mesh::Entity node = nodeVec[inv];
              double* theCoord = stk::mesh::field_data(*coordinates, node);

              // always output time and coordinates
              myfile << std::left << std::setw(w_)
                     << std::setprecision(precisionvar_) << currentTime
                     << std::setw(w_);
              for (int jj = 0; jj < nDim; ++jj) {
                myfile << theCoord[jj] << std::setw(w_);
              }

              // now all of the other fields required
              for (size_t ifi = 0; ifi < probeSpec->fieldInfo_.size(); ++ifi) {
                const std::string fieldName = probeSpec->fieldInfo_[ifi].first;
                const stk::mesh::FieldBase* theField =
                  metaData.get_field(stk::topology::NODE_RANK, fieldName);
                double* theF = (double*)stk::mesh::field_data(*theField, node);

                const int fieldSize = probeSpec->fieldInfo_[ifi].second;
                for (int jj = 0; jj < fieldSize; ++jj) {
                  myfile << theF[jj] << std::setw(w_);
                }
              }
              // node output complete
              myfile << std::endl;
            }
            // all nodal output is complete, close
            myfile.close();
          } else {
            // nothing to do for this probe on this processor
          }
        } else if (probeInfo->geomType_[inp] == DataProbeGeomType::PLANE) {
          // -- Output the plane in text file --

          // open the file for this probe
          const int timeStepCount = realm_.get_time_step_count();
          const int processorId = probeInfo->processorId_[inp];
          const int gzlevel = gzLevel_;
          const bool printcoords = writeCoords_; // false;
          std::ostringstream ss;
          ss << std::setw(7) << std::setfill('0') << timeStepCount;
          ss << "_" << processorId;
          std::string fileName =
            probeInfo->partName_[inp] + "_" + ss.str() + ".dat";
          if (processorId == NaluEnv::self().parallel_rank()) {

            const int N1 = probeInfo->edge1NumPoints_[inp];
            const int N2 = probeInfo->edge2NumPoints_[inp];
            const int pointsPerPlane = N1 * N2;

// Use gzip compression when writing
#ifdef NALU_USES_BOOST
            boost::iostreams::filtering_streambuf<boost::iostreams::output>
              outbuf;
            if ((0 < gzlevel) && (gzlevel < 10)) {
              fileName = fileName + ".gz";
              outbuf.push(
                boost::iostreams::gzip_compressor(boost::iostreams::gzip_params(
                  gzlevel, boost::iostreams::zlib::deflated, 15, 9,
                  boost::iostreams::zlib::huffman_only)));
            }
#endif

            // Get the path to the file name, and create any directories
            // necessary
#ifdef NALU_USES_BOOST
            boost::filesystem::path pathdir{fileName};
            if (pathdir.has_parent_path()) {
              if (!boost::filesystem::exists(pathdir.parent_path().string())) {
                try {
                  boost::filesystem::create_directories(
                    pathdir.parent_path().string());
                } catch (const boost::filesystem::filesystem_error& e) {
                  NaluEnv::self().naluOutputP0()
                    << "Error creating " << pathdir.parent_path().string()
                    << std::endl;
                  NaluEnv::self().naluOutputP0()
                    << e.code().message() << std::endl;
                  throw std::runtime_error(e.code().message());
                }
              }
            }
#endif

            // ** Check to see if we need to add a coordinate file
            std::string coordFileName =
              probeInfo->partName_[inp] + "_coordXYZ.dat";
            if (!printcoords) {
              if ((0 < gzlevel) && (gzlevel < 10))
                coordFileName += ".gz";
              const bool addCoordFile =
                std::ifstream(coordFileName.c_str()) ? false : true;
              if (addCoordFile) {
                // -- Add the coordinate file
#ifdef NALU_USES_BOOST
                boost::iostreams::filtering_streambuf<boost::iostreams::output>
                  outstream;
                if ((0 < gzlevel) && (gzlevel < 10)) {
                  outstream.push(boost::iostreams::gzip_compressor(
                    boost::iostreams::gzip_params(gzlevel)));
                }
                std::ofstream coordfile(
                  coordFileName.c_str(), std::ios_base::out);
                outstream.push(coordfile);
                std::ostream myfile(&outstream);
                // -- output the header
                myfile << "#Time: " << std::setprecision(precisionvar_)
                       << currentTime << std::endl;
                myfile << "# ";
                myfile << std::setw(w_ - 1) << std::right << "Plane_Number"
                       << std::setw(w_) << std::right << "Index_j"
                       << std::setw(w_) << std::right << "Index_i";
                for (int jj = 0; jj < nDim; ++jj) {
                  myfile << std::setw(w_ - 2) << std::right << "coordinates["
                         << jj << "]";
                }
                myfile << '\n';
                // -- Done with header
                // reference to the nodeVector
                std::vector<stk::mesh::Entity>& nodeVec =
                  probeInfo->nodeVector_[inp];
                // -- output indices and coordinates in a single row
                for (size_t inv = 0; inv < nodeVec.size(); ++inv) {
                  stk::mesh::Entity node = nodeVec[inv];
                  double* theCoord =
                    (double*)stk::mesh::field_data(*coordinates, node);
                  // Output plane indices
                  const int planei = inv / pointsPerPlane;
                  const int localn = inv - planei * pointsPerPlane;
                  const int indexj = localn / N1;
                  const int indexi = localn - indexj * N1;
                  myfile << std::right << std::setw(w_) << planei
                         << std::setw(w_) << indexj << std::setw(w_) << indexi
                         << std::setw(w_);
                  // Output coordinates
                  for (int jj = 0; jj < nDim; ++jj) {
                    myfile << std::setprecision(precisionvar_) << theCoord[jj]
                           << std::setw(w_);
                  }
                  myfile << '\n';
                }

                boost::iostreams::close(outstream);
                coordfile.close();
#endif
                // -- Done with the coordinate file
              }
            }

            std::ofstream file(fileName.c_str(), std::ios_base::out);
#ifdef NALU_USES_BOOST
            outbuf.push(file);
#endif
            std::string filestring;
            std::string coordfilestring("");
            char buffer[1000];
            filestring.reserve(5000000);

            if (!printcoords)
              coordfilestring = "CoordinateFile: " + coordFileName;

            snprintf(
              buffer, 1000, "#Time: %18.12e %s\n#", currentTime,
              coordfilestring.c_str());
            filestring.append(buffer);
            if (printcoords) {
              filestring += "Plane_Number Index_j Index_i";
              for (int jj = 0; jj < nDim; ++jj) {
                snprintf(buffer, 1000, " coordinates[%i]", jj);
                filestring.append(buffer);
              }
            }
            for (size_t ifi = 0; ifi < probeSpec->fieldInfo_.size(); ++ifi) {
              const std::string fieldName = probeSpec->fieldInfo_[ifi].first;
              if (
                (probeInfo->onlyOutputField_[inp] == "") ||
                (probeInfo->onlyOutputField_[inp] == fieldName)) {
                const int fieldSize = probeSpec->fieldInfo_[ifi].second;
                for (int jj = 0; jj < fieldSize; ++jj) {
                  snprintf(buffer, 1000, " %s[%i]", fieldName.c_str(), jj);
                  filestring.append(buffer);
                }
              }
            }
            filestring += '\n';

            // Get some pointers to all of the data files
            std::vector<stk::mesh::FieldBase*> allFields(
              probeSpec->fieldInfo_.size());
            std::vector<size_t> fieldSize;
            std::vector<std::string> allFieldNames;
            for (size_t ifi = 0; ifi < probeSpec->fieldInfo_.size(); ++ifi) {
              const std::string fieldName = probeSpec->fieldInfo_[ifi].first;
              allFieldNames.push_back(fieldName);
              (allFields[ifi]) =
                metaData.get_field(stk::topology::NODE_RANK, fieldName);
              fieldSize.push_back(probeSpec->fieldInfo_[ifi].second);
            }

            // reference to the nodeVector
            std::vector<stk::mesh::Entity>& nodeVec =
              probeInfo->nodeVector_[inp];

            // output in a single row
            for (size_t inv = 0; inv < nodeVec.size(); ++inv) {
              stk::mesh::Entity node = nodeVec[inv];
              // only output coordinates if required
              if (printcoords) {
                double* theCoord = stk::mesh::field_data(*coordinates, node);
                // Output plane indices
                const int planei = inv / pointsPerPlane;
                const int localn = inv - planei * pointsPerPlane;
                const int indexj = localn / N1;
                const int indexi = localn - indexj * N1;

                snprintf(
                  buffer, 1000, "%18i %18i %18i", planei, indexj, indexi);
                filestring.append(buffer);

                // Output coordinates
                for (int jj = 0; jj < nDim; ++jj) {
                  snprintf(buffer, 1000, " %12.5e", theCoord[jj]);
                  filestring.append(buffer);
                }
              }
              // now all of the other fields required
              for (size_t ifi = 0; ifi < probeSpec->fieldInfo_.size(); ++ifi) {

                if (
                  (probeInfo->onlyOutputField_[inp] == "") ||
                  (probeInfo->onlyOutputField_[inp] == allFieldNames[ifi])) {
                  double* theF =
                    (double*)stk::mesh::field_data(*(allFields[ifi]), node);
                  for (size_t jj = 0; jj < fieldSize[ifi]; ++jj) {
                    snprintf(buffer, 1000, " %12.6e", theF[jj]);
                    filestring.append(buffer);
                  }
                }
              }
              // row complete
              filestring += '\n';
            }
#ifdef NALU_USES_BOOST
            // done with file output
            std::ostream fileout(&outbuf);
            fileout << filestring;
            boost::iostreams::close(outbuf); // Don't forget this!
#endif
            file.close();

          } // END if ( processorId == NaluEnv::self().parallel_rank())
        }
      }
    }
  }
}

void
DataProbePostProcessing::provide_output_exodus(const double currentTime)
{
  NaluEnv::self().naluOutputP0()
    << "DataProbePostProcessing::Writing dataprobes..." << std::endl;
  io->process_output_request(fileIndex_, currentTime);
}

//--------------------------------------------------------------------------
//-------- get_inactive_selector -------------------------------------------
//--------------------------------------------------------------------------
stk::mesh::Selector&
DataProbePostProcessing::get_inactive_selector()
{
  return inactiveSelector_;
}

} // namespace nalu
} // namespace sierra
