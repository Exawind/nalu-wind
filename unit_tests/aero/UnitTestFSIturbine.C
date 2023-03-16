#include "gtest/gtest.h"

#include "stk_util/environment/WallTime.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

#include "UnitTestUtils.h"

#include "KokkosInterface.h"
#include "aero/fsi/FSIturbine.h"
#include "aero/fsi/OpenfastFSI.h"

namespace {

// using block_1 and surface_1 for everything is no doubt silly...
// as we ramp up the FSI testing we will no doubt need to use
// more sophisticated input. But this gets us off the
// ground, so to speak.
const std::string fsiInputs = "tower_parts: [block_1] \n"
                               "hub_parts: [block_1]\n"
                               "nacelle_parts: [block_1] \n"
                               "blade_parts:\n"
                               "  - [block_1]\n"
                               "  - [block_1]\n"
                               "deflection_ramping:\n"
                               "  span_ramp_distance: 10.0\n"
                               "  zero_theta_ramp_angle: 180.0\n"
                               "  theta_ramp_span: 15.0\n"
                               "  temporal_ramp_start: 0\n"
                               "  temporal_ramp_end: 10\n"
                               "tower_boundary_parts: [surface_1] \n"
                               "hub_boundary_parts: [surface_1]\n"
                               "nacelle_boundary_parts: [surface_1] \n"
                               "blade_boundary_parts:\n"
                               "  - [surface_1]\n";

const std::string openfastFSIinputs = "openfast_fsi:\n"
                                      "n_turbines_glob: 1\n"
                                      "dry_run:  True\n"
                                      "debug:    True\n"
                                      "sim_start: init\n"
                                      "t_start: 0.0\n"
                                      "t_max:   13.106159895150718\n"
                                      "n_checkpoint: 144\n"
                                      "dt_FAST: 0.0011376874908984999\n"
                                      "n_substeps: 40\n"
                                      "Turbine0:\n"
                                      "  turbine_base_pos: [5.0191, 0., -89.56256]\n"
                                      "  turbine_hub_pos: [5.0191, 0.0, 0.0 ]\n"
                                      "  FAST_input_filename: nrel5mw.fst\n"
                                      "  sim_type: ext-loads\n"
                                      "  tower_parts: [block_1] \n"
                                      "  hub_parts: [block_1]\n"
                                      "  nacelle_parts: [block_1] \n"
                                      "  blade_parts:\n"
                                      "    - [block_1]\n"
                                      "    - [block_1]\n"
                                      "  deflection_ramping:\n"
                                      "    span_ramp_distance: 10.0\n"
                                      "    zero_theta_ramp_angle: 180.0\n"
                                      "    theta_ramp_span: 15.0\n"
                                      "    temporal_ramp_start: 0\n"
                                      "    temporal_ramp_end: 10\n"
                                      "  tower_boundary_parts: [surface_2] \n"
                                      "  hub_boundary_parts: [surface_1]\n"
                                      "  nacelle_boundary_parts: [surface_1] \n"
                                      "  blade_boundary_parts:\n"
                                      "    - [surface_1]\n";

YAML::Node
create_fsi_yaml_node()
{
  YAML::Node yamlNode = YAML::Load(fsiInputs);
  return yamlNode;
}

YAML::Node
create_openfastFSI_yaml_node()
{
  YAML::Node yamlNode = YAML::Load(openfastFSIinputs);
  return yamlNode;
}

void get_mesh_bounding_box(const stk::mesh::BulkData& mesh,
                           const stk::mesh::Selector& selector,
                           vs::Vector& minCoords, vs::Vector& maxCoords)
{
  const stk::mesh::BucketVector& nodeBuckets = mesh.get_buckets(stk::topology::NODE_RANK, selector);
  minCoords.x() = std::numeric_limits<double>::max();
  minCoords.y() = std::numeric_limits<double>::max();
  minCoords.z() = std::numeric_limits<double>::max();
  maxCoords.x() = std::numeric_limits<double>::min();
  maxCoords.y() = std::numeric_limits<double>::min();
  maxCoords.z() = std::numeric_limits<double>::min();

  const stk::mesh::FieldBase* nodeCoordField = mesh.mesh_meta_data().coordinate_field();
  const unsigned spatialDim = mesh.mesh_meta_data().spatial_dimension();

  for(const stk::mesh::Bucket* bptr : nodeBuckets) {
    const double* nodeCoords = reinterpret_cast<const double*>(stk::mesh::field_data(*nodeCoordField, *bptr));
    for(unsigned i=0; i<bptr->size(); ++i) {
      minCoords.x() = std::min(minCoords.x(), nodeCoords[i*spatialDim]);
      minCoords.y() = std::min(minCoords.y(), nodeCoords[i*spatialDim+1]);
      minCoords.z() = std::min(minCoords.z(), nodeCoords[i*spatialDim+2]);

      maxCoords.x() = std::max(maxCoords.x(), nodeCoords[i*spatialDim]);
      maxCoords.y() = std::max(maxCoords.y(), nodeCoords[i*spatialDim+1]);
      maxCoords.z() = std::max(maxCoords.z(), nodeCoords[i*spatialDim+2]);
    }
  }
}

void set_tower_ref_pos(const stk::mesh::BulkData& mesh, sierra::nalu::fsiTurbine& fsiTurb)
{
  fast::turbineDataType& params = fsiTurb.params_;
  vs::Vector minCoords, maxCoords;
  stk::mesh::Selector towerPart = *mesh.mesh_meta_data().get_part("block_1");
  get_mesh_bounding_box(mesh, towerPart, minCoords, maxCoords);
  std::cout<<"twr min: ("<<minCoords.x()<<","<<minCoords.y()<<","<<minCoords.z()<<")"
           <<"; twr max: ("<<maxCoords.x()<<","<<maxCoords.y()<<","<<maxCoords.z()<<")"<<std::endl;

  const unsigned numIntervals = static_cast<unsigned>(maxCoords.z()-minCoords.z());
  const unsigned numNodes = numIntervals + 1;
  params.nBRfsiPtsTwr = numNodes;
  params.numBlades = 0;

  fsiTurb.initialize();
  fast::turbBRfsiDataType& brFSIdata = fsiTurb.brFSIdata_;
  std::cout<<"brFSIdata.twr_ref_pos.size(): "<<brFSIdata.twr_ref_pos.size()<<std::endl;

  const double deltaZ = (maxCoords.z() - minCoords.z()) / numIntervals;
  double z =minCoords.z();
  for(unsigned n=0; n<numNodes; ++n) {
    double centerX = (maxCoords.x() + minCoords.x()) / 2;
    double centerY = (maxCoords.y() + minCoords.y()) / 2;

    brFSIdata.twr_ref_pos[n*6] = centerX;
    brFSIdata.twr_ref_pos[n*6 + 1] = centerY;
    brFSIdata.twr_ref_pos[n*6 + 2] = z;

    z += deltaZ;
  }
}

void set_x_mesh_disp(const stk::mesh::BulkData& mesh, double val)
{
  const stk::mesh::FieldBase* meshDispField = mesh.mesh_meta_data().get_field(stk::topology::NODE_RANK, "mesh_displacement");
  EXPECT_TRUE(meshDispField != nullptr);
  const unsigned spatialDim = mesh.mesh_meta_data().spatial_dimension();

  stk::mesh::Selector meshDispSelector(*meshDispField);
  const stk::mesh::BucketVector& nodeBuckets = mesh.get_buckets(stk::topology::NODE_RANK, meshDispSelector);
  for(const stk::mesh::Bucket* bptr : nodeBuckets) {
    double* meshDispData = reinterpret_cast<double*>(stk::mesh::field_data(*meshDispField, *bptr));
    for(unsigned n=0; n<bptr->size(); ++n) {
      meshDispData[n*spatialDim] = val;
    }
  }
}

void set_x_values(std::vector<double>& vec, double val)
{
  constexpr unsigned dofsPerNode = 6;
  size_t offset = 0;
  for(unsigned i=0; i<vec.size(); i+=dofsPerNode) {
    vec[i] = val;
  }
}

TEST_F(CylinderMesh, construct_FSIturbine)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) { GTEST_SKIP(); }

  const double innerRadius = 1.0;
  const double outerRadius = 2.0;
  fill_mesh_and_initialize_test_fields(20, 20, 20, innerRadius, outerRadius);

  YAML::Node yamlNode = create_fsi_yaml_node();
  sierra::nalu::fsiTurbine fsiTurb(0, yamlNode);
  EXPECT_NO_THROW(fsiTurb.setup(bulk));
}

TEST_F(CylinderMesh, construct_OpenfastFSI)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) { GTEST_SKIP(); }

  const double innerRadius = 1.0;
  const double outerRadius = 2.0;
  fill_mesh_and_initialize_test_fields(20, 20, 20, innerRadius, outerRadius);

  YAML::Node yamlNode = create_openfastFSI_yaml_node();
  sierra::nalu::OpenfastFSI openfastFSI(yamlNode);
  const double dtNalu = 6.25e-3;
  EXPECT_NO_THROW(openfastFSI.setup(dtNalu, bulk));
  EXPECT_NO_THROW(openfastFSI.end_openfast());
}

TEST_F(CylinderMesh, call_fsiTurbine_mapLoads)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) { GTEST_SKIP(); }

  const double innerRadius = 1.0;
  const double outerRadius = 2.0;
  fill_mesh_and_initialize_test_fields(20, 20, 20, innerRadius, outerRadius);

  YAML::Node yamlNode = create_openfastFSI_yaml_node();
  sierra::nalu::OpenfastFSI openfastFSI(yamlNode);
  const double dtNalu = 6.25e-3;
  EXPECT_NO_THROW(openfastFSI.setup(dtNalu, bulk));
  std::cout<<"nTurbinesGlob: "<<openfastFSI.get_nTurbinesGlob()<<std::endl;
  EXPECT_EQ(1, openfastFSI.get_nTurbinesGlob());

  const int turbIndex = 0;
  sierra::nalu::fsiTurbine* fsiTurb = openfastFSI.get_fsiTurbineData(turbIndex);
  EXPECT_TRUE(fsiTurb != nullptr);
  fast::turbineDataType& params = fsiTurb->params_;

  EXPECT_EQ(0, params.nBRfsiPtsTwr);
  EXPECT_EQ(0, params.numBlades);

  set_tower_ref_pos(*bulk, *fsiTurb);

  fsiTurb->computeLoadMapping();

  //stk::mesh::field_fill(0.1, *tforceSCS_);
  set_x_mesh_disp(*bulk, 0.1);
  set_x_values(fsiTurb->brFSIdata_.twr_def, 0.1);

  fsiTurb->mapLoads();

  constexpr double expectedEndLoad = -120;
  constexpr double expectedInteriorLoad = -240;
  fast::turbBRfsiDataType& brFSIdata = fsiTurb->brFSIdata_;
  const std::vector<double>& twrLoad = brFSIdata.twr_ld;
  
  for(unsigned n=0; n<params.nBRfsiPtsTwr; ++n) {
    const bool isEndNode = n==0 || n==(params.nBRfsiPtsTwr-1);
    if (isEndNode) {
      EXPECT_EQ(expectedEndLoad, twrLoad[n*6]);
      EXPECT_EQ(expectedEndLoad, twrLoad[n*6 + 1]);
      EXPECT_EQ(expectedEndLoad, twrLoad[n*6 + 2]);
    }
  }

  EXPECT_NO_THROW(openfastFSI.end_openfast());
}

} // namespace
