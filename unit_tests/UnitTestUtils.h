#ifndef _UnitTestUtils_h_
#define _UnitTestUtils_h_

#include <array>
#include <string>
#include <ostream>
#include <random>
#include <cmath>
#include <math.h>

#include <SimdInterface.h>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_io/IossBridge.hpp>

#include <Kokkos_Core.hpp>

#include <master_element/Hex8CVFEM.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>
#include <FieldManager.h>
#include <SmartField.h>

#include <FieldTypeDef.h>

#include <gtest/gtest.h>

using IdFieldType = sierra::nalu::ScalarFieldType;

namespace unit_test_utils {

void fill_mesh_1_elem_per_proc_hex8(stk::mesh::BulkData& bulk);
void fill_hex8_mesh(const std::string& meshSpec, stk::mesh::BulkData& bulk);
void
perturb_coord_hex_8(stk::mesh::BulkData& bulk, double perturbationSize = 0.125);

void dump_mesh(
  stk::mesh::BulkData& bulk,
  std::vector<stk::mesh::FieldBase*> fields,
  std::string name = "out.e");

std::ostream& nalu_out();

stk::mesh::Entity
create_one_reference_element(stk::mesh::BulkData& bulk, stk::topology topo);
stk::mesh::Entity
create_one_perturbed_element(stk::mesh::BulkData& bulk, stk::topology topo);

double quadratic(double a, const double* b, const double* H, const double* x);

double vector_norm(
  const std::vector<double>& vec,
  const stk::ParallelMachine& comm = MPI_COMM_WORLD);

double global_norm(
  const double& norm,
  const size_t& N,
  const stk::ParallelMachine& comm = MPI_COMM_WORLD);

double initialize_quadratic_scalar_field(
  const stk::mesh::BulkData& bulk,
  const sierra::nalu::VectorFieldType& coordField,
  const sierra::nalu::ScalarFieldType& qField);

std::array<double, 9> random_rotation_matrix(int dim, std::mt19937& rng);
std::array<double, 9>
random_linear_transformation(int dim, double scale, std::mt19937& rng);

} // namespace unit_test_utils

const double tol = 1.e-10;

class Hex8Mesh : public ::testing::Test
{
protected:
  Hex8Mesh()
    : comm(MPI_COMM_WORLD),
      spatialDimension(3),
      topo(stk::topology::HEX_8),
      partVec(),
      coordField(nullptr),
      exactLaplacian(0.0)
  {
    const int numStates = 2;
    stk::mesh::MeshBuilder meshBuilder(comm);
    meshBuilder.set_spatial_dimension(spatialDimension);
    bulk = meshBuilder.create();
    meta = &bulk->mesh_meta_data();
    meta->use_simple_fields();
    fieldManager =
      std::make_shared<sierra::nalu::FieldManager>(*meta, numStates);

    double one = 1.0;
    double zero = 0.0;
    double zeroVec[3] = {0.0, 0.0, 0.0};
    const stk::mesh::PartVector parts(1, &meta->universal_part());
    elemCentroidField =
      fieldManager->register_field<double>("elemCentroid", parts, zeroVec);
    nodalPressureField =
      fieldManager->register_field<double>("nodalPressure", parts, &one);
    discreteLaplacianOfPressure =
      fieldManager->register_field<double>("discreteLaplacian", parts, &zero);
    scalarQ = fieldManager->register_field<double>("scalarQ", parts, &zero);
    diffFluxCoeff =
      fieldManager->register_field<double>("diffFluxCoeff", parts, &zero);
    idField = fieldManager->register_field<double>("idField", parts, &zero);
  }

  ~Hex8Mesh() {}

  void fill_mesh(const std::string& meshSpec = "generated:20x20x20")
  {
    unit_test_utils::fill_hex8_mesh(meshSpec, *bulk);
  }

  void fill_mesh_and_initialize_test_fields(
    std::string meshSpec = "generated:20x20x20",
    const bool generateSidesets = false)
  {
    if (generateSidesets)
      meshSpec += "|sideset:xXyYzZ";

    fill_mesh(meshSpec);

    partVec = {meta->get_part("block_1")};

    coordField = static_cast<const sierra::nalu::VectorFieldType*>(
      meta->coordinate_field());
    EXPECT_TRUE(coordField != nullptr);

    exactLaplacian = unit_test_utils::initialize_quadratic_scalar_field(
      *bulk, *coordField, *nodalPressureField);
    stk::mesh::field_fill(0.0, *discreteLaplacianOfPressure);
    stk::mesh::field_fill(0.1, *scalarQ);
    stk::mesh::field_fill(0.2, *diffFluxCoeff);
  }

  void check_discrete_laplacian(double exactLaplacian);

  stk::ParallelMachine comm;
  unsigned spatialDimension;
  stk::mesh::MetaData* meta;
  std::shared_ptr<stk::mesh::BulkData> bulk;
  std::shared_ptr<sierra::nalu::FieldManager> fieldManager;
  stk::topology topo;
  sierra::nalu::VectorFieldType* elemCentroidField;
  sierra::nalu::ScalarFieldType* nodalPressureField;
  sierra::nalu::ScalarFieldType* discreteLaplacianOfPressure;
  sierra::nalu::ScalarFieldType* scalarQ;
  sierra::nalu::ScalarFieldType* diffFluxCoeff;
  IdFieldType* idField;
  stk::mesh::PartVector partVec;
  const sierra::nalu::VectorFieldType* coordField;
  double exactLaplacian;
};

class Hex8MeshWithNSOFields : public Hex8Mesh
{
protected:
  Hex8MeshWithNSOFields();

  sierra::nalu::GenericFieldType* massFlowRate;
  sierra::nalu::GenericFieldType* Gju;
  sierra::nalu::VectorFieldType* velocity;
  sierra::nalu::VectorFieldType* dpdx;
  sierra::nalu::GenericFieldType* exposedAreaVec;
  sierra::nalu::ScalarFieldType* density;
  sierra::nalu::ScalarFieldType* viscosity;
  sierra::nalu::ScalarFieldType* pressure;
  sierra::nalu::ScalarFieldType* udiag;
  sierra::nalu::ScalarFieldType* dnvField;
};

class Hex8ElementWithBCFields : public ::testing::Test
{
protected:
  Hex8ElementWithBCFields()
  {
    const double one = 1.0;
    const double oneVecThree[3] = {one, one, one};
    const double oneVecFour[4] = {one, one, -one, -one};
    const double oneVecNine[9] = {one, one, one, one, one, one, one, one, one};
    const double oneVecTwelve[12] = {one, one, one, one, one, one,
                                     one, one, one, one, one, one};

    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(3);
    bulk = meshBuilder.create();
    meta = &bulk->mesh_meta_data();
    meta->use_simple_fields();

    velocity =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "velocity");
    bcVelocity = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "wall_velocity_bc");
    density = &meta->declare_field<double>(stk::topology::NODE_RANK, "density");
    viscosity =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "viscosity");
    bcHeatFlux =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "heat_flux_bc");
    specificHeat =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "specific_heat");
    exposedAreaVec =
      &meta->declare_field<double>(meta->side_rank(), "exposed_area_vector");
    wallFrictionVelocityBip = &meta->declare_field<double>(
      meta->side_rank(), "wall_friction_velocity_bip");
    wallNormalDistanceBip = &meta->declare_field<double>(
      meta->side_rank(), "wall_normal_distance_bip");
    bcVelocityOpen = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "open_velocity_bc");
    openMdot =
      &meta->declare_field<double>(meta->side_rank(), "open_mass_flow_rate");
    Gjui = &meta->declare_field<double>(stk::topology::NODE_RANK, "dudx");
    scalarQ =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "scalar_q");
    bcScalarQ =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "bc_scalar_q");
    Gjq = &meta->declare_field<double>(stk::topology::NODE_RANK, "Gjq");

    stk::mesh::put_field_on_mesh(
      *velocity, meta->universal_part(), 3, oneVecThree);
    stk::io::set_field_output_type(
      *velocity, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *bcVelocity, meta->universal_part(), 3, oneVecThree);
    stk::io::set_field_output_type(
      *bcVelocity, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(*density, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*viscosity, meta->universal_part(), &one);
    stk::mesh::put_field_on_mesh(*bcHeatFlux, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *specificHeat, meta->universal_part(), nullptr);

    const sierra::nalu::MasterElement* meFC =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        stk::topology::QUAD_4);
    stk::mesh::put_field_on_mesh(
      *exposedAreaVec, meta->universal_part(),
      3 * meFC->num_integration_points(), oneVecTwelve);
    stk::mesh::put_field_on_mesh(
      *wallFrictionVelocityBip, meta->universal_part(),
      meFC->num_integration_points(), nullptr);
    stk::mesh::put_field_on_mesh(
      *wallNormalDistanceBip, meta->universal_part(),
      meFC->num_integration_points(), nullptr);

    stk::mesh::put_field_on_mesh(
      *bcVelocityOpen, meta->universal_part(), 3, oneVecThree);
    stk::io::set_field_output_type(
      *bcVelocityOpen, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *openMdot, meta->universal_part(), 4, oneVecFour);
    stk::mesh::put_field_on_mesh(
      *Gjui, meta->universal_part(), 3 * 3, oneVecNine);
    stk::io::set_field_output_type(
      *Gjui, stk::io::FieldOutputType::FULL_TENSOR_36);

    stk::mesh::put_field_on_mesh(*scalarQ, meta->universal_part(), &one);
    stk::mesh::put_field_on_mesh(*bcScalarQ, meta->universal_part(), &one);
    stk::mesh::put_field_on_mesh(*Gjq, meta->universal_part(), 3, oneVecThree);
    stk::io::set_field_output_type(*Gjq, stk::io::FieldOutputType::VECTOR_3D);

    unit_test_utils::create_one_reference_element(
      *bulk, stk::topology::HEXAHEDRON_8);
  }

  ~Hex8ElementWithBCFields() {}

  stk::mesh::MetaData* meta;
  std::shared_ptr<stk::mesh::BulkData> bulk;
  sierra::nalu::VectorFieldType* velocity;
  sierra::nalu::VectorFieldType* bcVelocity;
  sierra::nalu::ScalarFieldType* density;
  sierra::nalu::ScalarFieldType* viscosity;
  sierra::nalu::ScalarFieldType* bcHeatFlux;
  sierra::nalu::ScalarFieldType* specificHeat;
  sierra::nalu::GenericFieldType* exposedAreaVec;
  sierra::nalu::GenericFieldType* wallFrictionVelocityBip;
  sierra::nalu::GenericFieldType* wallNormalDistanceBip;
  sierra::nalu::VectorFieldType* bcVelocityOpen;
  sierra::nalu::GenericFieldType* openMdot;
  sierra::nalu::TensorFieldType* Gjui;
  sierra::nalu::ScalarFieldType* scalarQ;
  sierra::nalu::ScalarFieldType* bcScalarQ;
  sierra::nalu::VectorFieldType* Gjq;
};

class CylinderMesh : public ::testing::Test
{
protected:
  CylinderMesh()
    : comm(MPI_COMM_WORLD),
      spatialDimension(3),
      xMax(0),
      yMax(0),
      zMax(0),
      topo(stk::topology::HEX_8),
      coordField(nullptr)
  {
    stk::mesh::MeshBuilder meshBuilder(comm);
    meshBuilder.set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
    meshBuilder.set_spatial_dimension(spatialDimension);
    bulk = meshBuilder.create();
    meta = &bulk->mesh_meta_data();
    meta->use_simple_fields();

    testField =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "testField");
    curCoords_ = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "current_coordinates");
    meshDisp_ = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "mesh_displacement");

    deflectionRamp_ =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "deflection_ramp");
    dispMap_ = &meta->declare_field<int>(stk::topology::NODE_RANK, "disp_map");
    dispMapInterp_ =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "disp_map_interp");
    loadMap_ = &meta->declare_field<int>(stk::topology::NODE_RANK, "load_map");
    loadMapInterp_ =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "load_map_interp");
    tforceSCS_ =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "tforce_scs");
    mesh_displacement_ref_ = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "mesh_displacement_ref");
    mesh_velocity_ref_ = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "mesh_velocity_ref");
    div_mesh_velocity_ = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "div_mesh_velocity");
    density_ = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "density", 3 /*num-states*/);
    pressure_ =
      &meta->declare_field<double>(stk::topology::NODE_RANK, "pressure");
    viscosity_ = &meta->declare_field<double>(
      stk::topology::NODE_RANK, "effective_viscosity_u");
    exposedAreaVec_ =
      &meta->declare_field<double>(meta->side_rank(), "exposed_area_vector");
    dudx_ = &meta->declare_field<double>(stk::topology::NODE_RANK, "dudx");

    const double zeroVecThree[3] = {0.0, 0.0, 0.0};
    stk::mesh::put_field_on_mesh(
      *testField, meta->universal_part(), 3, zeroVecThree);
    stk::io::set_field_output_type(
      *testField, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *curCoords_, meta->universal_part(), 3, zeroVecThree);
    stk::io::set_field_output_type(
      *curCoords_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *meshDisp_, meta->universal_part(), 3, zeroVecThree);
    stk::io::set_field_output_type(
      *meshDisp_, stk::io::FieldOutputType::VECTOR_3D);

    stk::mesh::put_field_on_mesh(
      *deflectionRamp_, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*dispMap_, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dispMapInterp_, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*loadMap_, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *loadMapInterp_, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*tforceSCS_, meta->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *mesh_displacement_ref_, meta->universal_part(), 3, nullptr);
    stk::io::set_field_output_type(
      *mesh_displacement_ref_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *mesh_velocity_ref_, meta->universal_part(), 3, nullptr);
    stk::io::set_field_output_type(
      *mesh_velocity_ref_, stk::io::FieldOutputType::VECTOR_3D);
    stk::mesh::put_field_on_mesh(
      *div_mesh_velocity_, meta->universal_part(), nullptr);
    constexpr double one = 1.0;
    stk::mesh::put_field_on_mesh(*density_, meta->universal_part(), &one);
    stk::mesh::put_field_on_mesh(*pressure_, meta->universal_part(), &one);
    stk::mesh::put_field_on_mesh(*viscosity_, meta->universal_part(), &one);
    const sierra::nalu::MasterElement* meFC =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        stk::topology::QUAD_4);
    const double oneVecTwelve[12] = {one, one, one, one, one, one,
                                     one, one, one, one, one, one};
    const double oneVecNine[9] = {one, one, one, one, one, one, one, one, one};
    stk::mesh::put_field_on_mesh(
      *exposedAreaVec_, meta->universal_part(),
      3 * meFC->num_integration_points(), oneVecTwelve);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta->universal_part(), 3 * 3, oneVecNine);

    meta->enable_late_fields();
  }

  void fill_mesh(const std::string& meshSpec = "generated:20x20x20")
  {
    unit_test_utils::fill_hex8_mesh(meshSpec, *bulk);
  }

  void fill_mesh_and_initialize_test_fields(
    int xDim,
    int yDim,
    int zDim,
    double innerRad,
    double outerRad,
    const bool generateSidesets = true)
  {
    std::ostringstream oss;
    oss << "generated:" << xDim << "x" << yDim << "x" << zDim;
    std::string meshSpec = oss.str();

    xMax = xDim;
    yMax = yDim;
    zMax = zDim;

    if (generateSidesets) {
      meshSpec += "|sideset:xXyYzZ";
    }

    fill_mesh(meshSpec);
    coordField = static_cast<const sierra::nalu::VectorFieldType*>(
      meta->coordinate_field());
    EXPECT_TRUE(coordField != nullptr);

    transform_to_cylinder(innerRad, outerRad);

    stk::mesh::field_fill(0.1, *testField);

    coordField->modify_on_host();
    testField->modify_on_host();
    stk::mesh::communicate_field_data(*bulk, {coordField, testField});
  }

  void transform_to_cylinder(double innerRad, double outerRad)
  {
    stk::mesh::Selector sel =
      (meta->locally_owned_part() | meta->globally_shared_part());
    const stk::mesh::BucketVector& bkts =
      bulk->get_buckets(stk::topology::NODE_RANK, sel);

    const double xfac = (outerRad - innerRad) / xMax;
    const double yfac = 2 * M_PI / yMax;
    auto nodeCoord =
      sierra::nalu::MakeSmartField<tags::LEGACY, tags::READ_WRITE>()(
        const_cast<sierra::nalu::VectorFieldType*>(coordField));

    for (const stk::mesh::Bucket* bptr : bkts) {
      for (stk::mesh::Entity node : *bptr) {
        const double radius = innerRad + nodeCoord(node)[0] * xfac;
        const double theta = nodeCoord(node)[1] * yfac;
        nodeCoord(node)[0] = radius * std::cos(theta);
        nodeCoord(node)[1] = radius * std::sin(theta);
      }
    }
  }

  stk::ParallelMachine comm;
  unsigned spatialDimension;
  int xMax, yMax, zMax;
  stk::mesh::MetaData* meta;
  std::shared_ptr<stk::mesh::BulkData> bulk;
  stk::topology topo;
  const sierra::nalu::VectorFieldType* coordField;
  sierra::nalu::VectorFieldType* testField;

  sierra::nalu::VectorFieldType* curCoords_;
  sierra::nalu::VectorFieldType* meshDisp_;
  sierra::nalu::ScalarFieldType* deflectionRamp_;
  sierra::nalu::ScalarIntFieldType* dispMap_;
  sierra::nalu::ScalarFieldType* dispMapInterp_;
  sierra::nalu::GenericIntFieldType* loadMap_;
  sierra::nalu::GenericFieldType* loadMapInterp_;
  sierra::nalu::GenericFieldType* tforceSCS_;
  sierra::nalu::VectorFieldType* mesh_displacement_ref_;
  sierra::nalu::VectorFieldType* mesh_velocity_ref_;
  sierra::nalu::ScalarFieldType* div_mesh_velocity_;
  sierra::nalu::ScalarFieldType* density_;
  sierra::nalu::ScalarFieldType* pressure_;
  sierra::nalu::ScalarFieldType* viscosity_;
  sierra::nalu::GenericFieldType* exposedAreaVec_;
  sierra::nalu::GenericFieldType* dudx_;
};

class ABLWallFunctionHex8ElementWithBCFields : public Hex8ElementWithBCFields
{
public:
  ABLWallFunctionHex8ElementWithBCFields()
    : Hex8ElementWithBCFields(),
      rhoSpec_(1.0),
      utauSpec_(0.1),
      upSpec_(1.0),
      ypSpec_(0.25)
  {
  }

  using ::testing::Test::SetUp;

  void
  SetUp(const double& rho, const double& utau, const double up, const double yp)
  {
    rhoSpec_ = rho;
    utauSpec_ = utau;
    upSpec_ = up;
    ypSpec_ = yp;

    // create an object for creating SmartField's
    sierra::nalu::MakeSmartField<tags::LEGACY, tags::READ_WRITE> smartener;

    // Assign some values to the nodal fields
    // all these fields will sync_to_host here and call modified_on_host when
    // they go out of scope
    auto smrtDensity = smartener(density);
    auto smrtVelocity = smartener(velocity);
    auto smrtBcVelocity = smartener(bcVelocity);
    auto smrtBcHeatFlux = smartener(bcHeatFlux);
    auto smrtSpecificHeat = smartener(specificHeat);
    for (const auto* ib :
         bulk->get_buckets(stk::topology::NODE_RANK, meta->universal_part())) {
      const auto& b = *ib;
      const size_t length = b.size();
      for (size_t k = 0; k < length; ++k) {
        stk::mesh::Entity node = b[k];
        *smrtDensity(node) = rhoSpec_;
        double* vel = smrtVelocity(node);
        vel[0] = upSpec_;
        vel[1] = 0.0;
        vel[2] = 0.0;
        double* bcVel = smrtBcVelocity(node);
        bcVel[0] = 0.0;
        bcVel[1] = 0.0;
        bcVel[3] = 0.0;
        *smrtBcHeatFlux(node) = 0.0;
        *smrtSpecificHeat(node) = 1000.0;
      }
    }

    // Assign some values to the boundary integration point fields
    auto utauIp = smartener(wallFrictionVelocityBip);
    auto ypIp = smartener(wallNormalDistanceBip);

    const sierra::nalu::MasterElement* meFC =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        stk::topology::QUAD_4);
    const int numScsBip = meFC->num_integration_points();
    stk::mesh::BucketVector const& face_buckets =
      bulk->get_buckets(meta->side_rank(), meta->universal_part());
    for (stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
         ib != face_buckets.end(); ++ib) {
      stk::mesh::Bucket& b = **ib;
      const size_t length = b.size();
      for (size_t k = 0; k < length; ++k) {
        stk::mesh::Entity face = b[k];
        for (int ip = 0; ip < numScsBip; ++ip) {
          utauIp(face)[ip] = utauSpec_;
          ypIp(face)[ip] = ypSpec_;
        }
      }
    }
  }

private:
  double rhoSpec_;
  double utauSpec_;
  double upSpec_;
  double ypSpec_;
};

#endif
