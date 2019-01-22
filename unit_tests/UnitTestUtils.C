#include <gtest/gtest.h>
#include <NaluEnv.h>

#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/SkinBoundary.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_topology/topology.hpp>

#include <master_element/TensorOps.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedElementIO.h>
#include <nalu_make_unique.h>

#include "UnitTestUtils.h"
#include "UnitTestKokkosUtils.h"

#include <algorithm>
#include <string>
#include <array>
#include <random>

namespace unit_test_utils {

void fill_mesh_1_elem_per_proc_hex8(stk::mesh::BulkData& bulk)
{
    int nprocs = bulk.parallel_size();
    std::string meshSpec = "generated:1x1x"+std::to_string(nprocs);
    fill_hex8_mesh(meshSpec, bulk);
}

void perturb_coord_hex_8(stk::mesh::BulkData& bulk, double perturbSize)
{

  // ensure that the rng isn't machine/compiler dependent.
  struct Lcg {
    Lcg(uint32_t seed) : prev(seed) {};

    double operator()()
    {
      prev = (a * static_cast<uint64_t>(prev) + c) % m;
      return (2. * (prev / static_cast<double>(m)) - 1.);
    }
    uint32_t a{1103515245};
    uint32_t c{12345};
    uint32_t m{2147483647};
    uint32_t prev;
  };
  Lcg lcg(bulk.parallel_rank() + 1);

  const auto& meta = bulk.mesh_meta_data();
  const VectorFieldType* coordField = dynamic_cast<const VectorFieldType*>(meta.coordinate_field());
  ThrowRequire(coordField != nullptr);

  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, meta.locally_owned_part())) {
    const auto& b = *ib;
    double* coords = stk::mesh::field_data(*coordField, b);
    for (size_t k = 0u; k < b.size(); ++k) {
      size_t offset = k * meta.spatial_dimension();
      for (unsigned d = 0; d < meta.spatial_dimension(); ++d) {
        coords[offset + d] += perturbSize * lcg();
      }
    }
  }
  stk::mesh::copy_owned_to_shared(bulk, {coordField});
}

void fill_hex8_mesh(const std::string& meshSpec, stk::mesh::BulkData& bulk)
{
    auto& meta = bulk.mesh_meta_data();
    auto& surfPart = meta.declare_part_with_topology("surface_1", stk::topology::QUAD_4);

    stk::io::StkMeshIoBroker io(bulk.parallel());
    io.set_bulk_data(bulk);
    io.add_mesh_database(meshSpec, stk::io::READ_MESH);
    io.create_input_mesh();
    io.populate_bulk_data();

    auto& blockPart = meta.get_topology_root_part(stk::topology::HEX_8);
    stk::mesh::create_exposed_block_boundary_sides(bulk,  blockPart, {&surfPart});
}

void fill_and_promote_hex_mesh(const std::string& meshSpec, stk::mesh::BulkData& bulk, int polyOrder)
{
    stk::io::StkMeshIoBroker io(bulk.parallel());
    io.set_bulk_data(bulk);
    io.add_mesh_database(meshSpec, stk::io::READ_MESH);
    io.create_input_mesh();

    stk::mesh::MetaData& meta = bulk.mesh_meta_data();
    stk::mesh::Part* blockPart = meta.get_part("block_1");
    stk::mesh::Part* surfPart = &meta.declare_part_with_topology("surface_1", stk::topology::QUAD_4);

    auto elemDesc = sierra::nalu::ElementDescription::create(3,polyOrder);

    const std::string superName = sierra::nalu::super_element_part_name("block_1");
    stk::topology topo = stk::create_superelement_topology(static_cast<unsigned>(elemDesc->nodesPerElement));
    meta.declare_part_with_topology(superName, topo);

    stk::mesh::Part* superSuperPart =
        &meta.declare_part(sierra::nalu::super_element_part_name("surface_1"), stk::topology::FACE_RANK);

    const auto sidePartName = sierra::nalu::super_subset_part_name("surface_1");
    auto sideTopo = stk::create_superface_topology(static_cast<unsigned>(elemDesc->nodesPerSide));
    stk::mesh::Part* superSidePart = &meta.declare_part_with_topology(sidePartName, sideTopo);
    meta.declare_part_subset(*superSuperPart, *superSidePart);

    stk::mesh::Part* edgePart = &meta.declare_part("edge_part", stk::topology::EDGE_RANK);
    stk::mesh::Part* facePart = &meta.declare_part("face_part", stk::topology::FACE_RANK);

    io.populate_bulk_data();
    stk::mesh::create_exposed_block_boundary_sides(bulk, *blockPart, {surfPart});

    VectorFieldType* coords = meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::PartVector baseParts = {blockPart, surfPart};
    sierra::nalu::promotion::promote_elements(bulk, *elemDesc, *coords, baseParts, edgePart, facePart);
}

void dump_mesh(stk::mesh::BulkData& bulk, std::vector<stk::mesh::FieldBase*> fields, std::string name)
{
  stk::io::StkMeshIoBroker io(bulk.parallel());
  io.set_bulk_data(bulk);
  auto fileId = io.create_output_mesh(name, stk::io::WRITE_RESULTS);

  for (auto* field : fields) {
    io.add_field(fileId, *field);
  }
  io.process_output_request(fileId, 0.0);
}

void dump_promoted_mesh_file(stk::mesh::BulkData& bulk, int polyOrder)
{
    const auto& meta = bulk.mesh_meta_data();
    const stk::mesh::PartVector& outParts = meta.get_mesh_parts();
    std::string fileName = "out.e" ;

    auto desc = sierra::nalu::ElementDescription::create(meta.spatial_dimension(), polyOrder);
    VectorFieldType* coordField = meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

    auto io = sierra::nalu::make_unique<sierra::nalu::PromotedElementIO>(
      *desc,
      meta,
      bulk,
      outParts,
      fileName,
      *coordField
    );
    io->write_database_data(0.0);
}

std::ostream& nalu_out()
{
  return sierra::nalu::NaluEnv::self().naluOutput();
}

stk::mesh::Entity create_one_element(
  stk::mesh::BulkData& bulk,
  stk::topology topo, const
  std::vector<std::vector<double>>& nodeLocations)
{
  // create just one element
   auto& meta = bulk.mesh_meta_data();
   stk::mesh::Part& block_1 = meta.declare_part_with_topology("block_1", topo);
   stk::io::put_io_part_attribute(block_1);

   stk::mesh::PartVector allSurfaces = { &meta.declare_part("all_surfaces", meta.side_rank()) };
   stk::io::put_io_part_attribute(*allSurfaces.front());

   stk::mesh::PartVector individualSurfaces(topo.num_sides());
   for (unsigned k = 0u; k < topo.num_sides(); ++k) {
     individualSurfaces[k] = &meta.declare_part_with_topology("surface_" + std::to_string(k), topo.side_topology(k));
   }

   // set a coordinate field
   using vector_field_type = stk::mesh::Field<double, stk::mesh::Cartesian3d>;
   auto& coordField = meta.declare_field<vector_field_type>(stk::topology::NODE_RANK, "coordinates");
   stk::mesh::put_field_on_mesh(coordField, block_1, nullptr);
   stk::mesh::put_field_on_mesh(coordField, stk::mesh::selectUnion(allSurfaces), nullptr);
   meta.set_coordinate_field(&coordField);
   meta.commit();

   stk::mesh::EntityIdVector nodeIds(topo.num_nodes());
   std::iota(nodeIds.begin(), nodeIds.end(), 1);

   
   bulk.modification_begin();

   for (auto id : nodeIds) {
     bulk.declare_entity(stk::topology::NODE_RANK, id, stk::mesh::PartVector{});
   }
   auto elem = stk::mesh::declare_element(bulk, block_1, bulk.parallel_rank()+1, nodeIds);
   stk::mesh::create_all_sides(bulk, block_1, allSurfaces, false);

   bulk.modification_end();

   auto surfaceSelector = stk::mesh::selectUnion(allSurfaces);
   stk::mesh::EntityVector all_faces;
   stk::mesh::get_selected_entities(surfaceSelector, bulk.get_buckets(meta.side_rank(), surfaceSelector), all_faces);
   ThrowRequire(all_faces.size() == topo.num_sides());

   bulk.modification_begin();
   for (unsigned k = 0u; k < all_faces.size(); ++k) {
     const int ordinal = bulk.begin_element_ordinals(all_faces[k])[0];
     bulk.change_entity_parts(all_faces[k], {individualSurfaces[ordinal]}, stk::mesh::PartVector{});
   }
   bulk.modification_end();

   const auto* nodes = bulk.begin_nodes(elem);
   for (unsigned j = 0; j  < bulk.num_nodes(elem); ++j) {
     for (unsigned i = 0; i < topo.dimension(); ++i) {
       stk::mesh::field_data(coordField, nodes[j])[i] = nodeLocations.at(j).at(i);
     }
   }

   return elem;
}

stk::mesh::Entity create_one_reference_quad4_element(stk::mesh::BulkData& bulk)
{
  std::vector<std::vector<double>> nodeLocations =
  {
      {-0.5,-0.5}, {+0.5,-0.5},
      {+0.5,+0.5}, {-0.5,+0.5}
  };
  return create_one_element(bulk, stk::topology::QUADRILATERAL_4_2D, nodeLocations);
}

stk::mesh::Entity create_one_reference_quad9_element(stk::mesh::BulkData& bulk)
{
  std::vector<std::vector<double>> nodeLocations =
  {
      {-1.0,-1.0}, {+1.0,-1.0},
      {+1.0,+1.0}, {-1.0,+1.0},
      {0.0, -1.0}, {+1.0, 0.0}, {0.0, +1.0}, {-1.0, 0.0},
      {0.0, 0.0}
  };
  return create_one_element(bulk, stk::topology::QUADRILATERAL_9_2D, nodeLocations);
}

stk::mesh::Entity create_one_reference_tri3_element(stk::mesh::BulkData& bulk)
{
  std::vector<std::vector<double>> nodeLocations =
  {
      {0.0,0.0}, {1.0,0}, {0.0,1.0}
  };
  return create_one_element(bulk, stk::topology::TRIANGLE_3_2D, nodeLocations);
}

stk::mesh::Entity create_one_reference_tet4_element(stk::mesh::BulkData& bulk)
{
  std::vector<std::vector<double>> nodeLocations =
  {
      {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
  };
   return create_one_element(bulk, stk::topology::TET_4, nodeLocations);
}

stk::mesh::Entity create_one_reference_hex8_element(stk::mesh::BulkData& bulk)
{
   std::vector<std::vector<double>> nodeLocations =
   {
       {-0.5,-0.5,-0.5}, {+0.5,-0.5,-0.5}, {+0.5,+0.5,-0.5}, {-0.5,+0.5,-0.5},
       {-0.5,-0.5,+0.5}, {+0.5,-0.5,+0.5}, {+0.5,+0.5,+0.5}, {-0.5,+0.5,+0.5}
   };
   return create_one_element(bulk, stk::topology::HEX_8, nodeLocations);
}

stk::mesh::Entity create_one_reference_hex27_element(stk::mesh::BulkData& bulk)
{
   std::vector<std::vector<double>> nodeLocations =
   {
       {-1.0,-1.0,-1.0}, {+1.0,-1.0,-1.0}, {+1.0,+1.0,-1.0}, {-1.0,+1.0,-1.0},
       {-1.0,-1.0,+1.0}, {+1.0,-1.0,+1.0}, {+1.0,+1.0,+1.0}, {-1.0,+1.0,+1.0},
       {+0.0,-1.0,-1.0}, {+1.0,+0.0,-1.0}, {+0.0,+1.0,-1.0}, {-1.0,+0.0,-1.0},
       {-1.0,-1.0,+0.0}, {+1.0,-1.0,+0.0}, {+1.0,+1.0, 0.0}, {-1.0,+1.0,+0.0},
       {+0.0,-1.0,+1.0}, {+1.0,+0.0,+1.0}, {+0.0,+1.0,+1.0}, {-1.0,+0.0,+1.0},
       {+0.0,+0.0,+0.0},
       {+0.0,+0.0,-1.0}, {+0.0,+0.0,+1.0},
       {-1.0,+0.0,+0.0}, {+1.0,+0.0,+0.0},
       {+0.0,-1.0,+0.0}, {+0.0,+1.0,+0.0}
   };
   return create_one_element(bulk, stk::topology::HEX_27, nodeLocations);
}

stk::mesh::Entity create_one_reference_pyramid5_element(stk::mesh::BulkData& bulk)
{
  std::vector<std::vector<double>> nodeLocations =
  {
      {-1.0, -1.0, +0.0}, {+1.0, -1.0, +0.0}, {+1.0, +1.0, +0.0}, {-1.0, +1.0, +0.0},
      {0.0, 0.0, +1.0}
  };
  return create_one_element(bulk, stk::topology::PYRAMID_5, nodeLocations);
}

stk::mesh::Entity create_one_reference_wedge6_element(stk::mesh::BulkData& bulk)
{
  std::vector<std::vector<double>> nodeLocations =
  {
      {0.0,0.0,-1.0}, {+1.0,0.0,-1.0}, {0.0,+1.0,-1.0},
      {0.0,0.0,+1.0}, {+1.0,0.0,+1.0}, {0.0,+1.0,+1.0}
  };
  return create_one_element(bulk, stk::topology::WEDGE_6, nodeLocations);
}


stk::mesh::Entity create_one_reference_element(stk::mesh::BulkData& bulk, stk::topology topo)
{
  switch (topo.value())
  {
    case stk::topology::TRIANGLE_3_2D:
      return create_one_reference_tri3_element(bulk);

    case stk::topology::QUADRILATERAL_4_2D:
      return create_one_reference_quad4_element(bulk);

    case stk::topology::QUADRILATERAL_9_2D:
      return create_one_reference_quad9_element(bulk);

    case stk::topology::TETRAHEDRON_4:
      return create_one_reference_tet4_element(bulk);

    case stk::topology::PYRAMID_5:
      return create_one_reference_pyramid5_element(bulk);

    case stk::topology::WEDGE_6:
      return create_one_reference_wedge6_element(bulk);

    case stk::topology::HEXAHEDRON_8:
      return create_one_reference_hex8_element(bulk);

    case stk::topology::HEXAHEDRON_27:
      return create_one_reference_hex27_element(bulk);

    default:
      EXPECT_TRUE(false) << " Element type " + topo.name() + " not implemented";
      return stk::mesh::Entity(stk::mesh::Entity::InvalidEntity);
  }
}


stk::mesh::Entity create_one_perturbed_element(stk::mesh::BulkData& bulk, stk::topology topo)
{

  std::vector<std::vector<double>> nodeLocations{{}};

  switch (topo.value())
  {
    case stk::topology::TRIANGLE_3_2D:
       nodeLocations = {
          {0.0464223, 0.172133},
          {1.17897, 0.173626},
          {0.0617818, 0.942191}
      };
      break;

    case stk::topology::QUADRILATERAL_4_2D:
      nodeLocations = {
         {-0.453578, -0.327867},
         {0.678973, -0.326374},
         {0.561782, 0.442191},
         {-0.601233, 0.278356}
     };
     break;

    case stk::topology::QUADRILATERAL_9_2D:
      nodeLocations = {
         {-0.953578, -0.827867},
         {1.17897, -0.826374},
         {1.06178, 0.942191},
         {-1.10123, 0.778356},
         {-0.113672, -1.01117},
         {1.15608, -0.0100114},
         {-0.0536076, 1.16804},
         {-1.0813, 0.0740859},
         {-0.0658792, 0.228578}
     };
     break;

    case stk::topology::TETRAHEDRON_4:
      nodeLocations = {
         {0.0464223, 0.172133, 0.178973},
         {1.17363, 0.0617818, -0.0578091},
         {-0.101233, 0.778356, -0.113672},
         {-0.0111674, 0.156084, 0.989989}
     };
     break;

    case stk::topology::PYRAMID_5:
      nodeLocations = {
         {-0.953578, -0.827867, 0.178973},
         {1.17363, -0.938218, -0.0578091},
         {0.898767, 0.778356, -0.113672},
         {-1.01117, 1.15608, -0.0100114},
         {-0.0536076, 0.168039, 0.918698}
     };
     break;

    case stk::topology::WEDGE_6:
      nodeLocations = {
         {0.0464223, 0.172133, -0.821027},
         {1.17363, 0.0617818, -1.05781},
         {-0.101233, 0.778356, -1.11367},
         {-0.0111674, 0.156084, 0.989989},
         {0.946392, 0.168039, 0.918698},
         {0.0740859, 0.934121, 1.22858}
     };
     break;


    case stk::topology::HEXAHEDRON_8:
      nodeLocations = {
         {-0.453578, -0.327867, -0.321027},
         {0.673626, -0.438218, -0.557809},
         {0.398767, 0.278356, -0.613672},
         {-0.511167, 0.656084, -0.510011},
         {-0.553608, -0.331961, 0.418698},
         {0.574086, -0.565879, 0.728578},
         {0.320175, 0.685044, 0.486804},
         {-0.349545, 0.510239, 0.58944}
     };
     break;

    case stk::topology::HEXAHEDRON_27:
      nodeLocations = {
         {-0.953578, -0.827867, -0.821027},
         {1.17363, -0.938218, -1.05781},
         {0.898767, 0.778356, -1.11367},
         {-1.01117, 1.15608, -1.01001},
         {-1.05361, -0.831961, 0.918698},
         {1.07409, -1.06588, 1.22858},
         {0.820175, 1.18504, 0.986804},
         {-0.849545, 1.01024, 1.08944},
         {0.110316, -0.95899, -0.981313},
         {1.12931, -0.197046, -1.0132},
         {-0.156834, 1.11846, -1.14172},
         {-1.18239, -0.0879295, -1.17516},
         {-1.13884, -1.05676, 0.201299},
         {0.974975, -0.943468, 0.201174},
         {0.79964, 1.2349, 0.07657},
         {-1.16455, 0.929076, 0.125343},
         {0.0539153, -1.08748, 0.769213},
         {1.06714, 0.229475, 1.0764},
         {0.0675294, 1.24765, 1.04093},
         {-1.04282, -0.0126512, 1.06176},
         {-0.0809962, 0.0873762, -0.0913991},
         {0.139173, 0.224786, -0.918737},
         {-0.243214, 0.061423, 1.08683},
         {-0.764028, 0.189097, 0.00481219},
         {0.777857, -0.0244204, -0.240006},
         {-0.0291445, -0.760207, -0.0702778},
         {-0.00955323, 1.09433, 0.190238}
     };
     break;

    default:
      EXPECT_TRUE(false) << " Element type " + topo.name() + " not implemented";
      return stk::mesh::Entity(stk::mesh::Entity::InvalidEntity);
  }

  return create_one_element(bulk, topo, nodeLocations);
}

double linear(double a, const double* b, const double* x) {
  return (a + b[0]*x[0] + b[1]*x[1] + b[2]*x[2]);
}

double quadratic(double a, const double* b, const double* H, const double* x)
{
  double quad = x[0] * (H[0]*x[0] + H[1]*x[1] + H[2]*x[2])
              + x[1] * (H[3]*x[0] + H[4]*x[1] + H[5]*x[2])
              + x[2] * (H[6]*x[0] + H[7]*x[1] + H[8]*x[2]);

  return (linear(a,b,x) + 0.5*quad);
}

double vector_norm(const std::vector<double> & vec, const stk::ParallelMachine& comm)
{
  size_t N = vec.size();
  size_t g_N = 0;
  double norm = 0.0;
  double g_norm = 0.0;

  for (size_t i = 0; i < N; ++i) {
    norm += vec[i]*vec[i];
  }
  stk::all_reduce_sum(comm, &N, &g_N, 1);
  stk::all_reduce_sum(comm, &norm, &g_norm, 1);
  g_norm = std::sqrt(g_norm/g_N);

  return g_norm;
}

double global_norm(const double & norm, const size_t & N, const stk::ParallelMachine& comm)
{
  size_t g_N = 0;
  double g_norm = 0.0;

  stk::all_reduce_sum(comm, &N, &g_N, 1);
  stk::all_reduce_sum(comm, &norm, &g_norm, 1);
  g_norm = std::sqrt(g_norm/g_N);

  return g_norm;
}

double initialize_linear_scalar_field(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordField,
  const ScalarFieldType& qField)
{
  // q = a + b^T x + 1/2 x^T H x
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double a  = coeff(rng);

  double b[3];
  for (unsigned j = 0; j < 3; ++j) {
    b[j] = coeff(rng);
  }

  const auto& meta = bulk.mesh_meta_data();
  EXPECT_EQ(meta.spatial_dimension(), 3u);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : buckets)
  {
    for(stk::mesh::Entity node : *bptr)
    {
      const double* coords = stk::mesh::field_data(coordField, node);
      *stk::mesh::field_data(qField, node) = linear(a, b, coords);
    }
  }

  return std::sqrt(b[0] * b[0] + b[1]* b[1] + b[2] * b[2]);
}

double initialize_quadratic_scalar_field(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordField,
  const ScalarFieldType& qField)
{
  // q = a + b^T x + 1/2 x^T H x
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double a  = coeff(rng);

  double b[3];
  for (unsigned j = 0; j < 3; ++j) {
    b[j] = coeff(rng);
  }

  double H[9];
  for (unsigned j = 0; j < 9; ++j) {
    H[j] = coeff(rng);
  }
  H[3] = H[1];
  H[6] = H[2];
  H[7] = H[5];

  const auto& meta = bulk.mesh_meta_data();
  EXPECT_EQ(meta.spatial_dimension(), 3u);

  const stk::mesh::Selector selector = meta.locally_owned_part() | meta.globally_shared_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : buckets)
  {
    for(stk::mesh::Entity node : *bptr)
    {
      const double* coords = stk::mesh::field_data(coordField, node);
      *stk::mesh::field_data(qField, node) = quadratic(a,b,H, coords);
    }
  }

  double traceOfHessian = H[0] + H[4] + H[8];

  return (traceOfHessian);
}

std::array<double,9> random_rotation_matrix(int dim, std::mt19937& rng)
{
  // always 3D, but set rotation around Z so only the relevant 2x2 subset matters
  std::uniform_real_distribution<double> angle(0, 2*M_PI);
  std::uniform_real_distribution<double> axis(0, 1);

  const double theta = angle(rng);
  const double cosTheta = std::cos(theta);
  const double sinTheta = std::sin(theta);
  std::array<double,3> n = {{(dim==3) ? axis(rng) : 0, (dim==3) ? axis(rng) : 0, (dim==3) ? axis(rng) : 1}};

  if( dim == 3 ) {
    double nMag = 0.0;
    for (int j = 0; j < 3; ++j) {
      nMag += n[j]*n[j];
    }
    nMag = 1.0/std::sqrt(nMag);
    for (int j = 0; j < dim; ++j) {
      n[j] *= nMag;
    }
  }

  std::array<double, 9> nX = {{
      0,    -n[2],+n[1],
      +n[2],    0,-n[0],
      -n[1], +n[0],   0
  }};

  std::array<double, 9> rot = {{
      cosTheta, 0, 0,
      0, cosTheta, 0,
      0, 0, cosTheta
  }};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      rot[j*3+i] += (1-cosTheta)*n[i]*n[j] + sinTheta*nX[j*3+i];
    }
  }
  return rot;
}

std::array<double,9> random_linear_transformation(int dim, double scale, std::mt19937& rng)
{
  std::uniform_real_distribution<double> entry(-scale, scale);

  if (dim == 3) {
    std::array<double,9> Q = {{
        3*scale, 0, 0,
        0, 3*scale, 0,
        0, 0, 3*scale
    }};

    for (auto& coeff : Q) {
      coeff += entry(rng);
    }

    std::array<double, 9> rot = random_rotation_matrix(dim, rng);

    std::array<double, 9> QR = {{}};
    sierra::nalu::mxm33(Q.data(), rot.data(), QR.data());

    return QR;
  }
  else {
    std::array<double,4> Q = {{
        2*scale, 0,
        0, 2*scale,
    }};

    for (auto& coeff : Q) {
      coeff += entry(rng);
    }

    std::array<double, 9> rot = random_rotation_matrix(dim, rng);

    std::array<double, 9> QR = {{}};
    sierra::nalu::mxm22(Q.data(), rot.data(), QR.data());

    return QR;
  }
}


}//namespace unit_test_utils

#ifndef KOKKOS_HAVE_CUDA

void Hex8Mesh::check_discrete_laplacian(double exactLaplacian)
{
   const stk::mesh::Selector selector = meta.locally_owned_part() & !meta.globally_shared_part();
   const stk::mesh::BucketVector& nodeBuckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
   kokkos_thread_team_bucket_loop(nodeBuckets, [&](stk::mesh::Entity node)
   {
     if (bulk.num_elements(node) == 8) {
         EXPECT_NEAR(*stk::mesh::field_data(*discreteLaplacianOfPressure, node), exactLaplacian    , tol);
     }
   });
}

#endif

