#include <gtest/gtest.h>
#include <limits>

#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/HexFixture.hpp>
#include <stk_mesh/base/SkinMesh.hpp>

#include <master_element/MasterElementHO.h>

#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedElementIO.h>

#include <nalu_make_unique.h>
#include <NaluEnv.h>
#include <BucketLoop.h>

#include <memory>
#include <random>

#include <element_promotion/ElementDescription.h>
#include "UnitTestUtils.h"

namespace {

  typedef stk::mesh::Field<double> ScalarFieldType;
  typedef stk::mesh::Field<int> ScalarIntFieldType;
  typedef stk::mesh::Field<double,stk::mesh::Cartesian> VectorFieldType;

  size_t count_nodes(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& selector)
  {
    size_t nodeCount = 0u;
    for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, selector)) {
      nodeCount += ib->size();
    }
    return nodeCount;
  }

  double linear(double a, const double* b, const double* coords) {
     return (a + b[0] * coords[0] + b[1] * coords[1] + b[2] * coords[2]);
  }


  std::string vector_field_data_to_string(const VectorFieldType& vecField, stk::mesh::Entity node)
  {
    const double* vec = stk::mesh::field_data(vecField,node);

    std::stringstream msg;
    msg << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";

    return msg.str();
  }


}//namespace


class PromoteElementHexTest : public ::testing::Test
{
protected:
  PromoteElementHexTest()
: comm(MPI_COMM_WORLD),
  nDim(3)
{};

  void init(int nx, int ny, int nz,  int in_polyOrder)
  {
    auto aura = stk::mesh::BulkData::NO_AUTO_AURA;
    fixture = sierra::nalu::make_unique<stk::mesh::fixtures::HexFixture>(comm, nx, ny, nz, aura);
    meta = &fixture->m_meta;
    bulk = &fixture->m_bulk_data;
    surfSupPart = nullptr;
    surfSubPart = nullptr;
    topo = stk::topology::HEX_8;
    hexPart = fixture->m_elem_parts[0];
    ThrowRequire(hexPart != nullptr);
    dnvField = &meta->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
    qField = &meta->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar");
    dqdxField = &meta->declare_field<VectorFieldType>(stk::topology::NODE_RANK, "dqdx");
    coordField = &meta->declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coords");
    intField = &meta->declare_field<ScalarIntFieldType>(stk::topology::NODE_RANK, "integer field");

    poly_order = in_polyOrder;

    surfSupPart = &meta->declare_part("surface_1", stk::topology::FACE_RANK);
    surfSubPart = &meta->declare_part_with_topology("surface_1_hex8_quad4", stk::topology::QUAD_4);

    meta->declare_part_subset(*surfSupPart, *surfSubPart);
    edgePart = &meta->declare_part("edge_part", stk::topology::EDGE_RANK);
    facePart = &meta->declare_part("face_part", stk::topology::FACE_RANK);
    baseParts = {hexPart, surfSupPart};

    setup_promotion();

    const double zeroDouble = 0.0;
    stk::mesh::put_field_on_mesh(*dnvField, meta->universal_part(), 1, &zeroDouble);
    stk::mesh::put_field_on_mesh(*qField, meta->universal_part(), 1, &zeroDouble);
    stk::mesh::put_field_on_mesh(*dqdxField, meta->universal_part(), nDim, &zeroDouble);
    stk::mesh::put_field_on_mesh(*coordField, meta->universal_part(), nDim, &zeroDouble);

    int zeroInt = 0;
    stk::mesh::put_field_on_mesh(*intField, meta->universal_part(), 1, &zeroInt);
    fixture->m_meta.commit();
    fixture->generate_mesh(stk::mesh::fixtures::FixedCartesianCoordinateMapping(nx, ny, nz, nx, ny, nz));
    stk::mesh::PartVector surfParts = {surfSubPart};
    stk::mesh::skin_mesh(*bulk, surfParts);

    const auto& meshCoordField = *static_cast<const VectorFieldType*>(meta->coordinate_field());

    double Q[9] = {1, 2, -4, 0, 1, 1, .5, 0, 1 };

    sierra::nalu::bucket_loop(bulk->get_buckets(stk::topology::NODE_RANK, meta->universal_part()), [&](stk::mesh::Entity node) {
      double* coords = stk::mesh::field_data(*coordField, node);
      const double* model_coords = stk::mesh::field_data(meshCoordField, node);
      for (unsigned d = 0; d < meta->spatial_dimension(); ++d) {
        coords[0] = Q[0]*model_coords[0]+Q[1]*model_coords[1]+Q[2]*model_coords[2];
        coords[1] = Q[3]*model_coords[0]+Q[4]*model_coords[1]+Q[5]*model_coords[2];
        coords[2] = Q[6]*model_coords[0]+Q[7]*model_coords[1]+Q[8]*model_coords[2];
      }
    });
  }

  void setup_promotion() {
    elemDesc = sierra::nalu::ElementDescription::create(nDim, poly_order);

    // declare super parts mirroring the orginal parts
    const auto superName = sierra::nalu::super_element_part_name(hexPart->name());
    topo = stk::create_superelement_topology(static_cast<unsigned>(elemDesc->nodesPerElement));
    const stk::mesh::Part* superPart = &meta->declare_part_with_topology(superName, topo);
    superParts.push_back(superPart);

    stk::mesh::Part* superSuperPart =
        &meta->declare_part(sierra::nalu::super_element_part_name(surfSupPart->name()), stk::topology::FACE_RANK);

    const auto sidePartName = sierra::nalu::super_subset_part_name(surfSubPart->name());
    auto sideTopo = stk::create_superface_topology(static_cast<unsigned>(elemDesc->nodesPerSide));
    stk::mesh::Part* superSidePart = &meta->declare_part_with_topology(sidePartName, sideTopo);
    meta->declare_part_subset(*superSuperPart, *superSidePart);
    superParts.push_back(superSuperPart);
  }

  void promote_mesh() {
    sierra::nalu::promotion::promote_elements(*bulk, *elemDesc, *coordField, baseParts, edgePart, facePart);
  }

  void output_mesh()
  {
    const stk::mesh::PartVector& outParts = {hexPart};
    std::string fileName = "hv2.e" ;

    io = sierra::nalu::make_unique<sierra::nalu::PromotedElementIO>(
      *elemDesc, *meta, *bulk, outParts, fileName, *coordField
    );

    io->add_fields({dnvField,qField,dqdxField});
    io->write_database_data(0.0);
  }

  void dump_coords()
  {
    sierra::nalu::bucket_loop(
      bulk->get_buckets(stk::topology::NODE_RANK, meta->universal_part()), [&](stk::mesh::Entity node) {
      std::cout << "Node id: " << bulk->identifier(node) << ": "
                << vector_field_data_to_string(*coordField, node) << std::endl;
    });
  }

  size_t expected_node_count(size_t originalNodeCount)
  {
    size_t expectedNodeCount = std::pow(poly_order*(static_cast<int>(std::cbrt(originalNodeCount+1))-1)+1,3);
    return expectedNodeCount;
  }

  double initialize_linear_scalar_field()
   {
     // q = a + b^T x
     std::mt19937 rng;
     rng.seed(0); // fixed seed
     std::uniform_real_distribution<double> coeff(-1.0, 1.0);

     double a = coeff(rng);

     double b[3];
     for (unsigned j = 0; j < 3; ++j) {
       b[j] = coeff(rng);
     }

     const stk::mesh::Selector selector = meta->locally_owned_part() | meta->globally_shared_part();
     const auto& buckets = bulk->get_buckets(stk::topology::NODE_RANK, selector);
     sierra::nalu::bucket_loop(buckets, [&](stk::mesh::Entity node)  {
       *stk::mesh::field_data(*qField, node) = ::linear(a, b, stk::mesh::field_data(*coordField, node));
     });

     return std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
   }

   void compute_dual_nodal_volume()
   {
     auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);
     auto quad = sierra::nalu::TensorProductQuadratureRule("GaussLegendre", poly_order);

     sierra::nalu::HigherOrderHexSCV meSCV(*elemDesc, basis, quad);

     // extract master element specifics
     const int nodesPerElement = meSCV.nodesPerElement_;
     const int numScvIp = meSCV.num_integration_points();
     const int* ipNodeMap = meSCV.ipNodeMap();

     // define scratch fields
     std::vector<double> ws_scv_volume(numScvIp);
     std::vector<double> ws_coordinates(nodesPerElement * nDim);

     const stk::mesh::Selector superSelector = stk::mesh::selectUnion(superParts);
     const auto& elem_buckets = bulk->get_buckets(stk::topology::ELEM_RANK, superSelector);

     sierra::nalu::bucket_loop(elem_buckets, [&](stk::mesh::Entity elem) {
       stk::mesh::Entity const* node_rels = bulk->begin_nodes(elem);
       for (int ni = 0; ni < nodesPerElement; ++ni) {
         const double* const coords = stk::mesh::field_data(*coordField, node_rels[ni]);
         const int offSet = ni * nDim;
         for (unsigned j = 0; j < nDim; ++j) {
           ws_coordinates[offSet + j] = coords[j];
         }
       }

       // compute integration point volume
       double scv_error = 1.0;
       meSCV.determinant(1, ws_coordinates.data(), ws_scv_volume.data(), &scv_error);

       // assemble dual volume while scattering ip volume
       for (int ip = 0; ip < numScvIp; ++ip) {
         *stk::mesh::field_data(*dnvField, node_rels[ipNodeMap[ip]]) += ws_scv_volume[ip];
       }
     });
   }

   void compute_projected_nodal_gradient_interior()
   {
     auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);
     auto quad = sierra::nalu::TensorProductQuadratureRule("GaussLegendre", poly_order);

     sierra::nalu::HigherOrderHexSCS meSCS(*elemDesc, basis, quad);

     auto numScsIp = meSCS.num_integration_points();
     auto nodesPerElement = meSCS.nodesPerElement_;

     int dimension = nDim;
     std::vector<double> ws_scalar(nodesPerElement);
     std::vector<double> ws_dualVolume(nodesPerElement);
     std::vector<double> ws_coords(dimension*nodesPerElement);
     std::vector<double> ws_areav(dimension*numScsIp);
     std::vector<double> ws_dqdx(dimension*numScsIp);
     const auto* lrscv = meSCS.adjacentNodes();
     std::vector<double> ws_shape_function(nodesPerElement*numScsIp);
     meSCS.shape_fcn(ws_shape_function.data());

     const auto& buckets = bulk->get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(superParts));
     sierra::nalu::bucket_loop(buckets, [&](stk::mesh::Entity elem) {
       const auto* node_rels = bulk->begin_nodes(elem);

       for (int ni = 0; ni < nodesPerElement; ++ni) {
         stk::mesh::Entity node = node_rels[ni];

         const double * coords = stk::mesh::field_data(*coordField, node);

         // gather scalars
         ws_scalar[ni]     = *stk::mesh::field_data(*qField, node);
         ws_dualVolume[ni] = *stk::mesh::field_data(*dnvField, node);

         // gather vectors
         const int offSet = ni*dimension;
         for ( int j=0; j < dimension; ++j ) {
           ws_coords[offSet+j] = coords[j];
         }
       }

       double scs_error = 0.0;
       meSCS.determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

       for (int ip = 0; ip < numScsIp; ++ip) {
         const int il = lrscv[2*ip];
         const int ir = lrscv[2*ip+1];

         double* gradQL = stk::mesh::field_data(*dqdxField, node_rels[il]);
         double* gradQR = stk::mesh::field_data(*dqdxField, node_rels[ir]);

         double qIp = 0.0;
         const int offSet = ip*nodesPerElement;
         for (int ic = 0; ic < nodesPerElement; ++ic) {
           qIp += ws_shape_function[offSet+ic]*ws_scalar[ic];
         }

         double inv_volL = 1.0/ws_dualVolume[il];
         double inv_volR = 1.0/ws_dualVolume[ir];

         for ( int j = 0; j < dimension; ++j ) {
           double fac = qIp*ws_areav[ip*dimension+j];
           gradQL[j] += fac*inv_volL;
           gradQR[j] -= fac*inv_volR;
         }
       }
     });
   }

   void compute_projected_nodal_gradient_boundary()
   {
     auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMapBC, elemDesc->nodeLocs1D);
     auto quad = sierra::nalu::TensorProductQuadratureRule("GaussLegendre", poly_order);

     sierra::nalu::HigherOrderQuad3DSCS meBC(*elemDesc, basis, quad);

     auto numScsIp = meBC.num_integration_points();
     auto nodesPerFace = meBC.nodesPerElement_;

     std::vector<double> ws_scalar(nodesPerFace);
     std::vector<double> ws_dualVolume(nodesPerFace);
     std::vector<double> ws_coords(nDim*nodesPerFace);
     std::vector<double> ws_areav(nDim*numScsIp);
     std::vector<double> ws_dqdx(nDim*numScsIp);
     const auto* ipNodeMap = meBC.ipNodeMap();

     std::vector<double> ws_shape_function(nodesPerFace*numScsIp);
     meBC.shape_fcn(ws_shape_function.data());

     const auto& buckets = bulk->get_buckets(stk::topology::FACE_RANK, stk::mesh::selectUnion(superParts));
     sierra::nalu::bucket_loop(buckets, [&](stk::mesh::Entity face) {
       const auto* face_node_rels = bulk->begin_nodes(face);
       for (int ni = 0; ni < nodesPerFace; ++ni) {
         stk::mesh::Entity node = face_node_rels[ni];

         const double * coords = stk::mesh::field_data(*coordField, node);

         // gather scalars
         ws_scalar[ni]     = *stk::mesh::field_data(*qField, node);
         ws_dualVolume[ni] = *stk::mesh::field_data(*dnvField, node);

         // gather vectors
         const int offSet = ni*nDim;
         for ( unsigned j=0; j < nDim; ++j ) {
           ws_coords[offSet+j] = coords[j];
         }
       }

       double scs_error = 0.0;
       meBC.determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

       for (int ip = 0; ip < numScsIp; ++ip) {
         const int nn = ipNodeMap[ip];

         stk::mesh::Entity nodeNN = face_node_rels[nn];

         // pointer to fields to assemble
         double *gradQNN = stk::mesh::field_data(*dqdxField, nodeNN);
         double volNN = *stk::mesh::field_data(*dnvField, nodeNN);

         // interpolate to scs point; operate on saved off ws_field
         double qIp = 0.0;
         const int offSet = ip*nodesPerFace;
         for ( int ic = 0; ic < nodesPerFace; ++ic ) {
           qIp += ws_shape_function[offSet+ic]*ws_scalar[ic];
         }

         // nearest node volume
         double inv_volNN = 1.0/volNN;

         // assemble to nearest node
         for ( unsigned j = 0; j < nDim; ++j ) {
           double fac = qIp*ws_areav[ip*nDim+j];
           gradQNN[j] += fac*inv_volNN;
         }
       }
     });
   }

  stk::ParallelMachine comm;
  unsigned nDim;
  std::unique_ptr<stk::mesh::fixtures::HexFixture> fixture;
  stk::mesh::MetaData* meta;
  stk::mesh::BulkData* bulk;
  unsigned poly_order;
  stk::mesh::Part* hexPart;
  stk::mesh::Part* surfSupPart;
  stk::mesh::Part* surfSubPart;
  stk::topology topo;
  std::unique_ptr<sierra::nalu::ElementDescription> elemDesc;
  stk::mesh::PartVector baseParts;
  stk::mesh::ConstPartVector superParts;
  stk::mesh::Part* edgePart;
  stk::mesh::Part* facePart;
  std::unique_ptr<sierra::nalu::PromotedElementIO> io;
  ScalarFieldType* dnvField;
  ScalarFieldType* qField;
  ScalarIntFieldType* intField;
  VectorFieldType* dqdxField;
  VectorFieldType* coordField;

};

TEST_F(PromoteElementHexTest, node_count)
{
  int polyOrder = 7;

  int nprocs = stk::parallel_machine_size(MPI_COMM_WORLD);
  int nprocx = std::cbrt(nprocs+0.5);
  if (nprocx*nprocx*nprocx != nprocs) {
    return;
  }

  init(nprocx, nprocx, nprocx, polyOrder);
  size_t originalNodeCount = ::count_nodes(*bulk, meta->universal_part());

  stk::mesh::PartVector supElemParts;
  stk::mesh::PartVector supSideParts;

  promote_mesh();
  EXPECT_EQ(expected_node_count(originalNodeCount), ::count_nodes(*bulk, meta->universal_part()));

  bool outputMesh = false;
  if (outputMesh) {
    compute_dual_nodal_volume();
    if (bulk->parallel_size() > 1) {
      stk::mesh::parallel_sum(*bulk, {dnvField});
    }
    EXPECT_NO_THROW(output_mesh());
  }
}


TEST_F(PromoteElementHexTest, png)
{
    if (stk::parallel_machine_size(MPI_COMM_WORLD) > 48) {
      return;
    }

    double tol = 1.0e-8;
    int polyOrder = 4;

    init(3, 2, 8, polyOrder);

    promote_mesh();

    double exactGradMag = initialize_linear_scalar_field();

    stk::mesh::field_fill(0.0, *dnvField);
    compute_dual_nodal_volume();
    if (bulk->parallel_size() > 1) {
      stk::mesh::parallel_sum(*bulk, {dnvField});
    }

    stk::mesh::field_fill(0.0, *dqdxField);
    compute_projected_nodal_gradient_interior();
    compute_projected_nodal_gradient_boundary();
    if (bulk->parallel_size() > 1) {
      stk::mesh::parallel_sum(*bulk, {dqdxField});
    }

    const auto& buckets = bulk->get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(superParts));
    sierra::nalu::bucket_loop(buckets, [&](stk::mesh::Entity node) {
      const double* dqdx = stk::mesh::field_data(*dqdxField, node);
      double dqdxMag = std::sqrt(dqdx[0] * dqdx[0] + dqdx[1] * dqdx[1]+dqdx[2]*dqdx[2]);
      EXPECT_NEAR(dqdxMag, exactGradMag, tol);
    });

    bool outputMesh = false;
    if (outputMesh) {
      EXPECT_NO_THROW(output_mesh());
    }
}

TEST_F(PromoteElementHexTest, node_sharing)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 2) {
    return;
  }

  int polyOrder = 2;
  init(2 ,1, 1, polyOrder);

  promote_mesh();
  ThrowRequire(!bulk->in_modifiable_state());

  stk::mesh::EntityIdVector sharedNodeIds = {2, 5, 8, 11, 21, 22, 23, 24, 33};

  for ( auto id  : sharedNodeIds) {
    auto newSharedNode = bulk->get_entity(stk::topology::NODE_RANK, id);
    *stk::mesh::field_data(*intField, newSharedNode) = bulk->parallel_rank() +1;
    if (bulk->parallel_size() > 1) {
      stk::mesh::parallel_sum(*bulk, {intField});
    }
    EXPECT_EQ(*stk::mesh::field_data(*intField, newSharedNode), 3);
  }
}
