#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <master_element/MasterElement.h>

#include <master_element/Hex8CVFEM.h>
#include <master_element/Hex27CVFEM.h>

// NGP-based includes
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include "UnitTestUtils.h"

namespace {

  // evaluate a polynomial, left-to-right
double poly_val(const std::vector<double>& coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j] * std::pow(x,j);
  }
  return val;
}

double poly_der(const std::vector<double>& coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += j*coeffs[j] * std::pow(x,j-1);
  }
  return val;
}

double poly_val(
  const std::vector<std::vector<double>>& coeffs, const double* x)
{
  if( coeffs.size() == 2) {
    return (poly_val(coeffs[0],x[0]) * poly_val(coeffs[1],x[1]));
  }
  return (poly_val(coeffs[0],x[0]) * poly_val(coeffs[1],x[1]) * poly_val(coeffs[2],x[2]));
}

double poly_der(
  const std::vector<std::vector<double>>& coeffs, const double* x, int dir)
{
  double val = 0.0;


  if( coeffs.size() == 2) {
    switch(dir) {
      case 0:
        val = poly_der(coeffs[0], x[0]) * poly_val(coeffs[1], x[1]);
        break;
      case 1:
        val = poly_val(coeffs[0], x[0]) * poly_der(coeffs[1], x[1]);
        break;
      default:
        throw std::runtime_error("invalid direction");
    }
  }

  if( coeffs.size() == 3) {
    switch(dir) {
      case 0:
          val = poly_der(coeffs[0], x[0]) * poly_val(coeffs[1], x[1]) * poly_val(coeffs[2], x[2]);
          break;
        case 1:
          val = poly_val(coeffs[0], x[0]) * poly_der(coeffs[1], x[1]) * poly_val(coeffs[2], x[2]);
          break;
        case 2:
          val = poly_val(coeffs[0], x[0]) * poly_val(coeffs[1], x[1]) * poly_der(coeffs[2], x[2]);
          break;
      default:
        throw std::runtime_error("invalid direction");
    }
  }

  return val;
}

void check_interpolation(
  const stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  sierra::nalu::MasterElement& me,
  unsigned poly_order,
  bool  /* usingNGP */ = false)
{
  // Check that we can interpolate a random 3D polynomial
  // to the integration points

  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);
  EXPECT_EQ(elems.size(), 1u); // single element test
  auto elem = elems.front();

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  std::vector<double> ws_field(topo.num_nodes());
  const auto* const coordField = bulk.mesh_meta_data().coordinate_field();
  EXPECT_TRUE(coordField != nullptr);

  unsigned dim = topo.dimension();

  // get random polynomial
  std::vector<std::vector<double>> coeffs(dim);
  for (unsigned j = 0; j < dim; ++j) {
    coeffs[j].resize(poly_order+1);
    for (unsigned i = 0; i < coeffs[j].size(); ++i) {
      coeffs[j][i] = coeff(rng);
    }
  }

  std::vector<double> polyResult(me.num_integration_points());
  for (int j = 0; j < me.num_integration_points(); ++j) {
    polyResult[j] = poly_val(coeffs, &me.integration_locations()[j*dim]);
  }

  const auto* nodes = bulk.begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coords = static_cast<const double*>(stk::mesh::field_data(*coordField, nodes[j]));
    ws_field[j] = poly_val(coeffs, coords);
  }

  std::vector<double> meResult(me.num_integration_points(), 0.0);
  std::vector<double> meShapeFunctions(me.num_integration_points() * topo.num_nodes());
  me.shape_fcn(meShapeFunctions.data());
  
  for (int j = 0; j < me.num_integration_points(); ++j) {
    for (unsigned i = 0; i < topo.num_nodes(); ++i) {
      meResult[j] += meShapeFunctions[j*topo.num_nodes()+i] * ws_field[i];
    }
  }
  
  for (unsigned j = 0 ; j < meResult.size(); ++j) {
    EXPECT_NEAR(meResult[j], polyResult[j], 1.0e-12);
  }
}

void check_derivatives(
  const stk::mesh::BulkData& bulk,
  const stk::topology& topo,
  sierra::nalu::MasterElement& me,
  unsigned poly_order)
{
  // Check that we can interpolate a random 3D polynomial
  // to the integration points

  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);
  EXPECT_EQ(elems.size(), 1u); // single element test
  auto elem = elems.front();

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  std::vector<double> ws_field(topo.num_nodes());
  const auto* const coordField = bulk.mesh_meta_data().coordinate_field();
  EXPECT_TRUE(coordField != nullptr);

  unsigned dim = topo.dimension();

  // get random polynomial
  std::vector<std::vector<double>> coeffs(dim);
  for (unsigned j = 0; j < dim; ++j) {
    coeffs[j].resize(poly_order+1);
    for (unsigned i = 0; i < coeffs[j].size(); ++i) {
      coeffs[j][i] = coeff(rng);
    }
  }

  std::vector<double> polyResult(me.num_integration_points() * dim);
  for (int j = 0; j < me.num_integration_points(); ++j) {
    for (unsigned d = 0; d < dim; ++d) {
      polyResult[j*dim+d] = poly_der(coeffs, &me.integration_locations()[j*dim], d);
    }
  }

  std::vector<double> ws_coords(topo.num_nodes() * dim);
  const auto* nodes = bulk.begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coords = static_cast<const double*>(stk::mesh::field_data(*coordField, nodes[j]));
    for (unsigned d = 0; d < dim; ++d) {
      ws_coords[j*dim+d] = coords[d];
    }
    ws_field[j] = poly_val(coeffs, coords);
  }

  std::vector<double> meResult(me.num_integration_points() * dim, 0.0);
  std::vector<double> meGrad(me.num_integration_points() * topo.num_nodes() * dim);
  std::vector<double> meDeriv(me.num_integration_points() * topo.num_nodes() * dim);
  std::vector<double> meDetj(me.num_integration_points());

  double error = 0.0;
  me.grad_op(1,ws_coords.data(), meGrad.data(), meDeriv.data(), meDetj.data(), &error);
  EXPECT_EQ(error, 0.0);

  for (int j = 0; j < me.num_integration_points(); ++j) {
    for (unsigned i = 0; i < topo.num_nodes(); ++i) {
      for (unsigned d = 0; d < dim; ++d) {
        meResult[j*dim+d] += meGrad[j*topo.num_nodes()*dim + i * dim + d] * ws_field[i];
      }
    }
  }

  // shape function derivatives and grad_op should be the same
  for (unsigned j = 0 ; j < meGrad.size(); ++j) {
    EXPECT_NEAR(meGrad[j], meDeriv[j], 1.0e-12);
  }

  // detj should be unity
  for (unsigned j = 0 ; j < meDetj.size(); ++j) {
    EXPECT_NEAR(1, meDetj[j], 1.0e-12);
  }

  for (unsigned j = 0 ; j < meResult.size(); ++j) {
    EXPECT_NEAR(meResult[j], polyResult[j], 1.0e-12);
  }
}

class MasterElementHexSerial : public ::testing::Test
{
protected:
    MasterElementHexSerial()
    : comm(MPI_COMM_WORLD), spatialDimension(3),
      meta(spatialDimension), bulk(meta, comm),
      poly_order(1), topo(stk::topology::HEX_8)
    {
    }

    void setup_poly_order_1_hex_8() {
      poly_order = 1;
      topo = stk::topology::HEX_8;
      unit_test_utils::create_one_reference_element(bulk, stk::topology::HEX_8);
    }

    void setup_poly_order_2_hex_27() {
      poly_order = 2;
      topo = stk::topology::HEX_27;
      unit_test_utils::create_one_reference_element(bulk, stk::topology::HEX_27);
    }

    stk::ParallelMachine comm;
    unsigned spatialDimension;
    stk::mesh::MetaData meta;
    stk::mesh::BulkData bulk;
    unsigned poly_order;
    stk::topology topo;
};


TEST_F(MasterElementHexSerial, hex8_scs_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_1_hex_8();
    sierra::nalu::HexSCS hexscs;
    check_interpolation(bulk, topo, hexscs, poly_order);
  }
}

TEST_F(MasterElementHexSerial, hex8_scv_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_1_hex_8();
    sierra::nalu::HexSCV hexscv;
    check_interpolation(bulk, topo, hexscv, poly_order);
  }
}

TEST_F(MasterElementHexSerial, hex8_scs_derivatives)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_1_hex_8();
    sierra::nalu::HexSCS hexscs;
    check_derivatives(bulk, topo, hexscs, poly_order);
  }
}

TEST_F(MasterElementHexSerial, hex27_scs_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_2_hex_27();
    sierra::nalu::Hex27SCS hex27scs;
    check_interpolation(bulk, topo, hex27scs, poly_order);
  }
}

TEST_F(MasterElementHexSerial, hex27_scv_interpolation)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_2_hex_27();
    sierra::nalu::Hex27SCV hex27scv;
    check_interpolation(bulk, topo, hex27scv, poly_order);
  }
}

TEST_F(MasterElementHexSerial, hex27_scs_derivatives)
{
  if (stk::parallel_machine_size(comm) == 1) {
    setup_poly_order_2_hex_27();
    sierra::nalu::Hex27SCS hex27scs;
    check_derivatives(bulk, topo, hex27scs, poly_order);
  }
}

}//namespace

