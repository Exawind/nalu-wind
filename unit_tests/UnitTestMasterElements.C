#include <gtest/gtest.h>
#include <limits>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <master_element/Quad42DCVFEM.h>
#include <master_element/TensorOps.h>

#include <memory>
#include <random>

#include "UnitTestUtils.h"

namespace {

using VectorFieldType = stk::mesh::Field<double, stk::mesh::Cartesian>;
//-------------------------------------------------------------------------
double linear_scalar_value(int dim, double a, const double* b, const double* x)
{
  if (dim == 2u) {
    return (a + b[0] * x[0] + b[1] * x[1]);
  }
  return (a + b[0] * x[0] + b[1] * x[1] + b[2] * x[2]);
}
//-------------------------------------------------------------------------
struct LinearField
{
  LinearField(int in_dim, double in_a, const double* in_b) : dim(in_dim), a(in_a) {
    b[0] = in_b[0];
    b[1] = in_b[1];
    if (dim == 3) b[2] = in_b[2];
  }

  double operator()(const double* x) { return linear_scalar_value(dim, a, b, x); }

  const int dim;
  const double a;
  double b[3];
};

LinearField make_random_linear_field(int dim, std::mt19937& rng)
{

  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffs(dim);

  double a = coeff(rng);
  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }
  return LinearField(dim, a, coeffs.data());
}


//-------------------------------------------------------------------------
void check_interpolation_at_ips(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& me)
{
  // Check that we can interpolate a random 3D polynomial
  // to the integration points
  int dim = me.nDim_;

  std::mt19937 rng;
  rng.seed(0);
  auto linField = make_random_linear_field(dim,rng);

  const auto& intgLoc = me.integration_locations();
  std::vector<double> polyResult(me.numIntPoints_);
  for (int j = 0; j < me.numIntPoints_; ++j) {
    polyResult[j] = linField(&intgLoc[j*dim]);
  }

  std::vector<double> ws_field(me.nodesPerElement_);
  for (int j = 0; j < me.nodesPerElement_; ++j) {
    ws_field[j] = linField(stk::mesh::field_data(coordField, node_rels[j]));
  }

  std::vector<double> meResult(me.numIntPoints_, 0.0);

  std::vector<double> meShapeFunctions(me.nodesPerElement_ * me.numIntPoints_);
  me.shape_fcn(meShapeFunctions.data());

  for (int j = 0; j < me.numIntPoints_; ++j) {
    for (int i = 0; i < me.nodesPerElement_; ++i) {
      meResult[j] += meShapeFunctions[j*me.nodesPerElement_+i] * ws_field[i];
    }
  }

  for (unsigned j = 0 ; j < meResult.size(); ++j) {
    EXPECT_NEAR(meResult[j], polyResult[j], tol);
  }
}
//-------------------------------------------------------------------------
void check_derivatives_at_ips(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& me)
{
  // Check that we can interpolate a random 3D linear field
  int dim = me.nDim_;

  std::mt19937 rng;
  rng.seed(0);
  auto linField = make_random_linear_field(dim,rng);

  std::vector<double> polyResult(me.numIntPoints_ * dim);
  for (int j = 0; j < me.numIntPoints_; ++j) {
    for (int d = 0; d < dim; ++d) {
      polyResult[j*dim+d] = linField.b[d];
    }
  }

  std::vector<double> ws_field(me.nodesPerElement_);
  std::vector<double> ws_coords(me.nodesPerElement_ * dim);
  for (int j = 0; j < me.nodesPerElement_; ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords[j*dim+d] = coords[d];
    }
    ws_field[j] = linField(coords);
  }

  std::vector<double> meResult(me.numIntPoints_ * dim, 0.0);
  std::vector<double> meGrad(me.numIntPoints_ * me.nodesPerElement_ * dim);
  std::vector<double> meDeriv(me.numIntPoints_ * me.nodesPerElement_ * dim);
  std::vector<double> meDetj(me.numIntPoints_);

  double error = 0.0;
  me.grad_op(1, ws_coords.data(), meGrad.data(), meDeriv.data(), meDetj.data(), &error);
  EXPECT_EQ(error, 0.0);

  for (int j = 0; j < me.numIntPoints_; ++j) {
    for (int i = 0; i < me.nodesPerElement_; ++i) {
      for (int d = 0; d < dim; ++d) {
        meResult[j*dim+d] += meGrad[j*me.nodesPerElement_*dim + i * dim + d] * ws_field[i];
      }
    }
  }

  // detj should be unity to floating point error
  for (unsigned j = 0 ; j < meDetj.size(); ++j) {
    EXPECT_NEAR(1, meDetj[j], tol) ;
  }

  // derivative should be exact to floating point error
  for (unsigned j = 0 ; j < meResult.size(); ++j) {
    EXPECT_NEAR(meResult[j], polyResult[j], tol);
  }
}
//-------------------------------------------------------------------------
void check_scv_shifted_ips_are_nodal(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& meSV)
{
  // check that the subcontrol volume ips are at the nodes for the shifted ips

  int dim = meSV.nDim_;
  std::vector<double> ws_coords(meSV.nodesPerElement_ * dim);

  for (int j = 0; j < meSV.nodesPerElement_; ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords[j*dim+d] = coords[d];
    }
  }

  const int nint = meSV.numIntPoints_*meSV.nDim_;
  const double* shiftedIps = meSV.integration_location_shift();
  EXPECT_EQ(ws_coords.size(), nint) << "P1 test";
  for (int j = 0; j < nint; ++j) {
    EXPECT_NEAR(ws_coords[j], shiftedIps[j], tol);
  }
}
//-------------------------------------------------------------------------
void check_volume_integration(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& meSV)
{
  int dim = meSV.nDim_;
  std::vector<double> ws_coords_mapped(meSV.nodesPerElement_ * dim, 0.0);
  std::vector<double> ws_coords(meSV.nodesPerElement_ * dim, 0.0);

  std::mt19937 rng;
  rng.seed(0);

  auto QR = unit_test_utils::random_linear_transformation(dim, 1.0, rng);
  for (int j = 0; j < meSV.nodesPerElement_; ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);

    if (dim == 3) {
      sierra::nalu::matvec33(QR.data(), coords, &ws_coords_mapped[j*dim]);
    }
    else {
      sierra::nalu::matvec22(QR.data(), coords, &ws_coords_mapped[j*dim]);
    }

    for (int k = 0; k < dim; ++k) {
      ws_coords[j*dim+k] = coords[k];
    }

  }
  const double detQR = (dim == 3) ? sierra::nalu::determinant33(QR.data()) : sierra::nalu::determinant22(QR.data());
  ASSERT_TRUE(detQR > 1.0e-15);

  double error = 0;
  std::vector<double> volume_integration_weights(meSV.numIntPoints_);
  meSV.determinant(1, ws_coords.data(), volume_integration_weights.data(), &error);
  ASSERT_DOUBLE_EQ(error, 0);

  std::vector<double> skewed_volume_integration_weights(meSV.numIntPoints_);
  meSV.determinant(1, ws_coords_mapped.data(), skewed_volume_integration_weights.data(), &error);
  ASSERT_DOUBLE_EQ(error, 0);

  for (int k = 0; k < meSV.numIntPoints_; ++k) {
    EXPECT_NEAR(detQR*volume_integration_weights[k], skewed_volume_integration_weights[k], tol);
  }

}
//-------------------------------------------------------------------------
#if 0
void check_exposed_face_shifted_ips_are_nodal(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& meSS)
{
  // check that the subcontrol volume ips are at the nodes for the shifted ips

  const int dim = meSS.nDim_;
  std::vector<std::vector<double>> coordList(meSS.nodesPerElement_);
  for (int j = 0; j < meSS.nodesPerElement_; ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    coordList.at(j).resize(dim);
    for (int d = 0; d < dim; ++d) {
      coordList.at(j).at(d) = coords[d];
     }
  }

  const double* shiftedIps = meSS.integration_exp_face_shift();

  int index = 0;
  const int nint = shiftedIps.size()/dim;
  std::vector<std::vector<double>> shiftedIpList(nint);
  for (int j = 0; j < nint; ++j) {
    shiftedIpList.at(j).resize(dim);
    for (int d = 0; d < dim; ++d) {
      shiftedIpList.at(j).at(d) = shiftedIps[index];
      ++index;
    }
  }

  auto is_same_vector = [] (const std::vector<double>& u,  const std::vector<double>& v, double tol) {
    if (u.size() != v.size()) {
      return false;
    }

    for (unsigned j = 0; j < u.size(); ++j) {
      if (std::abs(u[j] - v[j]) > tol) {
        return false;
      }
    }
    return true;
  };

  std::vector<int> countSame(shiftedIpList.size(),0);
  for (unsigned i = 0; i < shiftedIpList.size(); ++i) {
    for (unsigned j = 0; j < coordList.size(); ++j) {
      if (is_same_vector(coordList.at(j), shiftedIpList.at(i), tol)) {
        ++countSame.at(i);
      }
    }
  }

  for (unsigned j = 0; j <countSame.size(); ++j) {
    if (countSame.at(j) != 1 && dim == 3) {
      std::cout << "iploc: " << shiftedIpList.at(j)[0] << ", "
                             << shiftedIpList.at(j)[1] << ", "
                             << shiftedIpList.at(j)[2] << std::endl;
    }
    EXPECT_EQ(countSame.at(j), 1);
  }
}
#endif
//-------------------------------------------------------------------------
void check_is_in_element(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& me)
{
  // Check that the isoparametric coordinates are the same as the physical point
  // for the reference element

  int dim = me.nDim_;

  std::mt19937 rng;
  rng.seed(0);

  // randomly select a point within (boxmin, boxmax)^3 \subset reference element domain
  const double boxmin = 0.125;
  const double boxmax = 0.25;
  std::uniform_real_distribution<double> coeff(boxmin, boxmax);
  std::vector<double> random_pt(dim);
  for (int j = 0; j < dim; ++j) {
    random_pt[j] = coeff(rng);
  }

  // is in element uses a different stride for the coordinate data
  // compared to the gradient computation
  std::vector<double> ws_field(me.nodesPerElement_);
  std::vector<double> ws_coords(me.nodesPerElement_ * dim);

  // Hex8/Quad4's is_in_element and interpolatePoint routines use a different,
  // but self-consistent reference element compared to the core shape functions
  // and derivatives

  bool isHexSCS = dynamic_cast<sierra::nalu::HexSCS*>(&me) != nullptr;
  bool isQuadSCS = dynamic_cast<sierra::nalu::Quad42DSCS*>(&me) != nullptr;
  double fac = (isHexSCS || isQuadSCS) ? 2.0 : 1.0;

  for (int j = 0; j < me.nodesPerElement_; ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords[d * me.nodesPerElement_ + j] = fac*coords[d];
    }
  }

  std::vector<double> mePt(dim);
  auto dist = me.isInElement(ws_coords.data(), random_pt.data(), mePt.data());

  EXPECT_LT(dist, 1.0+tol);
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(random_pt[d], mePt[d], tol);
  }
}
//-------------------------------------------------------------------------
void check_is_not_in_element(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& me)
{
  // check that we correctly report that a point outside of an element is
  // outside of the element

  int dim = me.nDim_;

  // choose a point not in the element
  std::vector<double> exterior_pt = { 100, 100, 100 };

  std::vector<double> ws_field(me.nodesPerElement_);
  std::vector<double> ws_coords(me.nodesPerElement_ * dim);

  for (int j = 0; j < me.nodesPerElement_; ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords[d * me.nodesPerElement_ + j] = coords[d];
    }
  }

  std::vector<double> mePt(dim);
  double dist = me.isInElement(ws_coords.data(), exterior_pt.data(), mePt.data());
  EXPECT_GT(dist, 1 + tol);
}
//-------------------------------------------------------------------------
void check_particle_interp(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& me)
{
  // Check that, for a distorted element, we can find and interpolate values to
  // a random located point inside of the element


  int dim = me.nDim_;

  std::mt19937 rng;
  rng.seed(0);
  auto linField = make_random_linear_field(dim,rng);

  // randomly select a point within (boxmin, boxmax)^3 \subset reference element domain
  const double boxmin = 0.125;
  const double boxmax = 0.25;
  std::uniform_real_distribution<double> coeff(boxmin, boxmax);
  std::vector<double> random_pt(dim);

  for (int j = 0; j < dim; ++j) {
    random_pt[j] = coeff(rng);
  }

  std::vector<double> coeffs(dim);
  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }

  // randomly perturb each of the coordinates of by a factor of delta
  // the element still needs to actually contain the box, (boxmin, boxmax)^3
  const double delta = 0.25;
  std::uniform_real_distribution<double> coord_perturb(-delta/2, delta/2);

  // is in element uses a different stride for the coordinate data
  // compared to the gradient computation
  std::vector<double> ws_field(me.nodesPerElement_);
  std::vector<double> ws_coords(me.nodesPerElement_ * dim);
  std::vector<double> perturbed_coords(dim);

  for (int j = 0; j < me.nodesPerElement_; ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      perturbed_coords[d] = coords[d] + coord_perturb(rng);
      ws_coords[d * me.nodesPerElement_ + j] = perturbed_coords[d];
    }
    ws_field[j] = linField(perturbed_coords.data());
  }

  std::vector<double> mePt(dim);
  double dist = me.isInElement(ws_coords.data(), random_pt.data(), mePt.data());
  EXPECT_LT(dist, 1.0+tol);

  double meInterp = 0.0;
  me.interpolatePoint(1, mePt.data(), ws_field.data(), &meInterp);
  double exactVal = linField(random_pt.data());
  EXPECT_NEAR(meInterp, exactVal, tol);
}

/** Check implementation of general_shape_fcn for a given MasterElement
 *
 */
void
check_general_shape_fcn(
  const stk::mesh::Entity* node_rels,
  const VectorFieldType& coordField,
  sierra::nalu::MasterElement& me)
{
  const int dim = me.nDim_;

  std::random_device rd{};
  std::mt19937 rng{rd()};

  // 1. Generate a random point within the element
  const double boxmin = 0.125;
  const double boxmax = 0.25;
  std::uniform_real_distribution<double> coeff(boxmin, boxmax);
  std::vector<double> nodal_coords(dim);
  for (int d = 0; d < dim; d++)
    nodal_coords[d] = coeff(rng);

  // 2. Extract iso-parametric coordinates for this random point w.r.t element

  // see check_is_in_element for an explanation of the factor
  bool isHexSCS = dynamic_cast<sierra::nalu::HexSCS*>(&me) != nullptr;
  bool isQuadSCS = dynamic_cast<sierra::nalu::Quad42DSCS*>(&me) != nullptr;
  double fac = (isHexSCS || isQuadSCS) ? 2.0 : 1.0;
  std::vector<double> elem_coords(me.nodesPerElement_ * dim);

  for (int j = 0; j < me.nodesPerElement_; j++) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; d++)
      elem_coords[d * me.nodesPerElement_ + j] = fac * coords[d];
  }

  std::vector<double> isopar_coords(dim);
  auto dist = me.isInElement(
    elem_coords.data(), nodal_coords.data(), isopar_coords.data());

  // Catch any issues with random coordinates before general_shape_fcn check
  EXPECT_LT(dist, 1.0 + tol);

  //
  // 3. Finally, check general shape fcn
  //

  auto linField = make_random_linear_field(dim, rng);
  double polyResult = linField(nodal_coords.data());

  // The linear field at the nodes of the reference element
  std::vector<double> elem_field(me.nodesPerElement_);
  std::vector<double> ws_coord(dim);
  for (int j = 0; j < me.nodesPerElement_; j++) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; d++) {
      ws_coord[d] = fac * coords[d];
    }
    elem_field[j] = linField(ws_coord.data());
  }

  std::vector<double> gen_shape_fcn(me.nodesPerElement_);
  me.general_shape_fcn(1, isopar_coords.data(), gen_shape_fcn.data());

  double meResult = 0.0;
  for (int j = 0; j < me.nodesPerElement_; j++) {
    meResult += gen_shape_fcn[j] * elem_field[j];
  }

  EXPECT_NEAR(meResult, polyResult, tol);
}
}

class MasterElement : public ::testing::Test
{
protected:
    MasterElement() : comm(MPI_COMM_WORLD) {}

    void choose_topo(stk::topology topo)
    {
      meta = std::unique_ptr<stk::mesh::MetaData>(new stk::mesh::MetaData(topo.dimension()));
      bulk = std::unique_ptr<stk::mesh::BulkData>(new stk::mesh::BulkData(*meta, comm));
      elem = unit_test_utils::create_one_reference_element(*bulk, topo);
      meSS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);
      meSV = sierra::nalu::MasterElementRepo::get_volume_master_element(topo);
    }

    void scs_interpolation(stk::topology topo) {
      choose_topo(topo);
      check_interpolation_at_ips(bulk->begin_nodes(elem), coordinate_field(), *meSS);
    }

    void scv_interpolation(stk::topology topo) {
      choose_topo(topo);
      check_interpolation_at_ips(bulk->begin_nodes(elem), coordinate_field(), *meSV);
    }

    void volume_integration(stk::topology topo) {
      choose_topo(topo);
      check_volume_integration(bulk->begin_nodes(elem), coordinate_field(), *meSV);
    }

    void scs_derivative(stk::topology topo) {
      choose_topo(topo);
      check_derivatives_at_ips(bulk->begin_nodes(elem), coordinate_field(), *meSS);
    }

    void is_not_in_element(stk::topology topo) {
      choose_topo(topo);
      check_is_not_in_element(bulk->begin_nodes(elem), coordinate_field(), *meSS);
    }

    void scv_shifted_ips_are_nodal(stk::topology topo) {
      choose_topo(topo);
      check_scv_shifted_ips_are_nodal(bulk->begin_nodes(elem), coordinate_field(), *meSV);
    }

//  void exposed_face_shifted_ips_are_nodal(stk::topology topo) {
//    choose_topo(topo);
//    check_exposed_face_shifted_ips_are_nodal(bulk->begin_nodes(elem), coordinate_field(), *meSS);
//  }

    void is_in_element(stk::topology topo) {
      choose_topo(topo);
      check_is_in_element(bulk->begin_nodes(elem), coordinate_field(), *meSS);
    }

    void particle_interpolation(stk::topology topo) {
      choose_topo(topo);
      check_particle_interp(bulk->begin_nodes(elem), coordinate_field(), *meSS);
    }

    void general_shape_fcn(stk::topology topo) {
      choose_topo(topo);
      check_general_shape_fcn(bulk->begin_nodes(elem), coordinate_field(), *meSS);
    }

    const VectorFieldType& coordinate_field() const {
      return *static_cast<const VectorFieldType*>(meta->coordinate_field());
    }

    stk::ParallelMachine comm;
    std::unique_ptr<stk::mesh::MetaData> meta;
    std::unique_ptr<stk::mesh::BulkData> bulk;
    stk::mesh::Entity elem;
    sierra::nalu::MasterElement* meSS;
    sierra::nalu::MasterElement* meSV;
};

#define TEST_F_ALL_TOPOS_NO_PYR(x, y) \
    TEST_F(x, tri##_##y)   { y(stk::topology::TRI_3_2D); }   \
    TEST_F(x, quad4##_##y)  { y(stk::topology::QUAD_4_2D); } \
    TEST_F(x, quad9##_##y)  { y(stk::topology::QUAD_9_2D); } \
    TEST_F(x, tet##_##y)   { y(stk::topology::TET_4); }      \
    TEST_F(x, wedge##_##y) { y(stk::topology::WEDGE_6); }    \
    TEST_F(x, hex8##_##y)   { y(stk::topology::HEX_8); }     \
    TEST_F(x, hex27##_##y)  { y(stk::topology::HEX_27); }

#define TEST_F_ALL_TOPOS(x, y) \
    TEST_F(x, tri##_##y)   { y(stk::topology::TRI_3_2D); }   \
    TEST_F(x, quad4##_##y)  { y(stk::topology::QUAD_4_2D); } \
    TEST_F(x, quad9##_##y)  { y(stk::topology::QUAD_9_2D); } \
    TEST_F(x, tet##_##y)   { y(stk::topology::TET_4); }      \
    TEST_F(x, pyr##_##y) { y(stk::topology::PYRAMID_5); }    \
    TEST_F(x, wedge##_##y) { y(stk::topology::WEDGE_6); }    \
    TEST_F(x, hex8##_##y)   { y(stk::topology::HEX_8); }     \
    TEST_F(x, hex27##_##y)  { y(stk::topology::HEX_27); }

#define TEST_F_ALL_P1_TOPOS(x, y) \
    TEST_F(x, tri##_##y)   { y(stk::topology::TRI_3_2D); }   \
    TEST_F(x, quad4##_##y)  { y(stk::topology::QUAD_4_2D); } \
    TEST_F(x, tet##_##y)   { y(stk::topology::TET_4); }      \
    TEST_F(x, wedge##_##y) { y(stk::topology::WEDGE_6); }    \
    TEST_F(x, pyr##_##y) { y(stk::topology::PYRAMID_5); }    \
    TEST_F(x, hex8##_##y)   { y(stk::topology::HEX_8); }

// Patch tests: pyramids fail
TEST_F_ALL_TOPOS_NO_PYR(MasterElement, scs_interpolation)
TEST_F_ALL_TOPOS_NO_PYR(MasterElement, scs_derivative)
TEST_F_ALL_TOPOS_NO_PYR(MasterElement, scv_interpolation)
TEST_F_ALL_TOPOS_NO_PYR(MasterElement, volume_integration)

// Pyramid fails since the reference element
// since the constant Jacobian assumption is violated
TEST_F_ALL_TOPOS_NO_PYR(MasterElement, is_in_element)

// Pyramid works. Doesn't work for higher-order elements sicne they have more ips than nodes
TEST_F_ALL_P1_TOPOS(MasterElement, scv_shifted_ips_are_nodal)
//TEST_F_ALL_P1_TOPOS(MasterElement, exposed_face_shifted_ips_are_nodal)

// works fore everything
TEST_F_ALL_TOPOS(MasterElement, is_not_in_element)
TEST_F_ALL_TOPOS(MasterElement, particle_interpolation)

TEST_F_ALL_P1_TOPOS(MasterElement, general_shape_fcn)
