// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <aero/fsi/CalcLoads.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// basic c++
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace fsi {
using namespace sierra::nalu;
std::array<double, 6>
accumulateLoadsAndMoments(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::PartVector& surface,
  const sierra::nalu::VectorFieldType& modelCoords,
  const sierra::nalu::VectorFieldType& meshDisp,
  const sierra::nalu::GenericFieldType& tforceSCS,
  std::array<double, 3> &center_of_mass)
{
  std::array<double, 6> accumulated_forces_and_moments = {0., 0., 0., 0., 0., 0.};
  // nodal fields to gather and store at ip's
  std::vector<double> ws_face_shape_function;

  std::vector<double> ws_coordinates;
  std::vector<double> coord_bip(3, 0.0);
  std::vector<double> coordref_bip(3, 0.0);

  std::array<double, 3> tforce_bip;

  std::vector<double> tmpNodePos(
    3, 0.0); // Vector to temporarily store a position vector
  std::vector<double> tmpNodeDisp(
    3, 0.0); // Vector to temporarily store a displacement vector

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();

  stk::mesh::Selector sel(
    meta.locally_owned_part() & stk::mesh::selectUnion(surface));
  const auto& bkts = bulk.get_buckets(meta.side_rank(), sel);
  for (auto b : bkts) {
    // face master element
    MasterElement* meFC =
      MasterElementRepo::get_surface_master_element_on_host(b->topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal;
    // face perspective (use with face_node_relations)
    ws_face_shape_function.resize(numScsBip * nodesPerFace);

    SharedMemView<double**, HostShmem> p_face_shape_function(
      ws_face_shape_function.data(), numScsBip, nodesPerFace);

    meFC->shape_fcn<>(p_face_shape_function);

    ws_coordinates.resize(ndim * nodesPerFace);

    for (size_t in = 0; in < b->size(); in++) {
      // get face
      stk::mesh::Entity face = (*b)[in];
      // face node relations
      stk::mesh::Entity const* face_node_rels = bulk.begin_nodes(face);
      // gather nodal data off of face
      for (int ni = 0; ni < nodesPerFace; ++ni) {
        stk::mesh::Entity node = face_node_rels[ni];
        // gather coordinates
        const double* xyz = stk::mesh::field_data(modelCoords, node);
        const double* xyz_disp = stk::mesh::field_data(meshDisp, node);
        for (auto i = 0; i < ndim; i++) {
          ws_coordinates[ni * ndim + i] = xyz[i] + xyz_disp[i];
        }
      }

      // Get reference to load map and loadMapInterp at all ips on this face
      const double* tforce = stk::mesh::field_data(tforceSCS, face);

      for (int ip = 0; ip < numScsBip; ++ip) {
        // Get coordinates and pressure force at this ip
        for (auto i = 0; i < ndim; i++) {
          coord_bip[i] = 0.0;
        }
        for (int ni = 0; ni < nodesPerFace; ni++) {
          const double r = p_face_shape_function(ip, ni);
          for (int i = 0; i < ndim; i++) {
            coord_bip[i] += r * ws_coordinates[ni * ndim + i];
          }
        }
        for (auto idim = 0; idim < 3; idim++)
          tforce_bip[idim] = tforce[ip * 3 + idim];

        // Now compute the force and moment on the interpolated reference
        // position
        fsi::computeEffForceMoment(
          tforce_bip.data(), coord_bip.data(), accumulated_forces_and_moments.data(),
          center_of_mass.data());
      }
    }
  }

  return accumulated_forces_and_moments;

}
}

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// CalcLoads - post process sigma_ijnjdS and pdS directly on the SCS's
//    Mostly copied from SurfaceForceAndMomentAlgorithm after removing
//    unnecessary parts
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
CalcLoads::CalcLoads(stk::mesh::PartVector& partVec, bool useShifted)
  : partVec_(partVec),
    useShifted_(useShifted),
    coordinates_(NULL),
    pressure_(NULL),
    density_(NULL),
    viscosity_(NULL),
    dudx_(NULL),
    exposedAreaVec_(NULL),
    tforceSCS_(NULL)
{
}

void
CalcLoads::setup(std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
}

void
CalcLoads::initialize()
{

  auto& meta = bulk_->mesh_meta_data();
  coordinates_ =
    meta.get_field<double>(stk::topology::NODE_RANK, "current_coordinates");
  pressure_ = meta.get_field<double>(stk::topology::NODE_RANK, "pressure");
  density_ = meta.get_field<double>(stk::topology::NODE_RANK, "density");
  viscosity_ =
    meta.get_field<double>(stk::topology::NODE_RANK, "effective_viscosity_u");

  if (viscosity_ == nullptr) {
    viscosity_ = meta.get_field<double>(stk::topology::NODE_RANK, "viscosity");
  }
  dudx_ = meta.get_field<double>(stk::topology::NODE_RANK, "dudx");

  exposedAreaVec_ =
    meta.get_field<double>(meta.side_rank(), "exposed_area_vector");
  tforceSCS_ = meta.get_field<double>(meta.side_rank(), "tforce_scs");
}
//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
CalcLoads::~CalcLoads()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
CalcLoads::execute()
{

  // common
  auto& meta = bulk_->mesh_meta_data();
  const int nDim = meta.spatial_dimension();

  // nodal fields to gather
  std::vector<double> ws_pressure;
  std::vector<double> ws_density;
  std::vector<double> ws_viscosity;

  // master element
  std::vector<double> ws_face_shape_function;

  // deal with state
  ScalarFieldType& densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  pressure_->sync_to_host();
  density_->sync_to_host();
  densityNp1.sync_to_host();
  viscosity_->sync_to_host();
  exposedAreaVec_->sync_to_host();
  tforceSCS_->sync_to_host();
  coordinates_->sync_to_host();
  dudx_->sync_to_host();

  const auto& bkts = bulk_->get_buckets(
    meta.side_rank(),
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_));
  for (auto b : bkts) {

    // face master element
    MasterElement* meFC =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        b->topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal; face perspective (use with
    // face_node_relations)
    const int* faceIpNodeMap = meFC->ipNodeMap();

    // algorithm related; element
    ws_pressure.resize(nodesPerFace);
    ws_density.resize(nodesPerFace);
    ws_viscosity.resize(nodesPerFace);
    ws_face_shape_function.resize(numScsBip * nodesPerFace);

    // pointers
    double* p_pressure = &ws_pressure[0];
    double* p_density = &ws_density[0];
    double* p_viscosity = &ws_viscosity[0];
    SharedMemView<double**, HostShmem> p_face_shape_function(
      ws_face_shape_function.data(), numScsBip, nodesPerFace);

    // shape functions
    if (useShifted_)
      meFC->shifted_shape_fcn<>(p_face_shape_function);
    else
      meFC->shape_fcn<>(p_face_shape_function);

    const stk::mesh::Bucket::size_type length = b->size();

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      // get face
      stk::mesh::Entity face = (*b)[k];

      // face node relations
      stk::mesh::Entity const* face_node_rels = bulk_->begin_nodes(face);

      //======================================
      // gather nodal data off of face
      //======================================
      for (int ni = 0; ni < nodesPerFace; ++ni) {
        stk::mesh::Entity node = face_node_rels[ni];
        // gather scalars
        p_pressure[ni] = *stk::mesh::field_data(*pressure_, node);
        p_density[ni] = *stk::mesh::field_data(densityNp1, node);
        p_viscosity[ni] = *stk::mesh::field_data(*viscosity_, node);
      }

      // pointer to face data
      const double* areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
      double* tforce_scs = stk::mesh::field_data(*tforceSCS_, face);

      for (int ip = 0; ip < numScsBip; ++ip) {

        // offsets
        const int offSetAveraVec = ip * nDim;
        const int offSetForce = nDim * ip;
        const int localFaceNode = faceIpNodeMap[ip];

        // interpolate to bip
        double pBip = 0.0;
        // double rhoBip = 0.0;
        double muBip = 0.0;
        for (int ic = 0; ic < nodesPerFace; ++ic) {
          const double r = p_face_shape_function(ip, ic);
          pBip += r * p_pressure[ic];
          // rhoBip += r * p_density[ic];
          muBip += r * p_viscosity[ic];
        }

        // extract nodal fields
        stk::mesh::Entity node = face_node_rels[localFaceNode];
        const double* duidxj = stk::mesh::field_data(*dudx_, node);

        // divU and aMag
        double divU = 0.0;
        for (int j = 0; j < nDim; ++j)
          divU += duidxj[j * nDim + j];

        // assemble force -sigma_ij*njdS
        for (int i = 0; i < nDim; ++i) {
          const double ai = areaVec[offSetAveraVec + i];
          double dflux = 0.0;
          const int offSetI = nDim * i;
          for (int j = 0; j < nDim; ++j) {
            const int offSetTrans = nDim * j + i;
            dflux += -muBip * (duidxj[offSetI + j] + duidxj[offSetTrans]) *
                     areaVec[offSetAveraVec + j];
          }
          tforce_scs[offSetForce + i] =
            pBip * ai + dflux + 2.0 / 3.0 * muBip * divU * ai;
        }
      }
    }
  }
  tforceSCS_->modify_on_host();
}

} // namespace nalu
} // namespace sierra
