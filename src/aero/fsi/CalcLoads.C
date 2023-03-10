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
#include <master_element/MasterElementFactory.h>
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
CalcLoads::initialize() {
    
  auto& meta = bulk_->mesh_meta_data();
  coordinates_ = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  pressure_ =
    meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ =
    meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  viscosity_ = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "effective_viscosity_u");
  dudx_ = meta.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  exposedAreaVec_ =
    meta.get_field<GenericFieldType>(meta.side_rank(), "exposed_area_vector");
  tforceSCS_ = meta.get_field<GenericFieldType>(meta.side_rank(), "tforce_scs");
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

  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

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

    // extract connected element topology
    b->parent_topology(stk::topology::ELEMENT_RANK, parentTopo);
    ThrowAssert(parentTopo.size() == 1);
    stk::topology theElemTopo = parentTopo[0];

    // extract master element for this element topo
    MasterElement* meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(
        theElemTopo);

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

      // extract the connected element to this exposed face; should be single in
      // size!
      const stk::mesh::Entity* face_elem_rels = bulk_->begin_elements(face);
      ThrowAssert(bulk_->num_elements(face) == 1);

      // get element; its face ordinal number
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_->begin_element_ordinals(face)[0];

      // get the relations off of element
      stk::mesh::Entity const* elem_node_rels = bulk_->begin_nodes(element);

      for (int ip = 0; ip < numScsBip; ++ip) {

        // offsets
        const int offSetAveraVec = ip * nDim;
        const int localFaceNode = faceIpNodeMap[ip];
        const int opposingNode = meSCS->opposingNodes(face_ordinal, ip);

        // interpolate to bip
        double pBip = 0.0;
        double rhoBip = 0.0;
        double muBip = 0.0;
        for (int ic = 0; ic < nodesPerFace; ++ic) {
          const double r = p_face_shape_function(ip, ic);
          pBip += r * p_pressure[ic];
          rhoBip += r * p_density[ic];
          muBip += r * p_viscosity[ic];
        }

        // extract nodal fields
        stk::mesh::Entity node = face_node_rels[localFaceNode];
        const double* coord = stk::mesh::field_data(*coordinates_, node);
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
          tforce_scs[offSetAveraVec + i] =
            pBip * ai + dflux + 2.0 / 3.0 * muBip * divU * ai;
        }
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
