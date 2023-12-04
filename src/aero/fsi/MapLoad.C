// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <aero/fsi/MapLoad.h>
#include <aero/aero_utils/ForceMoment.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementRepo.h>

namespace fsi {

using namespace sierra::nalu;

void
mapTowerLoad(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::PartVector& twrBndyParts,
  const sierra::nalu::VectorFieldType& modelCoords,
  const sierra::nalu::VectorFieldType& meshDisp,
  const sierra::nalu::GenericIntFieldType& loadMap,
  const sierra::nalu::GenericFieldType& loadMapInterp,
  const sierra::nalu::GenericFieldType& tforceSCS,
  std::vector<double>& twrRefPos,
  std::vector<double>& twrDef,
  std::vector<double>& twrLoad)
{
  // nodal fields to gather and store at ip's
  std::vector<double> ws_face_shape_function;

  std::vector<double> ws_coordinates;
  std::vector<double> coord_bip(3, 0.0);
  std::vector<double> coordref_bip(3, 0.0);

  std::array<double, 3> face_center;

  std::array<double, 3> tforce_bip;

  std::vector<double> tmpNodePos(
    3, 0.0); // Vector to temporarily store a position vector
  std::vector<double> tmpNodeDisp(
    3, 0.0); // Vector to temporarily store a displacement vector

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();

  stk::mesh::Selector sel(
    meta.locally_owned_part() & stk::mesh::selectUnion(twrBndyParts));
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
      const int* loadMapFace = stk::mesh::field_data(loadMap, face);
      const double* loadMapInterpFace =
        stk::mesh::field_data(loadMapInterp, face);
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

        const int loadMap_bip = loadMapFace[ip];
        const double loadMapInterp_bip = loadMapInterpFace[ip];

        for (auto idim = 0; idim < 3; idim++)
          tforce_bip[idim] = tforce[ip * 3 + idim];

        // Find the interpolated reference position first
        linInterpVec(
          &twrRefPos[(loadMap_bip)*6], &twrRefPos[(loadMap_bip + 1) * 6],
          loadMapInterp_bip, tmpNodePos.data());
        // Find the interpolated linear displacement
        linInterpVec(
          &twrDef[(loadMap_bip)*6], &twrDef[(loadMap_bip + 1) * 6],
          loadMapInterp_bip, tmpNodeDisp.data());
        // Add displacement to find actual position
        for (auto idim = 0; idim < 3; idim++)
          tmpNodePos[idim] += tmpNodeDisp[idim];

        // Temporarily store total force and moment as (fX, fY, fZ, mX, mY, mZ)
        std::vector<double> tmpForceMoment(6, 0.0);
        // Now compute the force and moment on the interpolated reference
        // position
        fsi::computeEffForceMoment(
          tforce_bip.data(), coord_bip.data(), tmpForceMoment.data(),
          tmpNodePos.data());
        // Split the force and moment into the two surrounding nodes in a
        // variationally consistent manner using the interpolation factor
        fsi::splitForceMoment(
          tmpForceMoment.data(), loadMapInterp_bip, &(twrLoad[(loadMap_bip)*6]),
          &(twrLoad[(loadMap_bip + 1) * 6]));
      }
    }
  }
}

void
mapBladeLoad(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::PartVector& bldBndyParts,
  const sierra::nalu::VectorFieldType& modelCoords,
  const sierra::nalu::VectorFieldType& meshDisp,
  const sierra::nalu::GenericIntFieldType& loadMap,
  const sierra::nalu::GenericFieldType& loadMapInterp,
  const sierra::nalu::GenericFieldType& tforceSCS,
  int nPtsBlade,
  int iStart,
  std::vector<double>& bldRloc,
  std::vector<double>& bldRefPos,
  std::vector<double>& bldDef,
  std::vector<double>& bldLoad)
{
  std::vector<double> ws_face_shape_function;

  std::vector<double> ws_coordinates;
  std::vector<double> coord_bip(3, 0.0);
  std::vector<double> coordref_bip(3, 0.0);

  std::array<double, 3> face_center;

  std::array<double, 3> tforce_bip;

  std::vector<double> tmpNodePos(
    3, 0.0); // Vector to temporarily store a position vector
  std::vector<double> tmpNodeDisp(
    3, 0.0); // Vector to temporarily store a displacement vector

  const stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  stk::mesh::Selector sel(
    meta.locally_owned_part() & stk::mesh::selectUnion(bldBndyParts));
  const auto& bkts = bulk.get_buckets(meta.side_rank(), sel);
  for (auto b : bkts) {
    // face master element
    MasterElement* meFC =
      MasterElementRepo::get_surface_master_element_on_host(b->topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal;
    // face perspective (use with face_node_relations)
    const int* faceIpNodeMap = meFC->ipNodeMap();

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

      for (auto i = 0; i < ndim; i++)
        face_center[i] = 0.0;
      // gather nodal data off of face
      for (int ni = 0; ni < nodesPerFace; ++ni) {
        stk::mesh::Entity node = face_node_rels[ni];
        // gather coordinates
        const double* xyz = stk::mesh::field_data(modelCoords, node);
        const double* xyz_disp = stk::mesh::field_data(meshDisp, node);
        for (auto i = 0; i < ndim; i++) {
          ws_coordinates[ni * ndim + i] = xyz[i] + xyz_disp[i];
          face_center[i] += xyz[i];
        }
      }
      for (auto i = 0; i < ndim; i++)
        face_center[i] /= nodesPerFace;

      // Get reference to load map and loadMapInterp at all ips on this face
      const int* loadMapFace = stk::mesh::field_data(loadMap, face);
      const double* loadMapInterpFace =
        stk::mesh::field_data(loadMapInterp, face);
      const double* tforce = stk::mesh::field_data(tforceSCS, face);

      for (int ip = 0; ip < numScsBip; ++ip) {

        // Get coordinates and pressure force at this ip
        for (auto i = 0; i < ndim; i++)
          coord_bip[i] = 0.0;
        for (int ni = 0; ni < nodesPerFace; ni++) {
          const double r = p_face_shape_function(ip, ni);
          for (int i = 0; i < ndim; i++)
            coord_bip[i] += r * ws_coordinates[ni * ndim + i];
        }

        int loadMap_bip = loadMapFace[ip];
        // Radial location of scs center projected onto blade beam mesh
        double interpFac = loadMapInterpFace[ip];
        double r_n = bldRloc[iStart + loadMap_bip];
        double r_np1 = bldRloc[iStart + loadMap_bip + 1];
        double rloc_proj = r_n + interpFac * (r_np1 - r_n);

        // Find the interpolated reference position first
        fsi::linInterpVec(
          &bldRefPos[(loadMap_bip + iStart) * 6],
          &bldRefPos[(loadMap_bip + iStart + 1) * 6], interpFac,
          tmpNodePos.data());

        // Find the interpolated linear displacement
        fsi::linInterpVec(
          &bldDef[(loadMap_bip + iStart) * 6],
          &bldDef[(loadMap_bip + iStart + 1) * 6], interpFac,
          tmpNodeDisp.data());

        // Add displacement to find actual position
        for (auto idim = 0; idim < 3; idim++)
          tmpNodePos[idim] += tmpNodeDisp[idim];

        for (auto idim = 0; idim < 3; idim++)
          tforce_bip[idim] = tforce[ip * 3 + idim];

        // Temporarily store total force and moment as (fX, fY, fZ,
        // mX, mY, mZ)
        std::vector<double> tmpForceMoment(6, 0.0);
        // Now compute the force and moment on the interpolated
        // reference position
        fsi::computeEffForceMoment(
          tforce_bip.data(), coord_bip.data(), tmpForceMoment.data(),
          tmpNodePos.data());

        // Calculate suport length for this scs along local blade direction
        const int nfNode = faceIpNodeMap[ip];
        stk::mesh::Entity nnode = face_node_rels[nfNode];
        const double* xyz = stk::mesh::field_data(modelCoords, nnode);
        double sl = 0.0;
        for (auto i = 0; i < ndim; i++) {
          sl += (xyz[i] - face_center[i]) *
                (bldRefPos[(loadMap_bip + iStart + 1) * 6 + i] -
                 bldRefPos[(loadMap_bip + iStart) * 6 + i]);
        }
        sl = std::abs(sl / (r_np1 - r_n));
        double sl_ratio = sl / (r_np1 - r_n);

        if ((loadMap_bip < 1) || (loadMap_bip > (nPtsBlade - 3))) {
          // if (true) {
          // Now split the force and moment on the interpolated
          // reference position into the 'left' and 'right' nodes
          fsi::splitForceMoment(
            tmpForceMoment.data(), interpFac,
            &(bldLoad[(loadMap_bip + iStart) * 6]),
            &(bldLoad[(loadMap_bip + iStart + 1) * 6]));
        } else {

          // Split into 2 and distribute to nearest element
          for (auto i = 0; i < 6; i++)
            tmpForceMoment[i] *= 0.5;

          fsi::splitForceMoment(
            tmpForceMoment.data(), interpFac,
            &(bldLoad[(loadMap_bip + iStart) * 6]),
            &(bldLoad[(loadMap_bip + iStart + 1) * 6]));

          for (auto i = 0; i < 6; i++)
            tmpForceMoment[i] *= 0.5;

          if ((interpFac - 0.5 * sl_ratio) < 0.0) {
            double r_nm1 = bldRloc[iStart + loadMap_bip - 1];
            double l_interpFac = (rloc_proj - 0.5 * sl - r_nm1) / (r_n - r_nm1);
            fsi::splitForceMoment(
              tmpForceMoment.data(), l_interpFac,
              &(bldLoad[(loadMap_bip - 1 + iStart) * 6]),
              &(bldLoad[(loadMap_bip + iStart) * 6]));

          } else {
            double l_interpFac = interpFac - 0.5 * sl_ratio;
            fsi::splitForceMoment(
              tmpForceMoment.data(), l_interpFac,
              &(bldLoad[(loadMap_bip + iStart) * 6]),
              &(bldLoad[(loadMap_bip + iStart + 1) * 6]));
          }

          if ((interpFac + 0.5 * sl_ratio) > 1.0) {
            double r_np2 = bldRloc[iStart + loadMap_bip + 2];
            double r_interpFac =
              (rloc_proj + 0.5 * sl - r_np1) / (r_np2 - r_np1);
            fsi::splitForceMoment(
              tmpForceMoment.data(), r_interpFac,
              &(bldLoad[(loadMap_bip + iStart + 1) * 6]),
              &(bldLoad[(loadMap_bip + iStart + 2) * 6]));
          } else {
            double r_interpFac = interpFac + 0.5 * sl_ratio;
            fsi::splitForceMoment(
              tmpForceMoment.data(), r_interpFac,
              &(bldLoad[(loadMap_bip + iStart) * 6]),
              &(bldLoad[(loadMap_bip + iStart + 1) * 6]));
          }
        }
      }
    }
  }
}

} // namespace fsi
