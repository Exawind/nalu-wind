#include "mesh_motion/MeshMotionAlg.h"

#include "mesh_motion/FrameMoving.h"
#include "mesh_motion/FrameOpenFAST.h"

#include "NaluParsing.h"

#include <cassert>
#include <iostream>

namespace sierra{
namespace nalu{

MeshMotionAlg::MeshMotionAlg(
  stk::mesh::BulkData& bulk,
  const YAML::Node& node,
  OpenfastFSI* openfast)
{
  load(bulk, node, openfast);

  set_deformation_flag();
}

void MeshMotionAlg::load(
  stk::mesh::BulkData& bulk,
  const YAML::Node& node,
  OpenfastFSI* openfast)
{
  // get motion information for entire mesh
  const int num_groups = node.size();
  movingFrameVec_.resize(num_groups);

  for (int i=0; i < num_groups; i++) {

    // extract current motion group info
    const auto& ginfo = node[i];

    movingFrameVec_[i].reset(new FrameMoving(bulk, ginfo));
  }

  if (openfast != NULL) {
      int nTurbinesGlob = openfast->get_nTurbinesGlob();
      movingFrameVec_.resize(num_groups + nTurbinesGlob);

      int n_moving_turb = 0;
      for (auto iTurb=0; iTurb < nTurbinesGlob; iTurb++) {
          fsiTurbine *fsiTurbineData = openfast->get_fsiTurbineData(iTurb);
          YAML::Node node; //Empty node
          if (fsiTurbineData != NULL) { //Could be a turbine handled through actuator line or something
              movingFrameVec_[num_groups+n_moving_turb].reset(new FrameOpenFAST(bulk, node, fsiTurbineData));
              n_moving_turb += 1;
          }
      }
  }

}

void MeshMotionAlg::set_deformation_flag()
{
  for (size_t i=0; i < movingFrameVec_.size(); i++)
    if( movingFrameVec_[i]->is_deforming() )
      isDeforming_ = true;
}

void MeshMotionAlg::initialize( const double time )
{
  if(isInit_)
    throw std::runtime_error("MeshMotionAlg::initialize(): Re-initialization of MeshMotionAlg not valid");

  for (size_t i=0; i < movingFrameVec_.size(); i++)
  {
    movingFrameVec_[i]->setup();

    // update coordinates and velocity
    movingFrameVec_[i]->update_coordinates_velocity(time);
  }

  isInit_ = true;
}

void MeshMotionAlg::execute(const double time)
{
  for (size_t i=0; i < movingFrameVec_.size(); i++) {
    movingFrameVec_[i]->update_coordinates_velocity(time);
  }
}

void MeshMotionAlg::post_compute_geometry()
{
  for (size_t i=0; i < movingFrameVec_.size(); i++)
    movingFrameVec_[i]->post_compute_geometry();
}

stk::mesh::PartVector MeshMotionAlg::get_partvec()
{ 
  stk::mesh::PartVector fpartVec;
  for (size_t i=0; i < movingFrameVec_.size(); i++) {
    stk::mesh::PartVector fPartVec = movingFrameVec_[i]->get_partvec();
    for (auto p: fPartVec)
      fpartVec.push_back(p);
  }
  return fpartVec;
}

} // nalu
} // sierra
