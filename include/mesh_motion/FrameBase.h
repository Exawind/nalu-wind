#ifndef FRAMEBASE_H
#define FRAMEBASE_H

#include "NgpMotion.h"

// stk base header files
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/MetaData.hpp"

namespace YAML { class Node; }

namespace sierra{
namespace nalu{

class FrameBase
{
public:
  FrameBase(
    stk::mesh::BulkData&,
    const YAML::Node&);

  virtual ~FrameBase();

  virtual void setup();

  void compute_centroid_on_parts(
    mm::ThreeDVecType& centroid);

  void set_computed_centroid(const mm::ThreeDVecType& centroid)
  {
    for (size_t i=0; i < motionKernels_.size(); i++)
      motionKernels_[i]->set_computed_centroid(centroid);
  }

  virtual void post_compute_geometry()
  {
  }

  stk::mesh::PartVector get_partvec() {
    return partVec_;
  };

  bool is_deforming(){ return isDeforming_; }

protected:
  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

  /** Motion/Transformation vector
   *
   *  A vector of size number of motion/transformation groups
   */
  std::vector<std::unique_ptr<NgpMotion>> motionKernels_;

  /** Motion parts
   *
   *  A vector of size number of parts
   */
  stk::mesh::PartVector partVec_;

  /** Motion parts on Bc
   *
   *  A vector of size number of parts required for divergence computation
   */
  stk::mesh::PartVector partVecBc_;

  bool computeCentroid_ = false;

  // flag to denote if mesh deformation exists
  bool isDeforming_ = false;

private:
  FrameBase() = delete;
  FrameBase(const FrameBase&) = delete;

  void load(const YAML::Node&);

  void populate_part_vec(const YAML::Node&);
};

} // nalu
} // sierra

#endif /* FRAMEBASE_H */
