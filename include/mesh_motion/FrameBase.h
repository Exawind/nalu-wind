#ifndef FRAMEBASE_H
#define FRAMEBASE_H

#include "MotionBase.h"

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
    const YAML::Node&,
    bool);

  virtual ~FrameBase()
  {
  }

  void setup();

  void set_computed_centroid( std::vector<double>& centroid )
  {
    for (size_t i=0; i < meshMotionVec_.size(); i++)
      meshMotionVec_[i]->set_computed_centroid(centroid);
  }

  virtual void post_compute_geometry()
  {
  }

protected:
  /** Compute transformation matrix
   *
   * @return 4x4 matrix representing composite addition of motions
   */
  MotionBase::TransMatType compute_transformation(
    const double,
    const double*);

  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

  /** Motion/Transformation vector
   *
   *  A vector of size number of motion/transformation groups
   */
  std::vector<std::unique_ptr<MotionBase>> meshMotionVec_;

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

private:
  FrameBase() = delete;
  FrameBase(const FrameBase&) = delete;

  void load(const YAML::Node&);

  void populate_part_vec(const YAML::Node&);

  void compute_centroid_on_parts(
    std::vector<double> &centroid);
};

} // nalu
} // sierra

#endif /* FRAMEBASE_H */
