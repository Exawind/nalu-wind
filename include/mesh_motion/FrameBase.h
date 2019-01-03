#ifndef FRAMEBASE_H
#define FRAMEBASE_H

#include "MotionBase.h"

// stk base header files
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/MetaData.hpp"

namespace sierra{
namespace nalu{

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<double> ScalarFieldType;

class FrameBase
{
public:
  FrameBase(
    stk::mesh::MetaData&,
    stk::mesh::BulkData&,
    const YAML::Node&,
    bool);

  virtual ~FrameBase() {}

  virtual void update_coordinates_velocity(const double) = 0;

  virtual const MotionBase::transMatType& get_inertial_frame() const {
    throw std::runtime_error("FrameNonInertial: Invalid access of inertial frame"); };

  const std::vector<std::string> get_part_names() const {
    return partNamesVec_; }

  void set_ref_frame( MotionBase::transMatType& frame ) {
    refFrame_ = frame; }

  void set_computed_centroid( std::vector<double>& centroid ) {
    for (int i=0; i < meshMotionVec_.size(); i++)
      meshMotionVec_[i]->set_computed_centroid(centroid); }

  const bool is_inertial() const {
    return isInertial_; }

  const bool compute_centroid() const {
    return computeCentroid_; }

protected:
  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  /** Motion vector
   *
   *  A vector of size number of motion groups
   */
  std::vector<std::unique_ptr<MotionBase>> meshMotionVec_;

  /** Motion part names
   *
   *  A vector of size number of parts
   */
  std::vector<std::string> partNamesVec_;

  /** Motion parts
   *
   *  A vector of size number of parts
   */
  stk::mesh::PartVector partVec_;

  /** Reference frame
   *
   * A 4x4 matrix that defines the reference frame for subsequent motions
   * It is initialized to an identity matrix
   */
  MotionBase::transMatType refFrame_ = MotionBase::identityMat_;

  const bool isInertial_;

  bool computeCentroid_ = false;

private:
    FrameBase() = delete;
    FrameBase(const FrameBase&) = delete;

    void load(const YAML::Node&);
};

} // nalu
} // sierra

#endif /* FRAMEBASE_H */
