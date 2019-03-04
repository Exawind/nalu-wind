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
    stk::mesh::BulkData&,
    const YAML::Node&,
    bool);

  virtual ~FrameBase() {}

  void setup();

  virtual void update_coordinates_velocity(const double) = 0;

  virtual const MotionBase::TransMatType& get_inertial_frame() const {
    throw std::runtime_error("FrameNonInertial: Invalid access of inertial frame"); };

  const std::vector<std::string> get_part_names() const {
    return partNamesVec_; }

  void set_ref_frame( MotionBase::TransMatType& frame ) {
    refFrame_ = frame; }

  void set_computed_centroid( std::vector<double>& centroid ) {
    for (size_t i=0; i < meshMotionVec_.size(); i++)
      meshMotionVec_[i]->set_computed_centroid(centroid); }

  bool is_inertial() const {
    return isInertial_; }

protected:
  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

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
  MotionBase::TransMatType refFrame_ = MotionBase::identityMat_;

  const bool isInertial_;

  bool computeCentroid_ = false;

private:
    FrameBase() = delete;
    FrameBase(const FrameBase&) = delete;

    void load(const YAML::Node&);

    void compute_centroid_on_parts(
      std::vector<double> &centroid);
};

} // nalu
} // sierra

#endif /* FRAMEBASE_H */
