#ifndef HDF5TABLEPROPALGORITHM_H
#define HDF5TABLEPROPALGORITHM_H

#include <Algorithm.h>

#include <vector>
#include <set>
#include <string>
#include <map>

#include <tabular_props/HDF5Table.h>

namespace stk {
namespace mesh {
class FieldBase;
class Part;
class MetaData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

// Forward declarations
class Realm;
class H5IO;

/**
 *  @class  HDF5TablePropAlgorithm
 *  @brief  Object to manage property evaluation as a function of a set of
 *          input chemical state variables
 *
 *  This class provides the public interface for performing property
 *  evaluations as a function of chemical state variables.  It acts as a
 *  wrapper for a contained multidimensional lookup table and any number
 *  of (optional) input conversion operations required by the particular
 *  turbulence/chemistry interaction model being used.  The internal
 *  structure does not matter to the calling routine.
 */
class HDF5TablePropAlgorithm : public Algorithm
{

public:
  /**
   *  Construct an HDF5TablePropAlgorithm.
   */
  HDF5TablePropAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    H5IO* fileIO,
    stk::mesh::FieldBase* prop,
    std::string tablePropName,
    std::vector<std::string>& indVarNameVec,
    std::vector<std::string>& indVarTableNameVec,
    const stk::mesh::MetaData& meta_data);

  virtual ~HDF5TablePropAlgorithm();

  stk::mesh::FieldBase* prop_;
  std::string tablePropName_;
  std::vector<std::string> indVarTableNameVec_;
  const size_t indVarSize_;

  std::vector<stk::mesh::FieldBase*> indVar_;
  std::vector<double*> workIndVar_;
  std::vector<double> workZ_;

  /** execute Algorithm */
  virtual void execute();

  /** Get the name of the variable returned by a query to this
   * HDF5TablePropAlgorithm */
  const std::string& name() const { return tablePropName_; }

  /** Get the list of input variables required by calls to query(), in the
   *  order that they are to be provided.  */
  const std::vector<std::string>& input_names() const { return inputNames_; }

private:
  // Add the current values to the clipping event log
  void log_clip_event(const std::vector<double>& values) const;

  // HDF5 file pointer
  H5IO* fileIO_;

  // HDF5 table holding for tablePropName_
  HDF5Table* table_;

  // Names of the inputs required by the query() function
  std::vector<std::string> inputNames_;
};

// typedef SharedPtr<const HDF5TablePropAlgorithm>
// ConstHDF5TablePropAlgorithmPtr;

} // namespace nalu
} // namespace sierra

#endif
