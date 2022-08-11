#ifndef HDF5TABLE_H
#define HDF5TABLE_H

#include <vector>
#include <set>
#include <string>
#include <map>

#include "tabular_props/H5IO.h"

namespace sierra {
namespace nalu {

// Forward declarations
// class Realm;
class Converter;
class H5IO;
class BSpline;

struct ClipEvent
{
  double severity;
  std::vector<double> values;
};

template <class T>
class ClipEventSortCriterion
{
public:
  bool operator()(const T& v1, const T& v2) const
  {
    // Sort based on the severity only, in decreasing order
    return v1.severity > v2.severity;
  }
};

typedef std::set<ClipEvent, ClipEventSortCriterion<ClipEvent>> ClipEventLog;

/**
 *  @class  HDF5Table
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
class HDF5Table
{

public:
  typedef std::map<std::string, std::string> Attributes;

  /** Construct an empty HDF5Table
   *  This can be filled later with read_hdf5( H5IO fileIO ).
   */
  HDF5Table();
  /**
   *  Construct an HDF5Table starting with the HDF5 file pointer fileIO
   *  and calling read_hdf5() method to pull data from disk.
   */
  HDF5Table(
    H5IO* fileIO,
    std::string tablePropName,
    std::vector<std::string>& indVarNameVec,
    std::vector<std::string>& indVarTableNameVec);

  virtual ~HDF5Table();

  /** Insert the provided Converter into this HDF5Table object.  It will be
   *  automatically wired into the central Table, given synchronization of
   *  input and output variables.  More than one converter may be added
   *  to a HDF5Table. */
  void add_converter(const Converter* converter);

  /** Get the name of the variable returned by a query to this HDF5Table */
  const std::string& name() const { return name_; }

  /** Get the list of input variables required by calls to query(), in the
   *  order that they are to be provided.  */
  const std::vector<std::string>& input_names() const { return inputNames_; }

  /* Get the number of inputs required by the query() function */
  unsigned int dimension() const { return dimension_; }

  /**
   *  Return the property value as a function of the provided input variables.
   *  Input bounds clipping of the internal lookup table is enforced, and
   *  a log of clipped queries is stored.
   *
   *  @param inputs : Array of independent variable values
   *  @result : The property as a function of the inputs
   */
  double query(const std::vector<double>& inputs) const;

  /**
   *  Return the property value as a function of the provided input variables.
   *  WARNING: No input bounds clipping is enforced, and no logs are stored
   *  of out-of-bounds queries.  This query method allows extrapolation,
   *  which can be dangerous!  Only use this if you know what you are doing!
   *
   *  @param inputs : Array of independent variable values
   *  @result : The property as a function of the inputs
   */
  double raw_query(const std::vector<double>& inputs) const;

  /** Set the number of clipping events we want to log */
  void set_clipping_log_size(unsigned int size);

  /** Return the current count of clipping events that have occurred */
  unsigned int num_clipping_events() const;

  /** Return the buffer of input coordinates that resulted in a clipping
   *  event
   */
  const ClipEventLog& clipping_event_log() const;

  /** Return the list of input variables corresponding to the values in
   *  the clipping event log.  This is essentially the inputs to the
   *  internal interpolation table.
   */
  const std::vector<std::string>& clipping_event_input_names() const
  {
    return inputNames_;
  }

  /** Return the list of minimum internal clipping bounds */
  const std::vector<double>& clipping_event_min_bounds() const;

  /** Return the list of maximum internal clipping bounds */
  const std::vector<double>& clipping_event_max_bounds() const;

  /** Reset the internal clipping event logging */
  void clear_clipping_log();

  /** Return the number of Converters.  If the number is zero, then the
   *  inputs to the HDF5Table will match the inputs to the internal Table,
   *  and the Table can be queried for overall HDF5Table configuration like
   *  independent variable min, max, and log scale. */
  unsigned int num_converters() const { return converters_.size(); }

  /** Query if the table contains the named attribute */
  bool has_attribute(const std::string& name) const;

  /** Request the named attribute.  An empty string is returned if not found. */
  const std::string& attribute(const std::string& name) const;

  /** Read tablePropName_ entry from HDF5Table specified in fileIO_ */
  void read_hdf5_property();

  /** Read HDF5Table from the specified HDF5 device, an already-specified
   * subtable.  This is also used to read Converter tables. */
  void read_hdf5_table(H5IO& io);

private:
  // Add the current values to the clipping event log
  void log_clip_event(const std::vector<double>& values) const;

  /** Rewire the inputs and outputs of the Table and any optional Converters
   *  so that they talk to each other properly and inputs to the HDF5Table will
   *  be sent to the correct object. */
  void update_input_mapping();

  /** Return the index correspondig to the variable name in nameVector */
  int
  findix(const std::vector<std::string>& nameVector, const std::string& name);

  // HDF5 file pointer
  H5IO* fileIO_;

  // name of dependent variable as given in the table
  std::string tablePropName_;

  // names of independent variables as given in the table
  std::vector<std::string> indVarTableNameVec_;

  // number of independent variables
  const size_t indVarSize_;

  // Number of inputs required by the query() function
  unsigned int dimension_;

  // Name of the variable returned by calls to query()
  std::string name_;

  // Names of the inputs required by the query() function
  std::vector<std::string> inputNames_;

  // List of optional input converters for the table
  std::vector<const Converter*> converters_;

  // Table and Converter input and output index mapping data
  std::vector<unsigned int> directInputIndex_;
  std::vector<unsigned int> convTableIndex_;
  std::vector<std::vector<unsigned int>> convInputIndex_;
  // used to map between input ordered by indVarTableNameVec_
  // and input required by ordering of inputNames_
  std::vector<unsigned int> indexIndVar_;

  //////////////////////
  // FROM Atab_Table.h
  //////////////////////

  // Status of internal log scale use for each independent variable
  std::vector<unsigned int> inputLogScale_;

  // Minimum and maximum values for the independent variables
  std::vector<double> meshMin_;
  std::vector<double> meshMax_;

  // Minimum and maximum clipping values and clipping status for the
  // independent variables
  std::vector<double> inputMin_;
  std::vector<double> inputMax_;

  // Independent variable mesh points used to build the interpolator
  std::vector<std::vector<double>> mesh_;

  // Minimum and maximum interpolated variable value
  double valueMin_;
  double valueMax_;

  // Arbitrary metadata
  Attributes attributes_;

  // Internal interpolator used to perform table lookups
  BSpline* spline_;

  // Buffers for storing clipping diagnostic information
  mutable unsigned int clipEventLogSize_;
  mutable unsigned int numClipped_;
  mutable ClipEventLog clipEventLog_;

  // delete this and use lookupBuf_ instead
  //  double * tableBuf_;
  //  Scratch space for conversions
  mutable std::vector<double> converterBuf_;

  // Scratch space for doing log scale conversions, etc. on the input variables
  mutable std::vector<double> lookupBuffer_;
  // Scratch space for doing bounds clipping on lookupBuffer_
  mutable std::vector<double> lookupBufferChecked_;
};

// typedef SharedPtr<const HDF5Table> ConstHDF5TablePtr;

} // namespace nalu
} // namespace sierra

#endif
