#ifndef HDF5FILEPTR_H
#define HDF5FILEPTR_H

#include <vector>
#include <map>
#include <string>

namespace sierra {
namespace nalu {

// Forward declarations
class H5IO;

/**
 *  @class  HDF5FilePtr
 *  @brief  Provides a HDF5 formatted file pointer
 *
 *  Given a filename pertaining to an HDF5 formatted file, this file will
 *  be opened, the names of properties tabulatedin that file is read
 *  and the file pointer can be returned.
 */

class HDF5FilePtr
{

public:
  /**
   *  Construct an empty HDF5FilePtr.  It should then be filled with
   *  Property objects by making repeated calls to add_entry().
   */
  explicit HDF5FilePtr(const std::string& fileName = "");

  ~HDF5FilePtr();

  /** Query if the given property exists in the library */
  bool has_entry(const std::string& name) const;

  /** Get a list of all contained properties */
  std::vector<std::string> property_names() const;

  const std::string& filename() { return fileName_; }

  /**
   *  Read the entire library from an HDF5 file with the name set in the
   *  constructor.
   */
  void read_hdf5();

  /** returns the pointer to an opened HDF5 file */
  H5IO* get_H5IO();

  /**  Don't need to print summary in Nalu
   *  Print a summary of the contained data, including independent variable
   *  mesh points, clipping values, and the contained properties.
  void print_summary() const;
   */

private:
  HDF5FilePtr(const HDF5FilePtr&);           // no copying
  HDF5FilePtr operator=(const HDF5FilePtr&); // no assignment

  /** list of properties contained in the file */
  std::vector<std::string> propertyNames_;

  /** Name of the file that the library is tied to */
  std::string fileName_;

  /** File version to write */
  int exportFileVersion_;

  /** Pointer to table of properties */
  H5IO* fileIO_;
};

} // namespace nalu
} // namespace sierra

#endif
