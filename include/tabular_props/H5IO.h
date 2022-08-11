#ifndef H5IO_H
#define H5IO_H

#include <hdf5.h>

#include <string>
#include <vector>

namespace sierra {
namespace nalu {

//============================================================================
/**
 *  \class H5IO
 *  \brief Simple wrapper around the HDF5 library to make usage easier
 *
 *  This wrapper supports opening a new file for writing or an old file
 *  for reading.  Once open, the H5IO object allows reading and writing of
 *  both attributes (meta-data) and datasets (large blocks of data) to the
 *  file.  Sub-groups, similar to sub-directories, can be opened inside the
 *  file and written to.  Typical usage involves passing name/value pairs
 *  to the API, and works something like this:
 *
 *      H5IO io;
 *      io.create_file( "fileName.h5" );

 *      io.write_attribute( "attribute1", attribute1 );
 *      io.write_attribute( "attribute2", attribute2 );
 *      io.write_dataset( "data1", data1 );
 *      io.write_dataset( "data2", data2 );
 *
 *      H5IO group = io.create_group( "groupName" );
 *      group.write_attribute( "attribute3", attribute3 );
 *      group.write_dataset( "data3", data3 );
 *
 *      io.close_file();
 *
 *  This data can be read back from disk using a similar procedure.  Note
 *  that the variables can be read back from the file in a different order
 *  (or not at all).  The same order is preserved here, for clarity:
 *
 *      H5IO io;
 *      io.open_file( "fileName.h5" );

 *      io.read_attribute( "attribute1", attribute1 );
 *      io.read_attribute( "attribute2", attribute2 );
 *      io.read_dataset( "data1", data1 );
 *      io.read_dataset( "data2", data2 );
 *
 *      H5IO group = io.open_group( "groupName" );
 *      group.read_attribute( "attribute3", attribute3 );
 *      group.read_dataset( "data3", data3 );
 *
 *      io.close_file();
 *
 */
class H5IO
{

public:
  H5IO();
  ~H5IO();

  void create_file(const std::string& name, int version = 1);
  void open_file(const std::string& name);
  void close_file();

  H5IO create_group(const std::string& name);
  H5IO open_group(const std::string& name);

  unsigned int num_attributes();
  int file_version() const { return fileVersion_; }

  void write_attribute(const std::string& name, int value);
  void write_attribute(const std::string& name, unsigned int value);
  void write_attribute(const std::string& name, double value);
  void write_attribute(const std::string& name, const std::string& value);

  void write_attribute(const std::string& name, const std::vector<int>& value);
  void write_attribute(
    const std::string& name, const std::vector<unsigned int>& value);
  void
  write_attribute(const std::string& name, const std::vector<double>& value);
  void write_attribute(
    const std::string& name, const std::vector<std::string>& value);

  bool has_attribute(const std::string& name);

  void read_attribute(const std::string& name, int& value);
  void read_attribute(const std::string& name, unsigned int& value);
  void read_attribute(const std::string& name, double& value);
  void read_attribute(const std::string& name, std::string& value);
  void
  read_attribute(unsigned int index, std::string& name, std::string& value);

  void read_attribute(const std::string& name, std::vector<int>& value);
  void
  read_attribute(const std::string& name, std::vector<unsigned int>& value);
  void read_attribute(const std::string& name, std::vector<double>& value);
  void read_attribute(const std::string& name, std::vector<std::string>& value);

  void write_dataset(const std::string& name, const std::vector<double>& value);

  void read_dataset(const std::string& name, std::vector<double>& value);

private:
  void h5io_create_group(const std::string& name);
  void h5io_open_group(const std::string& name);
  void h5io_open_group();
  void h5io_close_group();
  hid_t h5io_create_scalar();
  hid_t h5io_create_1D_array(unsigned int size);
  hid_t h5io_create_attribute(const std::string& name, hid_t type, hid_t space);
  hid_t h5io_create_dataset(const std::string& name, hid_t type, hid_t space);

  std::string fileName_;
  std::string groupName_;
  hid_t file_;
  hid_t group_;
  int fileVersion_;
};

} // namespace nalu
} // namespace sierra

#endif
