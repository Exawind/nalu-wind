#ifndef CONVERTER_H
#define CONVERTER_H

#include <tabular_props/HDF5Table.h>
#include <vector>
#include <string>

namespace sierra {
namespace nalu {

// Forward declarations
class H5IO;
class Converter;

//============================================================================
/**
 *  @class ConverterFactory
 *  @brief Simple factory to build an empty Converter of the correct type
 *         based on its string name.
 */
class ConverterFactory
{
public:
  ConverterFactory() {};
  ~ConverterFactory() {};

  Converter* create(const std::string& converterType) const;
};

//============================================================================
/**
 *  @class Converter
 *  @brief Provides arbitrary processing of input data before handing off
 *         values to a Table for interpolation.
 *
 *  This abstract base class provides an interface for accepting any number
 *  of input variables inside a Property object, and calculating an arbitrary
 *  new variable for use as an input in the central Table.  Derived classes
 *  provide the conversion machinery in the query() function.
 */
class Converter
{

public:
  explicit Converter(const std::string& converterType);

  virtual ~Converter() {};

  /** Get the name of the variable returned by this Converter */
  const std::string& name() const { return name_; }

  /** Get the list of input variables, in the order that they are to be
   *  provided to the query() call */
  const std::vector<std::string>& input_names() const { return inputNames_; }

  /* Get the number of input variables for this Converter */
  unsigned int dimension() const { return dimension_; }

  /**
   *  Perform the calculation
   *
   *  @param inputs : Array of independent variable values
   *  @result : The resulting computed value
   */
  virtual double query(const std::vector<double>& inputs) const = 0;

  /** Print a summary of this Converter configuration */
  void print_summary() const;

  /** Read this Converter from the provided I/O device */
  virtual void read_hdf5(H5IO& io);

protected:
  // Name of the variable returned by this Converter
  std::string name_;

  // Names of the input variables, in the order they are required
  std::vector<std::string> inputNames_;

  // Number of input variables
  unsigned int dimension_;

private:
  // No copying.  (Assignment not possible with abstract class.)
  Converter(const Converter&);

  const std::string converterType_;
};

//============================================================================
/**
 *  @class NameConverter
 *  @brief Does a whole bunch of nothing other than renaming variables.
 *
 *  This converter simply passes the input variable right through to the
 *  output, without modification.  It is primarily used for testing and
 *  debugging, although it can be used for simply renaming table variables
 *  so that the Property class advertises a different input name.
 */
class NameConverter : public Converter
{

public:
  NameConverter();
  NameConverter(const std::string& outputName, const std::string& inputName);

  virtual double query(const std::vector<double>& inputs) const;

private:
  NameConverter operator=(const NameConverter&); // No assignment
};

//============================================================================
/**
 *  @class ChiConverter
 *  @brief Calculates reference scalar dissipation rate
 *
 *  This converter calculates a reference scalar dissipation rate, given
 *  mean and variance mixture fraction and mean scalar dissipation rate.
 *  This is used by the nonadiabatic flamelet models.
 */
class ChiConverter : public Converter
{

public:
  ChiConverter();
  /* This version is used in Atab_ConverterBuilder.C */
  /*
  ChiConverter( const std::string & outputName,
                const std::string & inputZMeanName,
                const std::string & inputZScaledVarianceMeanName,
                const std::string & inputChiMeanName,
                const double zStoich,
                const HDF5Table * fChiMeanTable );
  */

  virtual ~ChiConverter();

  /**
   *  Do a lookup in the table at the input variable coordinates provided
   *
   *  @param inputs : Array of input variable values
   *  @result : The state variable value interpolated from the table
   */
  virtual double query(const std::vector<double>& inputs) const;

  /** Read this Converter from the provided I/O device */
  virtual void read_hdf5(H5IO& io);

private:
  ChiConverter operator=(const ChiConverter&); // No assignment

  double zStoich_;
  const HDF5Table* fChiMeanTable_;
};

//============================================================================
/**
 *  @class DeltaChiConverter
 *  @brief Calculates reference scalar dissipation rate
 *
 *  This converter calculates a reference scalar dissipation rate for
 *  flow where turbulence/chemistry interactions can be neglected.
 */
class DeltaChiConverter : public Converter
{

public:
  DeltaChiConverter();
  /* This version is used in Atab_ConverterBuilder.C */
  /*
  DeltaChiConverter( const std::string & outputName,
                     const std::string & inputZMeanName,
                     const std::string & inputChiMeanName,
                     const double zStoich );
  */

  virtual ~DeltaChiConverter();

  /**
   *  Do a lookup in the table at the input variable coordinates provided
   *
   *  @param inputs : Array of input variable values
   *  @result : The state variable value interpolated from the table
   */
  virtual double query(const std::vector<double>& inputs) const;

  /** Read this Converter from the provided I/O device */
  virtual void read_hdf5(H5IO& io);

private:
  DeltaChiConverter operator=(const DeltaChiConverter&); // No assignment

  double zStoich_;
};

//============================================================================
/**
 *  @class GammaConverter
 *  @brief Calculates reference heat loss parameter
 *
 *  This converter calculates a reference heat loss parameter, given
 *  mean and variance mixture fraction and mean heat loss parameter.
 *  This is used by the nonadiabatic flamelet models.
 */
class GammaConverter : public Converter
{

public:
  GammaConverter();
  /* This version is used in Atab_ConverterBuilder.C */
  /*
  GammaConverter( const std::string & outputName,
                  const std::vector<std::string> & inputZMeanNames,
                  const std::string & inputZScaledVarianceMeanName,
                  const std::string & inputGammaMeanName,
                  const HDF5Table * fGammaMeanTable );
  */

  virtual ~GammaConverter();

  /**
   *  Do a lookup in the table at the input variable coordinates provided
   *
   *  @param inputs : Array of input variable values
   *  @result : The state variable value interpolated from the table
   */
  virtual double query(const std::vector<double>& inputs) const;

  /** Read this Converter from the provided I/O device */
  virtual void read_hdf5(H5IO& io);

private:
  GammaConverter operator=(const GammaConverter&); // No assignment

  unsigned int numTableInputs_;
  const HDF5Table* fGammaMeanTable_;
};

//============================================================================
/**
 *  @class DeltaGammaConverter
 *  @brief Calculates reference heat loss parameter
 *
 *  This converter calculates a reference heat loss parameter for
 *  flow where turbulence/chemistry interactions can be neglected.
 */
class DeltaGammaConverter : public Converter
{

public:
  DeltaGammaConverter();
  /* This version is used in Atab_ConverterBuilder.C */
  /*
  DeltaGammaConverter( const std::string & outputName,
                       const std::vector<std::string> & inputZMeanName,
                       const std::string & inputGammaMeanName,
                       const std::vector<std::vector<double> > & zStoich,
                       const std::vector<double> & gammaMaxStoich );
  */

  virtual ~DeltaGammaConverter();

  /**
   *  Do a lookup in the table at the input variable coordinates provided
   *
   *  @param inputs : Array of input variable values
   *  @result : The state variable value interpolated from the table
   */
  virtual double query(const std::vector<double>& inputs) const;

  /** Read this Converter from the provided I/O device */
  virtual void read_hdf5(H5IO& io);

private:
  DeltaGammaConverter operator=(const DeltaGammaConverter&); // No assignment

  unsigned int numMixFrac_;
  std::vector<std::vector<double>> zStoich_;
  std::vector<double> gammaMaxStoich_;
  mutable std::vector<double> zMean_;
};

//============================================================================
/**
 *  @class HStarConverter
 *  @brief Calculates reference heat loss parameter
 *
 *  This converter calculates the reference (median) conserved enthalpy
 *  value, given mixture fraction and conserved enthalpy.  This is used
 *  by the nonadiabatic flamelet model for flows where the turbulence/
 *  chemistry interactions may be either neglected (isDelta=true) or
 *  included (isDelta=false; default).
 */
class HStarConverter : public Converter
{

public:
  explicit HStarConverter(const bool isDelta = false);

  /* This version is used in Atab_ConverterBuilder.C */
  /*
  HStarConverter( const std::string & outputName,
                  const std::vector<std::string> & inputZMeanNames,
                  const std::string & inputHStarMeanName,
                  const double hStar_ref_min,
                  const std::vector<double> & hStar_stream_min,
                  const std::vector<double> & a,
                  const bool isDelta = false );
  */

  virtual ~HStarConverter();

  /**
   *  Do a lookup in the table at the input variable coordinates provided
   *
   *  @param inputs : Array of input variable values
   *  @result : The state variable value interpolated from the table
   */
  virtual double query(const std::vector<double>& inputs) const;

  /** Read this Converter from the provided I/O device */
  virtual void read_hdf5(H5IO& io);

private:
  HStarConverter operator=(const HStarConverter&); // No assignment

  void augmented_mixfrac(
    const std::vector<double>& mixFrac, std::vector<double>& mixFracAug) const;

  double mixture_property(
    const std::vector<double>& mixFracAug,
    const std::vector<double>& streamProp) const;

  unsigned int numMixFrac_;
  mutable std::vector<double> zAug_; // Augmented mixture fraction buffer space

  double hStar_ref_min_;
  std::vector<double> hStar_stream_min_;
  std::vector<double> a_;
};

//============================================================================

} // namespace nalu
} // namespace sierra

#endif
