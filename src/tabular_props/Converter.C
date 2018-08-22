#include <tabular_props/Converter.h>
#include <tabular_props/H5IO.h>
#include <tabular_props/Functions.h>
#include <tabular_props/HDF5Table.h>

#include <stk_util/util/ReportHandler.hpp>

#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace sierra {
namespace nalu {

//============================================================================
Converter *
ConverterFactory::create( const std::string & converterType ) const
{
  Converter * converter = NULL;
  if ( converterType == "NameConverter" ) {
    converter = new NameConverter();
  }
  else if ( converterType == "ChiConverter" ||
            converterType == "SLFMChiConverter" /* old name */ ) {
    converter = new ChiConverter();
  }
  else if ( converterType == "DeltaChiConverter" ) {
    converter = new DeltaChiConverter();
  }
  else if ( converterType == "GammaConverter" ||
            converterType == "SLFMGammaConverter" /* old name */ ) {
    converter = new GammaConverter();
  }
  else if ( converterType == "DeltaGammaConverter" ) {
    converter = new DeltaGammaConverter();
  }
  else if ( converterType == "HStarConverter" ||
            converterType == "SLFMHStarConverter" /* old name */ ) {
    const bool isDelta = false;
    converter = new HStarConverter( isDelta );
  }
  else if ( converterType == "DeltaHStarConverter" ) {
    const bool isDelta = true;
    converter = new HStarConverter( isDelta );
  }
  else {
    throw std::runtime_error("Unrecognized Converter type '" + converterType + "'.");
  }

  return converter;
}

//============================================================================
Converter::Converter( const std::string & converterType )
  : dimension_( 0 ),
    converterType_( converterType )
{ }
//----------------------------------------------------------------------------
void
Converter::print_summary() const
{
  // Get the width of the various columns
  //
  const unsigned int type_width = converterType_.size();
  const unsigned int output_width = name_.size();
  unsigned int input_width = 0;
  for ( unsigned int i = 0; i < inputNames_.size(); ++i ) {
    // Get the optimal width for this field.  (Note that std::max() on
    // Sun does not support unsigned int types.)
    input_width = (inputNames_[i].size() > input_width) ?
                   inputNames_[i].size() : input_width;
  }

  // Print the column headers
  std::cout << std::endl;
  std::cout << std::setw(type_width) << "Type" << " | "
            << std::setw(input_width) << "Inputs" << " | "
            << std::setw(output_width) << "Output" << " |" << std::endl;

  // Print a horizontal line
  //
  unsigned int total_width = type_width + 2 +
                             input_width + 3 +
                             output_width + 3;  // Content + margins/dividers
  for ( unsigned int i = 0; i < total_width; ++i ) {
    std::cout << "-";
  }
  std::cout << std::endl;

  // Print the first row of the table, which will have an entry for all columns
  //
  std::cout << std::setw(type_width) << converterType_ << " | "
            << std::setw(input_width) << inputNames_[0] << " | "
            << std::setw(output_width) << name_ << " |" << std::endl;

  // Print the remaining columns, which will only contain inputs.
  //
  for ( unsigned int i = 1; i < inputNames_.size(); ++i ) {
    std::cout << std::setw(type_width) << " " << " | "
              << std::setw(input_width) << inputNames_[i] << " | "
              << std::setw(output_width) << " " << " |" << std::endl;
  }
}
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void
Converter::read_hdf5( H5IO & io )
{
  //
  // Read the base data describing this converter variable.  Specializations
  // of this class must call this base class implementation in addition to
  // handling anything extra that they provide.
  //

  //
  // Double-check that we are reading a converter of the correct type
  //
  std::string typeInFile;
  io.read_attribute( "ConverterType", typeInFile );

  if ( typeInFile != converterType_ ) {
    std::ostringstream errmsg;
    errmsg
      << "ERROR: Attempted to read a Converter of type '" << typeInFile
        << "'" << std::endl
      << "       into a Converter of type '" << converterType_ << "'."
        << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  io.read_attribute( "name", name_ );
  io.read_attribute( "Dimension", dimension_ );
  io.read_attribute( "InputNames", inputNames_ );
}

//============================================================================
NameConverter::NameConverter()
  : Converter( "NameConverter" )
{
  // This default constructor is intended to be used when reading data in
  // from an HDF5 file.
}
//----------------------------------------------------------------------------
NameConverter::NameConverter( const std::string & outputName,
                              const std::string & inputName )
  : Converter( "NameConverter" )
{
  name_ = outputName;
  inputNames_.push_back( inputName );
  dimension_ = 1;
}
//----------------------------------------------------------------------------
double
NameConverter::query( const std::vector<double>  &inputs ) const
{
  // Just pass the single input variable on through, without modification
  return inputs[0];
}

//============================================================================
ChiConverter::ChiConverter()
  : Converter( "ChiConverter" ),
    zStoich_( -1.0 ),
    fChiMeanTable_( NULL )
{
  // This default constructor is intended to be used when reading data in
  // from an HDF5 file.
}
//----------------------------------------------------------------------------
ChiConverter::~ChiConverter()
{
  delete fChiMeanTable_;
}
//----------------------------------------------------------------------------
double
ChiConverter::query( const std::vector<double>  &inputs ) const
{
  // Note that the inputs will always contain these variables:
  //
  //   inputs[0] = ZMean
  //   inputs[1] = ZScaledVarianceMean
  //   inputs[2] = ChiMean
  //
  // The query() method on the table requires the first two variables in
  // the same order, so just pass it the inputs pointer and it will grab
  // what it needs.
  //
  const double fChiMean = std::max( fChiMeanTable_->query(inputs),
                                    1.e-12 ); // Clip to avoid div/0
  return inputs[2] * F_chi(zStoich_) / fChiMean;
}
//----------------------------------------------------------------------------
void
ChiConverter::read_hdf5( H5IO & io )
{
  // Make sure that the base class pieces get read
  Converter::read_hdf5( io );

  // Read the extra information required by this converter
  io.read_attribute( "ZStoich", zStoich_ );

  HDF5Table *table = new HDF5Table;
  H5IO tableIO = io.open_group( "FChiMeanTable" );
  table->read_hdf5_table( tableIO );
  fChiMeanTable_ = table;  // Assign to const data member

}

//============================================================================
DeltaChiConverter::DeltaChiConverter()
  : Converter( "DeltaChiConverter" ),
    zStoich_( -1.0 )
{
  // This default constructor is intended to be used when reading data in
  // from an HDF5 file.
}
//----------------------------------------------------------------------------
DeltaChiConverter::~DeltaChiConverter()
{
}
//----------------------------------------------------------------------------
double
DeltaChiConverter::query( const std::vector<double>  &inputs ) const
{
  // Note that the inputs will always contain these variables:
  //
  //   inputs[0] = ZMean
  //   inputs[1] = ChiMean
  //
  return inputs[1] * F_chi(zStoich_) / std::max( F_chi(inputs[0]), 1.e-12 );
}
//----------------------------------------------------------------------------
void
DeltaChiConverter::read_hdf5( H5IO & io )
{
  // Make sure that the base class pieces get read
  Converter::read_hdf5( io );

  // Read the extra information required by this converter
  io.read_attribute( "ZStoich", zStoich_ );
}

//============================================================================
GammaConverter::GammaConverter()
  : Converter( "GammaConverter" ),
    numTableInputs_( 0 ),
    fGammaMeanTable_( NULL )
{
  // This default constructor is intended to be used when reading data in
  // from an HDF5 file.
}
//----------------------------------------------------------------------------
GammaConverter::~GammaConverter()
{
  delete fGammaMeanTable_;
}
//----------------------------------------------------------------------------
double
GammaConverter::query( const std::vector<double>  &inputs ) const
{
  // Note that the inputs will always contain these variables:
  //
  //   inputs[0]                = ZMean
  //   inputs[1]                = ZScaledVarianceMean
  //   inputs[2]                = First delta PDF-convoluted Z
  //   ...
  //   inputs[numTableInputs-1] = Last delta PDF-convoluted Z
  //   inputs[numTableInputs]   = GammaMean
  //
  // The query() method on the table requires all but the last input in the
  // same order, so just pass it the inputs pointer and it will grab what
  // it needs.
  //
  const double gammaMean = inputs[numTableInputs_];
  const double fGammaMean = std::max( fGammaMeanTable_->query(inputs),
                                      1.e-12 ); // Clip to avoid div/0
  return gammaMean / fGammaMean;
}
//----------------------------------------------------------------------------
void
GammaConverter::read_hdf5( H5IO & io )
{
  // Make sure that the base class pieces get read
  Converter::read_hdf5( io );

  HDF5Table *table = new HDF5Table ;
  H5IO tableIO = io.open_group( "FGammaMeanTable" );
  table->read_hdf5_table( tableIO );
  fGammaMeanTable_ = table;  // Assign to const data member

  numTableInputs_ = fGammaMeanTable_->input_names().size();
}

//============================================================================
DeltaGammaConverter::DeltaGammaConverter()
  : Converter( "DeltaGammaConverter" ),
    numMixFrac_( 0 )
{
  // This default constructor is intended to be used when reading data in
  // from an HDF5 file.
}
//----------------------------------------------------------------------------
DeltaGammaConverter::~DeltaGammaConverter()
{
}
//----------------------------------------------------------------------------
double
DeltaGammaConverter::query( const std::vector<double>  &inputs ) const
{
  // Note that the inputs will always contain these variables:
  //
  //   inputs[0]             = ZMean[0]
  //   ...
  //   inputs[numMixFrac_-1] = ZMean[numMixFrac_-1]
  //   inputs[numMixFrac_]   = GammaMean
  //
  for ( unsigned int i = 0; i < numMixFrac_; ++i ) {
    zMean_[i] = inputs[i]; 
  }

  const double gammaMean = inputs[numMixFrac_];
  const double fGamma = std::max( F_gamma( zMean_, zStoich_, gammaMaxStoich_ ),
                                  1.e-12 );

  return gammaMean / fGamma;
}
//----------------------------------------------------------------------------
void
DeltaGammaConverter::read_hdf5( H5IO & io )
{
  // Make sure that the base class pieces get read
  Converter::read_hdf5( io );

  // Read the extra information required by this converter

  // Multiple mixture fraction support
  io.read_attribute( "NumMixFrac", numMixFrac_ );

  unsigned int numberZStoich = 0;
  io.read_attribute( "NumberZStoich", numberZStoich );
  for ( unsigned int i = 0; i < numberZStoich; ++i ) {
    std::ostringstream zName;
    zName << "ZStoich_" << std::setw(1) << i;

    std::vector<double> zStoich;
    io.read_attribute( zName.str(), zStoich );
    zStoich_.push_back( zStoich );
  }

  io.read_attribute( "GammaMaxStoich", gammaMaxStoich_ );

  zMean_.resize( numMixFrac_ );  // Buffer space
}

//============================================================================
HStarConverter::HStarConverter( const bool isDelta )
  : Converter( (isDelta) ? "DeltaHStarConverter" : "HStarConverter" ),
    numMixFrac_( 0 ),
    hStar_ref_min_( 0.0 )
{
  // This default constructor is intended to be used when reading data in
  // from an HDF5 file.
}
//----------------------------------------------------------------------------
HStarConverter::~HStarConverter()
{
}
//----------------------------------------------------------------------------
double
HStarConverter::query( const std::vector<double>  &inputs ) const
{
  //
  // Note that the inputs will always contain these (filtered) variables:
  //
  //   inputs[0]             = ZMean[0]
  //   ...
  //   inputs[numMixFrac_-1] = ZMean[numMixFrac_-1]
  //   inputs[numMixFrac_]   = hStarMean
  //
  // Convert these input variables to the reference h* value for
  // table lookups with the equation:
  //
  //     h*_ref = h*_ref,min + (h* - h*_min,Z) / a_Z
  //
  // where:
  //
  //     h* = Local filtered "conserved enthalpy" from Fuego simulation
  //     Z_n  = Local filtered mixture fraction vector from Fuego simulation
  //     h*_ref = Reference "conserved enthalpy" for table lookup
  //     Z_ref,n = Reference mixture fraction vector (at mixture fraction
  //               realizable region centroid)
  //     h*_stream,min,n = Minimum pure stream h*
  //     h*_stream,max,n = Maximum pure stream h*
  //     h*_ref,min = Minimum pure stream h*, weighted to the reference
  //                  mixture fraction (Z_ref,n):
  //                    h*_ref,min = \sum_{n=1}^N (h*_stream,min,n * Z_ref,n)
  //     h*_ref,max = Minimum pure stream h*, weighted to the reference
  //                  mixture fraction (Z_ref,n):
  //                    h*_ref,max = \sum_{n=1}^N (h*_stream,max,n * Z_ref,n)
  //     h*_min,Z = Minimum pure stream h*, weighted to the local filtered
  //                mixture fraction location (Z_n)
  //                  h*_min,Z = \sum_{n=1}^N (h*_stream,min,n * Z_n)
  //     a_Z = Local mixture-weighted stream variation proportionality
  //           constant, a_Z = \sum_{n=1}^N (a_n * Z_n), where:
  //             a_n = (h*_stream,max,n - h*_stream,min,n)/
  //                   (h*_ref,max - h*_ref,min)
  //

  augmented_mixfrac( inputs, zAug_ );
  const double hStar = inputs[numMixFrac_];

  const double a_Z         = mixture_property( zAug_, a_ );
  const double hStar_min_Z = mixture_property( zAug_, hStar_stream_min_ );

  double hStar_ref = 0.0;

  if ( std::abs( a_Z ) > 1.e-16 ) {
    hStar_ref = hStar_ref_min_ + (hStar - hStar_min_Z) / a_Z; 
  }
  else {
    //
    // Return just the baselin h*_ref value if we have a divide-by-zero
    // opportunity.  This should only occur if we're at a location where
    // there is no variation in h*, so that returning the baseline h*_ref
    // value is appropriate.
    //
    hStar_ref = hStar_ref_min_;
  }

  return hStar_ref;
}
//----------------------------------------------------------------------------
void
HStarConverter::read_hdf5( H5IO & io )
{
  // Make sure that the base class pieces get read
  Converter::read_hdf5( io );

  // Read the extra information required by this converter

  if ( io.file_version() >= 4 ) {
    io.read_attribute( "NumMixFrac", numMixFrac_ );
    io.read_attribute( "HStar_ref_min", hStar_ref_min_ );
    io.read_attribute( "HStar_stream_min", hStar_stream_min_ );
    io.read_attribute( "A", a_ );
  }
  else {
    // Single mixture fraction support
    numMixFrac_ = 1;
    io.read_attribute( "HStarRef_base", hStar_ref_min_ );

    double hStarZ0_base = 0.0;
    double hStarZ1_base = 0.0;
    io.read_attribute( "HStarZ0_base", hStarZ0_base );
    io.read_attribute( "HStarZ1_base", hStarZ1_base );
    hStar_stream_min_.push_back( hStarZ0_base );
    hStar_stream_min_.push_back( hStarZ1_base );

    double a = 0.0;
    double b = 0.0;
    io.read_attribute( "a", a );
    io.read_attribute( "b", b );
    a_.push_back( a );
    a_.push_back( b );
  }

  // Resize the buffer space
  zAug_.resize( numMixFrac_ + 1 );
}
//----------------------------------------------------------------------------
void
HStarConverter::augmented_mixfrac( const std::vector<double> &mixFrac,
                                   std::vector<double> & mixFracAug ) const
{
//  ThrowRequire( mixFrac.size() == streamProp.size()-1 );
// resize mixFracAug

  // Clip the mixture fraction vector to unity and augment it with the last
  // "implied" component so that its dimension matches the number of streams
  double lastMixFrac = 1.0;
  for ( unsigned int i = 0; i < numMixFrac_; ++i ) {
    mixFracAug[i] = std::min( mixFrac[i], lastMixFrac );
    lastMixFrac -= mixFracAug[i];
  }
  mixFracAug[numMixFrac_] = lastMixFrac;
}
//----------------------------------------------------------------------------
double
HStarConverter::mixture_property( const std::vector<double> & mixFracAug,
                                  const std::vector<double> & streamProp ) const
{
//  ThrowRequire( mixFracAug.size() == streamProp.size() );

  //
  // Compute the mixture-weighted material property as a linear weighting
  // of the pure-stream properties.
  //
  // phi = Z[1]*phi(Z_1) + Z[2]*phi(Z_2) + ... + Z[N]*phi(Z_N)
  //

  double prop = 0.0;
  for ( unsigned int i = 0; i < mixFracAug.size(); ++i ) {
    prop += mixFracAug[i] * streamProp[i];
  }

  return prop;
}


} // end nalu namespace
} // end sierra namespace

//============================================================================

