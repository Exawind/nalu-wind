#ifndef LU_H
#define LU_H

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {

/**
 *  @class LU
 *  @author James C. Sutherland
 *  @date   November, 2005
 *
 *  @brief Supports LU-decompositon for a matrix.
 */
class LU{
 public:
  LU( const int dim, const int bandwidth );
  ~LU();

  // Writable element access
  inline double & operator()( int row, int col ) {
    ThrowRequire( row <= dim_ );
    ThrowRequire( col <= dim_ );
    isReady_ = false;
    return AA_(row,col);
  };

  // Read-only element access
  inline double operator()( int row, int col ) const {
    ThrowRequire( row <= dim_ );
    ThrowRequire( col <= dim_ );
    return AA_(row,col);
  };

  // Read-only element access that works.  Compiler refuses to use
  // the overloaded const operator() when possible for some reason.
  inline double value( int row, int col ) const {
    ThrowRequire( row <= dim_ );
    ThrowRequire( col <= dim_ );
    return AA_(row,col);
  };

  // perform the LU-factorization
  void decompose();

  // perform back-substitution given the rhs.
  // Over-writes the rhs with the solution vector.
  void back_subs( double* rhs );

  void dump();

 private:

  LU();  // no default constructor.

  class SparseMatrix{

  public:
    SparseMatrix( const int dim, const int bandwidth );
    ~SparseMatrix();

    // Writable element access
    inline double & operator()( int row, int col ) {
      return AA_[row][col];
    };

    // Read-only element access
    inline double operator()( int row, int col ) const {
      return AA_[row][col];
    };

    void dump();

  private:
    SparseMatrix();

    double **AA_;
    const int dim_, band_;
  };

  const int dim_;
  bool isReady_;
  SparseMatrix AA_;

};

} // end nalu namespace
} // end sierra namespace

#endif
