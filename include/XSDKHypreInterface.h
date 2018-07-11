// @HEADER
// ***********************************************************************
//
//       xSDKTrilinos: Extreme-scale Software Development Kit Package
//                 Copyright (2016) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alicia Klinvex    (amklinv@sandia.gov)
//                    James Willenbring (jmwille@sandia.gov)
//                    Michael Heroux    (maherou@sandia.gov)         
//
// ***********************************************************************
// @HEADER

// This file was copied into Nalu source tree from
// https://github.com/trilinos/xSDKTrilinos. The contents have been modified to
// enable interfacing with Nalu. The original copyright and license have been
// retained.
//

#ifndef XSDKHYPREINTERFACE_H
#define XSDKHYPREINTERFACE_H

#include "FieldTypeDef.h"
#include "Ifpack2_ConfigDefs.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE.h"

#include "Ifpack2_Preconditioner.hpp"
#include "Ifpack2_Condest.hpp"

#include "Teuchos_RCP.hpp"

#include <cmath>

namespace Ifpack2 {

#ifndef HYPRE_ENUMS
#define HYPRE_ENUMS
namespace Hypre {
//! This enumerated type defines the allowed solvers and preconditioners in Hypre. Some can be used as both solver and preconditioner.
enum Hypre_Solver{
    BoomerAMG,
    ParaSails,
    Euclid,
    AMS,
    Hybrid,
    PCG,
    GMRES,
    COGMRES,
    FlexGMRES,
    LGMRES,
    BiCGSTAB
};

//! This enumerated type defines the two options for applying inverse, either solve or apply the preconditioner.
enum Hypre_Chooser{
    Solver,
    Prec
};
} // namespace Hypre
#endif //HYPRE_ENUMS

//! This class is used to help with passing parameters in the SetParameter() function. Use this class to call Hypre's internal parameters.
class FunctionParameter{
  using HypreIntType = sierra::nalu::HypreIntType;

  public:
    //! Single int constructor.
    FunctionParameter(
      Hypre::Hypre_Chooser chooser,
      HypreIntType (*funct_name)(HYPRE_Solver, HypreIntType),
      HypreIntType param1)
      : chooser_(chooser),
        option_(0),
        int_func_(funct_name),
        int_param1_(param1)
    {}

    //! Single double constructor.
    FunctionParameter(
      Hypre::Hypre_Chooser chooser,
      HypreIntType (*funct_name)(HYPRE_Solver, double),
      double param1)
      : chooser_(chooser),
        option_(1),
        double_func_(funct_name),
        double_param1_(param1)
    {}

    //! Single double, single HypreIntType constructor.
    FunctionParameter(
      Hypre::Hypre_Chooser chooser,
      HypreIntType (*funct_name)(HYPRE_Solver, double, HypreIntType),
      double param1,
      HypreIntType param2)
      : chooser_(chooser),
        option_(2),
        double_int_func_(funct_name),
        int_param1_(param2),
        double_param1_(param1)
    {}

    //! Two HypreIntTypes constructor.
    FunctionParameter(
      Hypre::Hypre_Chooser chooser,
      HypreIntType (*funct_name)(HYPRE_Solver, HypreIntType, HypreIntType),
      HypreIntType param1,
      HypreIntType param2)
      : chooser_(chooser),
        option_(3),
        int_int_func_(funct_name),
        int_param1_(param1),
        int_param2_(param2)
    {}

    //! HypreIntType pointer constructor.
    FunctionParameter(
      Hypre::Hypre_Chooser chooser,
      HypreIntType (*funct_name)(HYPRE_Solver, HypreIntType*),
      HypreIntType* param1)
      : chooser_(chooser),
        option_(4),
        int_star_func_(funct_name),
        int_star_param_(param1)
    {}

    //! Double pointer constructor.
    FunctionParameter(
      Hypre::Hypre_Chooser chooser,
      HypreIntType (*funct_name)(HYPRE_Solver, double*),
      double* param1)
      : chooser_(chooser),
        option_(5),
        double_star_func_(funct_name),
        double_star_param_(param1)
    {}

    //! char pointer constructor
    FunctionParameter(
      Hypre::Hypre_Chooser chooser,
      HypreIntType (*funct_name)(HYPRE_Solver, char*),
      char* param1)
      : chooser_(chooser),
        option_(6),
        char_star_func_(funct_name),
        char_star_param_(param1)
    {}

    //! Only method of this class. Calls the function pointer with the passed in HYPRE_Solver
    int CallFunction(HYPRE_Solver solver, HYPRE_Solver precond){
      if(chooser_ == Hypre::Solver){
        if(option_ == 0){
          return int_func_(solver, int_param1_);
        } else if(option_ == 1){
          return double_func_(solver, double_param1_);
        } else if(option_ == 2){
          return double_int_func_(solver, double_param1_, int_param1_);
        } else if (option_ == 3){
          return int_int_func_(solver, int_param1_, int_param2_);
        } else if (option_ == 4){
          return int_star_func_(solver, int_star_param_);
        } else if (option_ == 5) {
          return double_star_func_(solver, double_star_param_);
        } else if (option_ == 6) {
            return char_star_func_(solver, char_star_param_);
        }
      } else {
        if(option_ == 0){
          return int_func_(precond, int_param1_);
        } else if(option_ == 1){
          return double_func_(precond, double_param1_);
        } else if(option_ == 2){
          return double_int_func_(precond, double_param1_, int_param1_);
        } else if(option_ == 3) {
          return int_int_func_(precond, int_param1_, int_param2_);
        } else if(option_ == 4) {
          return int_star_func_(precond, int_star_param_);
        } else if (option_ == 5) {
            return double_star_func_(precond, double_star_param_);
        } else if (option_ == 6) {
            return char_star_func_(solver, char_star_param_);
        }
      }
      // Should never reach here, but quell warning
      return 0;
    }

  private:
    Hypre::Hypre_Chooser chooser_;
    int option_;
    HypreIntType (*int_func_)(HYPRE_Solver, HypreIntType);
    HypreIntType (*double_func_)(HYPRE_Solver, double);
    HypreIntType (*double_int_func_)(HYPRE_Solver, double, HypreIntType);
    HypreIntType (*int_int_func_)(HYPRE_Solver, HypreIntType, HypreIntType);
    HypreIntType (*int_star_func_)(HYPRE_Solver, HypreIntType*);
    HypreIntType (*double_star_func_)(HYPRE_Solver, double*);
    HypreIntType (*char_star_func_)(HYPRE_Solver, char*);
    HypreIntType int_param1_;
    HypreIntType int_param2_;
    double double_param1_;
    HypreIntType *int_star_param_;
    double *double_star_param_;
    char* char_star_param_;
};

} // namespace Ifpack2

#endif /* XSDKHYPREINTERFACE_H */
