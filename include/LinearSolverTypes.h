/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef LinearSolverTypes_h
#define LinearSolverTypes_h

#include <CrsGraphTypes.h>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>


// Forward declare templates
namespace Teuchos {

template <typename T>
class ArrayRCP;

template <typename T>
class MpiComm;

class ParameterList;

}

namespace Belos {

template <typename Scalar, typename MultiVector>
class MultiVecTraits;

template <typename Scalar, typename MultiVector, typename Operator>
class OperatorTraits;

template <typename Scalar, typename MultiVector, typename Operator>
class LinearProblem;

template <typename Scalar, typename MultiVector, typename Operator>
class SolverManager;

template <typename Scalar, typename MultiVector, typename Operator>
class TpetraSolverFactory;
}

namespace Ifpack2 {

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
class Preconditioner;

}

namespace sierra{
namespace nalu{

class TpetraLinearSolver;

struct LinSys {

typedef GraphTypes::GlobalOrdinal   GlobalOrdinal; // MUST be signed
typedef GraphTypes::LocalOrdinal    LocalOrdinal;  // MUST be signed
typedef double Scalar;

typedef GraphTypes::RowLengths                             RowLengths;
typedef GraphTypes::DeviceRowLengths                       DeviceRowLengths;
typedef GraphTypes::HostRowLengths                         HostRowLengths;
typedef GraphTypes::Node                                   Node;
typedef GraphTypes::Graph                                  Graph;
typedef GraphTypes::LocalGraph                             LocalGraph;
typedef GraphTypes::Comm                                   Comm;
typedef GraphTypes::Export                                 Export;
typedef GraphTypes::Import                                 Import;
typedef GraphTypes::Map                                    Map;

typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>        MultiVector;
typedef Teuchos::ArrayRCP<Scalar >                                         OneDVector;
typedef Teuchos::ArrayRCP<const Scalar >                                   ConstOneDVector;
typedef MultiVector::dual_view_type::t_host                                LocalVector;
typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>       Matrix;
typedef Matrix::local_matrix_type                                          LocalMatrix;
typedef Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>        Operator;
typedef Belos::MultiVecTraits<Scalar, MultiVector>                         MultiVectorTraits;
typedef Belos::OperatorTraits<Scalar,MultiVector, Operator>                OperatorTraits;
typedef Belos::LinearProblem<Scalar, MultiVector, Operator>                LinearProblem;
typedef Belos::SolverManager<Scalar, MultiVector, Operator>                SolverManager;
typedef Belos::TpetraSolverFactory<Scalar, MultiVector, Operator>          SolverFactory;
typedef Ifpack2::Preconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node> Preconditioner;

using EntityToLIDView = Kokkos::View<LocalOrdinal*,Kokkos::LayoutRight,LinSysMemSpace>;
};


} // namespace nalu
} // namespace Sierra

#endif
