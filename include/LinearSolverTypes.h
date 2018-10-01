/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef LinearSolverTypes_h
#define LinearSolverTypes_h

#include <KokkosInterface.h>
#include <Tpetra_CrsGraph.hpp>
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

typedef long   GlobalOrdinal; // MUST be signed
typedef int    LocalOrdinal;  // MUST be signed
typedef double Scalar;

typedef Kokkos::DualView<size_t*, DeviceSpace>                             RowLengths;
typedef Tpetra::Map<LocalOrdinal, GlobalOrdinal>::node_type                Node;
typedef Tpetra::CrsGraph< LocalOrdinal, GlobalOrdinal, Node>               Graph;
typedef typename Graph::local_graph_type                                   LocalGraph;
typedef Teuchos::MpiComm<int>                                              Comm;
typedef Tpetra::Export< LocalOrdinal, GlobalOrdinal, Node >                Export;
typedef Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node >                Import;
typedef Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node>                       Map;
typedef Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>        MultiVector;
typedef Teuchos::ArrayRCP<Scalar >                                         OneDVector;
typedef Teuchos::ArrayRCP<const Scalar >                                   ConstOneDVector;
typedef Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node>             Vector;
typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>       Matrix;
typedef Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>        Operator;
typedef Belos::MultiVecTraits<Scalar, MultiVector>                         MultiVectorTraits;
typedef Belos::OperatorTraits<Scalar,MultiVector, Operator>                OperatorTraits;
typedef Belos::LinearProblem<Scalar, MultiVector, Operator>                LinearProblem;
typedef Belos::SolverManager<Scalar, MultiVector, Operator>                SolverManager;
typedef Belos::TpetraSolverFactory<Scalar, MultiVector, Operator>          SolverFactory;
typedef Ifpack2::Preconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node> Preconditioner;
};


} // namespace nalu
} // namespace Sierra

#endif
