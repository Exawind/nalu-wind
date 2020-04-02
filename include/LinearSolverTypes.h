// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



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

  using Scalar        = Tpetra::Details::DefaultTypes::scalar_type;
  using GlobalOrdinal =  GraphTypes::GlobalOrdinal;
  using LocalOrdinal = GraphTypes::LocalOrdinal;

  using RowLengths        = GraphTypes::RowLengths;
  using DeviceRowLengths  = GraphTypes::DeviceRowLengths;
  using HostRowLengths    = GraphTypes::HostRowLengths;
  using Node              = GraphTypes::Node;
  using Graph             = GraphTypes::Graph;
  using LocalGraph        = GraphTypes::LocalGraph;
  using Comm              = GraphTypes::Comm;
  using Export            = GraphTypes::Export;
  using Import            = GraphTypes::Import;
  using Map               = GraphTypes::Map;
  using MultiVector       = Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>;
  using OneDVector        = Teuchos::ArrayRCP<Scalar >;
  using ConstOneDVector   = Teuchos::ArrayRCP<const Scalar >;
  using LocalVector       = MultiVector::dual_view_type::t_host;
  using Matrix            = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using LocalMatrix       = Matrix::local_matrix_type;
  using Operator          = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using MultiVectorTraits = Belos::MultiVecTraits<Scalar, MultiVector>;
  using OperatorTraits    = Belos::OperatorTraits<Scalar,MultiVector, Operator>;
  using LinearProblem     = Belos::LinearProblem<Scalar, MultiVector, Operator>;
  using SolverManager     = Belos::SolverManager<Scalar, MultiVector, Operator>;
  using SolverFactory     = Belos::TpetraSolverFactory<Scalar, MultiVector, Operator>;
  using Preconditioner    = Ifpack2::Preconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  using EntityToLIDView = Kokkos::View<LocalOrdinal*,Kokkos::LayoutRight,LinSysMemSpace>;
};


} // namespace nalu
} // namespace Sierra

#endif
