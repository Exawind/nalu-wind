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

#include <KokkosInterface.h>
#include <Tpetra_Details_DefaultTypes.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

// Forward declare templates
namespace Teuchos {

template <typename T>
class ArrayRCP;

template <typename T>
class MpiComm;

class ParameterList;

} // namespace Teuchos

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
} // namespace Belos

namespace Ifpack2 {

template <
  typename Scalar,
  typename LocalOrdinal,
  typename GlobalOrdinal,
  typename Node>
class Preconditioner;
}

namespace sierra {
namespace nalu {

class TpetraLinearSolver;

struct LinSys
{

  using Scalar = Tpetra::Details::DefaultTypes::scalar_type;
  using GlobalOrdinal = Tpetra::Details::DefaultTypes::global_ordinal_type;
  using LocalOrdinal = Tpetra::Details::DefaultTypes::local_ordinal_type;

  using RowLengths = Kokkos::DualView<size_t*, DeviceSpace>;
  using DeviceRowLengths = RowLengths::t_dev;
  using HostRowLengths = RowLengths::t_host;
  using Node = Tpetra::Map<LocalOrdinal, GlobalOrdinal>::node_type;
  using Graph = Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>;
  using LocalGraph = typename Graph::local_graph_device_type;
  using LocalGraphHost = typename Graph::local_graph_host_type;
  using Comm = Teuchos::MpiComm<int>;
  using Export = Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>;
  using Import = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
  using Map = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  using MultiVector =
    Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using OneDVector = Teuchos::ArrayRCP<Scalar>;
  using ConstOneDVector = Teuchos::ArrayRCP<const Scalar>;
  using LocalVector = MultiVector::dual_view_type::t_dev;
  using Matrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using LocalMatrix = Matrix::local_matrix_device_type;
  using LocalMatrixHost = Matrix::local_matrix_host_type;
  using LocalIndicesHost = Matrix::local_inds_host_view_type;
  using LocalValuesHost = Matrix::values_host_view_type;
  using Operator = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using MultiVectorTraits = Belos::MultiVecTraits<Scalar, MultiVector>;
  using OperatorTraits = Belos::OperatorTraits<Scalar, MultiVector, Operator>;
  using LinearProblem = Belos::LinearProblem<Scalar, MultiVector, Operator>;
  using SolverManager = Belos::SolverManager<Scalar, MultiVector, Operator>;
  using SolverFactory =
    Belos::TpetraSolverFactory<Scalar, MultiVector, Operator>;
  using Preconditioner =
    Ifpack2::Preconditioner<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  using EntityToLIDView =
    Kokkos::View<LocalOrdinal*, Kokkos::LayoutRight, LinSysMemSpace>;
  using EntityToLIDHostView = typename EntityToLIDView::HostMirror;
  using ConstEntityToLIDView =
    Kokkos::View<const LocalOrdinal*, Kokkos::LayoutRight, LinSysMemSpace>;
  using ConstEntityToLIDHostView = typename ConstEntityToLIDView::HostMirror;
};

} // namespace nalu
} // namespace sierra

#endif
