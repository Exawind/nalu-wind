/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef CrsGraphTypes_h
#define CrsGraphTypes_h

#include <KokkosInterface.h>
#include <Tpetra_CrsGraph.hpp>

// Forward declare templates
namespace Teuchos {

template <typename T>
class ArrayRCP;

template <typename T>
class MpiComm;

class ParameterList;

}

namespace sierra{
namespace nalu{

namespace GraphTypes {

typedef long   GlobalOrdinal; // MUST be signed
typedef int    LocalOrdinal;  // MUST be signed

typedef Kokkos::DualView<size_t*, DeviceSpace>                             RowLengths;
typedef RowLengths::t_dev                                                  DeviceRowLengths;
typedef RowLengths::t_host                                                 HostRowLengths;
typedef Tpetra::Map<LocalOrdinal, GlobalOrdinal>::node_type                Node;
typedef Tpetra::CrsGraph< LocalOrdinal, GlobalOrdinal, Node>               Graph;
typedef typename Graph::local_graph_type                                   LocalGraph;
typedef Teuchos::MpiComm<int>                                              Comm;
typedef Tpetra::Export< LocalOrdinal, GlobalOrdinal, Node >                Export;
typedef Tpetra::Import< LocalOrdinal, GlobalOrdinal, Node >                Import;
typedef Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node>                       Map;

//using EntityToLIDView = Kokkos::View<LocalOrdinal*,Kokkos::LayoutRight,LinSysMemSpace>;
}


} // namespace nalu
} // namespace Sierra

#endif
