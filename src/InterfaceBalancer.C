// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <InterfaceBalancer.h>
#include <NaluEnv.h>

#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/EntityKey.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/environment/Env.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/parallel/ParallelVectorConcat.hpp>

namespace sierra{
namespace nalu {

InterfaceBalancer::InterfaceBalancer(const stk::mesh::MetaData& meta,
                                     stk::mesh::BulkData& bulk) :
                            metaData_(meta), bulkData_(bulk) /*, tolerance_(0.01) */ {}

void
InterfaceBalancer::balance_node_entities(const double targetLoadBalance,
                                         const int maxIterations) {


  if(bulkData_.parallel_size() == 1) return;

  for(int bal = 0; bal < maxIterations; ++bal) {

    int numLocallyOwnedNodes = 0;
    double loadFactor = 0;

    getGlobalLoadImbalance (loadFactor, numLocallyOwnedNodes);

    //const double criterion = std::abs(loadFactor-targetLoadBalance)/targetLoadBalance;
    //const bool converged =  criterion <= tolerance_;
    const bool converged = loadFactor <= targetLoadBalance;
    NaluEnv::self().naluOutputP0() << "balance_nodes: loadFactor = " << loadFactor << " criterion = " << targetLoadBalance
        << " converged = " << converged << std::endl;

    if(converged) return;

    std::set<int> neighborProcessors;
    std::map<stk::mesh::Entity, std::vector<int> > interfaceNodesAndProcessors;

    getInterfaceDescription ( neighborProcessors,
                              interfaceNodesAndProcessors);


    std::map<int, int> numLocallyOwnedByRank;
    exchangeLocalSizes (neighborProcessors, numLocallyOwnedNodes, numLocallyOwnedByRank);


    changeOwnersOfNodes (interfaceNodesAndProcessors, numLocallyOwnedByRank,
                         numLocallyOwnedNodes);
  }
}

void
InterfaceBalancer::getInterfaceDescription (
    std::set<int>& neighborProcessors,
    std::map<stk::mesh::Entity, std::vector<int> >& interfaceNodesAndProcessors) {

  stk::mesh::Selector sharedSelector (metaData_.globally_shared_part ());
  const stk::mesh::BucketVector& buckets = bulkData_.get_buckets (stk::topology::NODE_RANK, sharedSelector);

  int numInterfaceNodesOwned = 0;

  for (auto ib = buckets.begin (); ib != buckets.end (); ++ib) {

    const stk::mesh::Bucket& b = **ib;
    const size_t nnodes = b.size ();
    const bool isOwned = b.owned();
    for (size_t n = 0; n < nnodes; ++n) {
      stk::mesh::Entity node = b[n];
      std::vector<int> shared_procs;

      bulkData_.comm_shared_procs (bulkData_.entity_key (node), shared_procs);
      neighborProcessors.insert (shared_procs.begin (), shared_procs.end ());
      std::pair<stk::mesh::Entity, std::vector<int> > item (node, shared_procs);
      interfaceNodesAndProcessors.insert (item);
      if(isOwned) {
        numInterfaceNodesOwned++;
      }
    }
  }
}

void InterfaceBalancer::getGlobalLoadImbalance (double &loadFactor,
                                                int& numLocallyOwnedNodes) {

  std::vector<int> numLocVec (1);
  std::vector<int> numLocVecGlob (NaluEnv::self().parallel_size());

  stk::mesh::Selector localSelector = metaData_.locally_owned_part ();
  {
    stk::mesh::EntityVector ownedNodes;
    bulkData_.get_entities (stk::topology::NODE_RANK, localSelector,
                            ownedNodes);
    numLocallyOwnedNodes = ownedNodes.size ();
    int maxLocallyOwned = 0;
    int minLocallyOwned = 0;
    stk::all_reduce_max (NaluEnv::self().parallel_comm(), &numLocallyOwnedNodes,
                         &maxLocallyOwned, 1);
    stk::all_reduce_min (NaluEnv::self().parallel_comm(), &numLocallyOwnedNodes,
                         &minLocallyOwned, 1);
    loadFactor = double (maxLocallyOwned) / double (minLocallyOwned);

    numLocVec[0] = numLocallyOwnedNodes;
    stk::parallel_vector_concat(NaluEnv::self().parallel_comm(), numLocVec, numLocVecGlob);

    for (int rank=0; rank < NaluEnv::self().parallel_size(); ++rank) {
      NaluEnv::self().naluOutputP0() << "Processor " << rank << " : Locally Owned Nodes = " << numLocVecGlob[rank] << "\n";
    }
    NaluEnv::self().naluOutputP0() << "Max locally owned nodes = " << maxLocallyOwned << "\n";
    NaluEnv::self().naluOutputP0() << "Min locally owned nodes = " << minLocallyOwned << "\n\n";
  }
}

void
InterfaceBalancer::exchangeLocalSizes (const std::set<int>& neighborProcessors,
                                       int& numLocallyOwnedNodes, std::map<int, int> &numLocallyOwnedByRank) {
  int numCommunications = neighborProcessors.size ();
  std::vector<int> recvBuffer (numCommunications);
  std::vector<MPI_Request> receiveRequests (numCommunications);
  std::vector<MPI_Request> sendRequests (numCommunications);
  int bufferCounter = 0;
  for (int p : neighborProcessors) {

    MPI_Irecv(&recvBuffer[bufferCounter], 1, MPI_INT, p,
              MPI_ANY_TAG, NaluEnv::self().parallel_comm(), &receiveRequests[bufferCounter]);
    ++bufferCounter;
  }
  bufferCounter = 0;
  for (int p : neighborProcessors) {

    MPI_Isend (&numLocallyOwnedNodes, 1, MPI_INT, p, 0, NaluEnv::self().parallel_comm(),
               &sendRequests[bufferCounter]);
    ++bufferCounter;
  }
  std::vector<MPI_Status> receiveStati (receiveRequests.size ());
  MPI_Waitall (receiveRequests.size (), &receiveRequests[0], &receiveStati[0]);
  std::vector<MPI_Status> sendStati (sendRequests.size ());
  MPI_Waitall (sendRequests.size (), &sendRequests[0], &sendStati[0]);

  int i = 0;
  for (int p : neighborProcessors) {
    int n = recvBuffer[i];
    numLocallyOwnedByRank.insert (std::pair<int, int> (p, n));
    ++i;
  }

}

void
InterfaceBalancer::changeOwnersOfNodes (
    const std::map<stk::mesh::Entity, std::vector<int> >& interfaceNodesAndProcessors,
    std::map<int, int>& numLocallyOwnedByRank,
    int numLocallyOwnedNodes) {

  int myRank = bulkData_.parallel_rank ();

  stk::mesh::EntityProcVec nodesToMove;

  for (const auto & nodeProcs : interfaceNodesAndProcessors) {

    stk::mesh::Entity node = nodeProcs.first;
    const bool isOwned = bulkData_.bucket (node).owned();
    if (!isOwned) continue;

    const auto & procs = nodeProcs.second;
    double maxLoad = 1;
    int destination = myRank;
    for (auto p : procs) {

      double numOwnedByP = numLocallyOwnedByRank[p];
      double load = double (numLocallyOwnedNodes) / numOwnedByP;
      if (load >  maxLoad) {  //could test for max rank
        maxLoad = load;
        destination = p;
      }
    }
    if (destination != myRank) {
      std::pair<stk::mesh::Entity, int> item (node, destination);
      nodesToMove.push_back (item);
      numLocallyOwnedNodes--;
      numLocallyOwnedByRank[destination] += 1;
    }

  }

  bulkData_.change_entity_owner(nodesToMove);
}


}
}


