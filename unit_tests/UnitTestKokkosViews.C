#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "UnitTestUtils.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

#include <master_element/MasterElementFactory.h>
#include <stk_util/parallel/Parallel.hpp>
#include <Kokkos_Core.hpp>

#include "UnitTestKokkosUtils.h"

namespace {

#ifndef KOKKOS_ENABLE_CUDA
//following tests can't run on cuda due to variety of reasons, including
//use of std::vectors, use of MasterElement functions (defined for host), etc.

void find_max_nodes_and_ips(const stk::mesh::BucketVector& buckets,
                            int& maxNodesPerElement, int& maxScsIp)
{
  maxNodesPerElement = 0;
  maxScsIp = 0;
  size_t numEntities = 0;
  for(const stk::mesh::Bucket* bptr : buckets) {
    stk::topology topo = bptr->topology();
    maxNodesPerElement = std::max(maxNodesPerElement, (int)topo.num_nodes());
    sierra::nalu::MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);
    maxScsIp = std::max(maxScsIp, meSCS->num_integration_points());
    numEntities += bptr->size();
  }
  std::cout<<"num entities: "<<numEntities<<std::endl;
}

//=========== Test class that mimics an element algorithm ==============
//=========== and uses std::vectors for scratch arrays    ==============
//
class TestElemAlgorithmWithVectors
{
public:
  TestElemAlgorithmWithVectors(stk::mesh::BulkData& bulk,
                    const VectorFieldType* coord, ScalarFieldType* discreteLaplacian,
                    ScalarFieldType* nodalPressure)
  : bulkData_(bulk),
    discreteLaplacianOfPressure(discreteLaplacian),
    nodalPressureField(nodalPressure), coordField(coord)
  {}

  void execute()
  {
      double scs_error = 0.0;
      const stk::mesh::MetaData& meta = bulkData_.mesh_meta_data();
  
      const int nDim = meta.spatial_dimension();
  
      std::vector<double> elemNodeCoords;
      std::vector<double> elemNodePressures;
  
      std::vector<double> scs_areav;
      std::vector<double> dndx;
      std::vector<double> deriv;
      std::vector<double> det_j;
  
      auto resizer = [&](int nodesPerElem, int numScsIp)
      {
          elemNodeCoords.resize(nodesPerElem*nDim);
          elemNodePressures.resize(nodesPerElem);
          scs_areav.resize(numScsIp*nDim);
          dndx.resize(nDim*numScsIp*nodesPerElem);
          deriv.resize(nDim*numScsIp*nodesPerElem);
          det_j.resize(numScsIp);
      };

      const stk::mesh::BucketVector& elemBuckets = bulkData_.get_buckets(stk::topology::ELEM_RANK, meta.locally_owned_part());
  
      int maxNodesPerElement = 0, maxNumScsIp = 0;
      find_max_nodes_and_ips(elemBuckets, maxNodesPerElement, maxNumScsIp);
      resizer(maxNodesPerElement, maxNumScsIp);

      bucket_loop_serial_only(elemBuckets,
          [&](stk::topology topo, sierra::nalu::MasterElement& meSCS)
          {
              const int nodesPerElem = topo.num_nodes();
              resizer(nodesPerElem, meSCS.num_integration_points());
          }
          ,
          [&](stk::mesh::Entity elem, stk::topology topo, sierra::nalu::MasterElement& meSCS)
          {
              const stk::mesh::Entity* elemNodes = bulkData_.begin_nodes(elem);
      
              double* p_elemNodeCoords = elemNodeCoords.data();
              double* p_elemNodePressures = elemNodePressures.data();
      
              double* p_scs_areav = scs_areav.data();
              double* p_dndx = dndx.data();
              double* p_deriv = deriv.data();
              double* p_det_j = det_j.data();
              const int* lrscv = meSCS.adjacentNodes();
      
              const int numScsIp = meSCS.num_integration_points();
              const int nodesPerElem = topo.num_nodes();
              for(int n=0; n<nodesPerElem; ++n) {
                  const double* nodeCoords = stk::mesh::field_data(*coordField, elemNodes[n]);
      
                  const int nodeOffset = n*nDim;
                  for(int d=0; d<nDim; ++d) {
                      p_elemNodeCoords[nodeOffset+d] = nodeCoords[d];
                  }
                  const double* nodePressure = stk::mesh::field_data(*nodalPressureField, elemNodes[n]);
                  p_elemNodePressures[n] = nodePressure[0];
              }
      
              meSCS.determinant(1, p_elemNodeCoords, p_scs_areav, &scs_error);
              meSCS.grad_op(1, p_elemNodeCoords, p_dndx, p_deriv, p_det_j, &scs_error);
      
              for (int ip = 0; ip < numScsIp; ++ip ) {
      
                double dpdxIp = 0.0;
                const int ipOffset = nDim*nodesPerElem*ip;
                for ( int ic = 0; ic < nodesPerElem; ++ic) {
                  const int offSetDnDx = ipOffset + ic*nDim;
                  for ( int j = 0; j < nDim; ++j ) {
                    dpdxIp += p_dndx[offSetDnDx+j]*p_elemNodePressures[ic]*p_scs_areav[ip*nDim+j];
                  }
                }
                EXPECT_TRUE(std::abs(dpdxIp) > tol);
      
                const stk::mesh::Entity lNode = elemNodes[lrscv[2*ip+0]];
                const stk::mesh::Entity rNode = elemNodes[lrscv[2*ip+1]];
      
                Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, lNode), dpdxIp);
                Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, rNode), -dpdxIp);
              }
          }
      );
  }

private:
  stk::mesh::BulkData& bulkData_;
  ScalarFieldType* discreteLaplacianOfPressure;
  ScalarFieldType* nodalPressureField;
  const VectorFieldType* coordField;
};


//======= templated element kernel function ==================

template<int nodesPerElem, int numScsIp>
void element_discrete_laplacian_kernel_3d(stk::mesh::BulkData& bulkData, stk::mesh::Entity elem,
                       sierra::nalu::MasterElement& meSCS,
                       ScalarFieldType* discreteLaplacianOfPressure,
                       ScalarFieldType* nodalPressureField,
                       const VectorFieldType* coordField)
{
    const int nDim = 3;
    const stk::mesh::Entity* elemNodes = bulkData.begin_nodes(elem);

    double p_elemNodeCoords[nodesPerElem*nDim];
    double p_elemNodePressures[nodesPerElem];

    double p_scs_areav[numScsIp*nDim];
    double p_dndx[nDim*numScsIp*nodesPerElem];
    double p_deriv[nDim*numScsIp*nodesPerElem];
    double p_det_j[numScsIp];
    const int* lrscv = meSCS.adjacentNodes();

    for(int n=0; n<nodesPerElem; ++n) {
        const double* nodeCoords = stk::mesh::field_data(*coordField, elemNodes[n]);

        const int nodeOffset = n*nDim;
        for(int d=0; d<nDim; ++d) {
            p_elemNodeCoords[nodeOffset+d] = nodeCoords[d];
        }
        const double* nodePressure = stk::mesh::field_data(*nodalPressureField, elemNodes[n]);
        p_elemNodePressures[n] = nodePressure[0];
    }

    double scs_error = 0;
    meSCS.determinant(1, p_elemNodeCoords, p_scs_areav, &scs_error);
    meSCS.grad_op(1, p_elemNodeCoords, p_dndx, p_deriv, p_det_j, &scs_error);

    for (int ip = 0; ip < numScsIp; ++ip ) {

      double dpdxIp = 0.0;
      const int ipOffset = nDim*nodesPerElem*ip;
      for ( int ic = 0; ic < nodesPerElem; ++ic) {
        const int offSetDnDx = ipOffset + ic*nDim;
        for ( int j = 0; j < nDim; ++j ) {
          dpdxIp += p_dndx[offSetDnDx+j]*p_elemNodePressures[ic]*p_scs_areav[ip*nDim+j];
        }
      }
      EXPECT_TRUE(std::abs(dpdxIp) > tol);

      const stk::mesh::Entity lNode = elemNodes[lrscv[2*ip+0]];
      const stk::mesh::Entity rNode = elemNodes[lrscv[2*ip+1]];

      Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, lNode), dpdxIp);
      Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, rNode), -dpdxIp);
    }
}

//=========== Test class that mimics an element algorithm ==============
//=========== and uses a templated kernel (with compile-time scratch arrays)  ==============
//
class TestElemAlgorithmWithTemplate
{
public:
  TestElemAlgorithmWithTemplate(stk::mesh::BulkData& bulk,
                    const VectorFieldType* coord, ScalarFieldType* discreteLaplacian,
                    ScalarFieldType* nodalPressure)
  : bulkData_(bulk),
    discreteLaplacianOfPressure(discreteLaplacian),
    nodalPressureField(nodalPressure), coordField(coord)
  {}

  void execute()
  {
      const stk::mesh::MetaData& meta = bulkData_.mesh_meta_data();
  
      const stk::mesh::BucketVector& elemBuckets = bulkData_.get_buckets(stk::topology::ELEM_RANK, meta.locally_owned_part());
  
      kokkos_thread_team_bucket_loop_with_topo(elemBuckets,
          [&](stk::mesh::Entity elem, stk::topology topo, sierra::nalu::MasterElement& meSCS)
          {
             //this is an incomplete switch, doesn't handle all possible topologies...
             //just an illustration for this test.
              switch(topo) {
              case stk::topology::HEX_8:
                  element_discrete_laplacian_kernel_3d<8,12>(bulkData_, elem, meSCS,
                       discreteLaplacianOfPressure, nodalPressureField, coordField);
                  break;
              case stk::topology::HEX_27:
                  element_discrete_laplacian_kernel_3d<27,216>(bulkData_, elem, meSCS,
                       discreteLaplacianOfPressure, nodalPressureField, coordField);
                  break;
              case stk::topology::TET_4:
                  element_discrete_laplacian_kernel_3d<4,6>(bulkData_, elem, meSCS,
                       discreteLaplacianOfPressure, nodalPressureField, coordField);
                  break;
              case stk::topology::PYRAMID_5:
                  element_discrete_laplacian_kernel_3d<5,8>(bulkData_, elem, meSCS,
                       discreteLaplacianOfPressure, nodalPressureField, coordField);
                  break;
              case stk::topology::WEDGE_6:
                  element_discrete_laplacian_kernel_3d<6,9>(bulkData_, elem, meSCS,
                       discreteLaplacianOfPressure, nodalPressureField, coordField);
                  break;
              default:
                  std::cerr<<"ERROR! Unhandled topology: "<<topo<<std::endl;
                  break;
              }
          }
      );
  }

private:
  stk::mesh::BulkData& bulkData_;
  ScalarFieldType* discreteLaplacianOfPressure;
  ScalarFieldType* nodalPressureField;
  const VectorFieldType* coordField;
};

//=========== Test class that mimics an element algorithm ==============
//=========== and uses Kokkos::views for scratch arrays    ==============
//
class TestElemAlgorithmWithViews
{
public:
  TestElemAlgorithmWithViews(stk::mesh::BulkData& bulk,
                    const VectorFieldType* coord, ScalarFieldType* discreteLaplacian,
                    ScalarFieldType* nodalPressure)
  : bulkData_(bulk),
    discreteLaplacianOfPressure(discreteLaplacian),
    nodalPressureField(nodalPressure), coordField(coord)
  {}

  void execute()
  {
    double scs_error = 0.0;
    const stk::mesh::MetaData& meta = bulkData_.mesh_meta_data();

    const int nDim = meta.spatial_dimension();

    const stk::mesh::BucketVector& elemBuckets = bulkData_.get_buckets(stk::topology::ELEM_RANK, meta.locally_owned_part());

    int maxNodesPerElement = 0, maxNumScsIp = 0;
    find_max_nodes_and_ips(elemBuckets, maxNodesPerElement, maxNumScsIp);

    const int bytes_per_team = 0;
    const int bytes_per_thread =
       sierra::nalu::SharedMemView<double**>::shmem_size(maxNodesPerElement, nDim) +
       sierra::nalu::SharedMemView<double*>::shmem_size(maxNodesPerElement) +
       sierra::nalu::SharedMemView<double**>::shmem_size(maxNumScsIp, nDim) +
       sierra::nalu::SharedMemView<double**>::shmem_size(maxNumScsIp, maxNodesPerElement*nDim) +
       sierra::nalu::SharedMemView<double**>::shmem_size(maxNumScsIp, maxNodesPerElement*nDim) +
       sierra::nalu::SharedMemView<double*>::shmem_size(maxNumScsIp);

    auto team_exec = sierra::nalu::get_host_team_policy(elemBuckets.size(), bytes_per_team, bytes_per_thread);
    Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
    {
        const stk::mesh::Bucket& bkt = *elemBuckets[team.league_rank()];
        stk::topology topo = bkt.topology();
        sierra::nalu::MasterElement& meSCS = *sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

        const int nodesPerElem = topo.num_nodes();
        const int numScsIp = meSCS.num_integration_points();

        sierra::nalu::SharedMemView<double**> elemNodeCoords = sierra::nalu::get_shmem_view_2D<double>(team, nodesPerElem, nDim);
        sierra::nalu::SharedMemView<double*> elemNodePressures = sierra::nalu::get_shmem_view_1D<double>(team, nodesPerElem);
     
        sierra::nalu::SharedMemView<double**> scs_areav = sierra::nalu::get_shmem_view_2D<double>(team, numScsIp, nDim);
        sierra::nalu::SharedMemView<double**> dndx = sierra::nalu::get_shmem_view_2D<double>(team, numScsIp, nodesPerElem*nDim);
        sierra::nalu::SharedMemView<double**> deriv = sierra::nalu::get_shmem_view_2D<double>(team, numScsIp, nodesPerElem*nDim);
        sierra::nalu::SharedMemView<double*> det_j = sierra::nalu::get_shmem_view_1D<double>(team, numScsIp);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bkt.size()), [&](const size_t& jj)
        {
             stk::mesh::Entity elem = bkt[jj];
              const stk::mesh::Entity* elemNodes = bulkData_.begin_nodes(elem);
              for(int n=0; n<nodesPerElem; ++n) {
                  const double* nodeCoords = stk::mesh::field_data(*coordField, elemNodes[n]);
      
                  for(int d=0; d<nDim; ++d) {
                      elemNodeCoords(n,d) = nodeCoords[d];
                  }
                  const double* nodePressure = stk::mesh::field_data(*nodalPressureField, elemNodes[n]);
                  elemNodePressures[n] = nodePressure[0];
              }
      
              meSCS.determinant(1, &elemNodeCoords(0,0), &scs_areav(0,0), &scs_error);
              meSCS.grad_op(1, &elemNodeCoords(0,0), &dndx(0,0), &deriv(0,0), &det_j(0), &scs_error);
              const int* lrscv = meSCS.adjacentNodes();
      
              for (int ip = 0; ip < numScsIp; ++ip ) {
      
                double dpdxIp = 0.0;
                for ( int ic = 0; ic < nodesPerElem; ++ic) {
                  for ( int j = 0; j < nDim; ++j ) {
                    dpdxIp += dndx(ip,ic*nDim+j)*elemNodePressures(ic)*scs_areav(ip,j);
                  }
                }
                EXPECT_TRUE(std::abs(dpdxIp) > tol);
      
                const stk::mesh::Entity lNode = elemNodes[lrscv[2*ip+0]];
                const stk::mesh::Entity rNode = elemNodes[lrscv[2*ip+1]];
      
                Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, lNode), dpdxIp);
                Kokkos::atomic_add(stk::mesh::field_data(*discreteLaplacianOfPressure, rNode), -dpdxIp);
              }
        });
    });
  }

private:
  stk::mesh::BulkData& bulkData_;
  ScalarFieldType* discreteLaplacianOfPressure;
  ScalarFieldType* nodalPressureField;
  const VectorFieldType* coordField;
};

//========= below are the test 'main's... ===============

TEST_F(Hex8Mesh, indexing_vectors)
{
    fill_mesh_and_initialize_test_fields();

    TestElemAlgorithmWithVectors testAlgorithm(bulk, coordField,
                          discreteLaplacianOfPressure, nodalPressureField);

    testAlgorithm.execute();

    check_discrete_laplacian(exactLaplacian);
}

TEST_F(Hex8Mesh, indexing_template_raw_arrays)
{
    fill_mesh_and_initialize_test_fields();

    TestElemAlgorithmWithTemplate testAlgorithm(bulk, coordField,
                          discreteLaplacianOfPressure, nodalPressureField);

    testAlgorithm.execute();

    check_discrete_laplacian(exactLaplacian);
}

TEST_F(Hex8Mesh, indexing_views)
{
    fill_mesh_and_initialize_test_fields();

    TestElemAlgorithmWithViews testAlgorithm(bulk, coordField,
                          discreteLaplacianOfPressure, nodalPressureField);

    testAlgorithm.execute();

    check_discrete_laplacian(exactLaplacian);
}

//end of stuff that's ifndef'd for KOKKOS_ENABLE_CUDA
#endif

}

