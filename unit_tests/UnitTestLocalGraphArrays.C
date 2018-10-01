#include <gtest/gtest.h>

#include <KokkosInterface.h>
#include <LinearSolver.h>

#include <limits>
#include <vector>

TEST(LocalGraphArrays, compute_row_pointers)
{
  unsigned N = 5;
  size_t nnz = 0;
  Kokkos::View<size_t*,sierra::nalu::HostSpace> rowLengths("rowLengths", N);
  for(unsigned i=0; i<N; ++i) {
    rowLengths(i) = i+2;
    nnz += rowLengths(i);
  }

  Kokkos::View<size_t*,sierra::nalu::HostSpace> rowPointers("rowPtrs", N+1);
  sierra::nalu::LocalGraphArrays::compute_row_pointers(rowPointers, rowLengths);
  EXPECT_EQ(nnz, rowPointers(N));
}

Teuchos::RCP<sierra::nalu::LocalGraphArrays> create_graph(const std::vector<size_t>& rowLens)
{
  unsigned N = rowLens.size();
  Kokkos::View<size_t*,sierra::nalu::MemSpace> rowLengths("rowLengths", N);
  for(unsigned i=0; i<N; ++i) {
    rowLengths(i) = rowLens[i];
  }
  return Teuchos::rcp(new sierra::nalu::LocalGraphArrays(rowLengths));
}

TEST(LocalGraphArrays, construct)
{
  Teuchos::RCP<sierra::nalu::LocalGraphArrays> csg = create_graph({2, 3, 4});
  EXPECT_EQ(2u, csg->get_row_length(0));
  EXPECT_EQ(3u, csg->get_row_length(1));
  EXPECT_EQ(4u, csg->get_row_length(2));

  size_t nnz = 2 + 3 + 4;
  EXPECT_EQ(nnz, csg->colIndices.size());
}

TEST(LocalGraphArrays, insertIndicesNumDof1)
{
  Teuchos::RCP<sierra::nalu::LocalGraphArrays> csg = create_graph({2, 3, 4});

  int numDof = 1;

  std::vector<LocalOrdinal> cols = {0, 1};
  csg->insertIndices(0, cols.size(), cols.data(), numDof);

  cols = {2, 3, 4};
  csg->insertIndices(1, cols.size(), cols.data(), numDof);

  cols = {7, 8, 6, 5};
  csg->insertIndices(2, cols.size(), cols.data(), numDof);

  for(size_t i=0; i<csg->colIndices.size(); ++i) {
    EXPECT_EQ((int)i, csg->colIndices[i]);
  }

  //now insert row-0 and row-1 cols again, then make sure the graph is still correct.
  std::vector<LocalOrdinal> dupCols = {0, 1};
  csg->insertIndices(0, dupCols.size(), dupCols.data(), numDof);
  dupCols = {2, 3, 4};
  csg->insertIndices(1, dupCols.size(), dupCols.data(), numDof);

  for(size_t i=0; i<csg->colIndices.size(); ++i) {
    EXPECT_EQ((int)i, csg->colIndices[i]);
  }
}

TEST(LocalGraphArrays, insertIndicesNumDof3)
{
  Teuchos::RCP<sierra::nalu::LocalGraphArrays> csg = create_graph({6, 9, 12});

  int numDof = 3;

  std::vector<LocalOrdinal> cols = {0, 3};
  csg->insertIndices(0, cols.size(), cols.data(), numDof);

  cols = {6, 9, 12};
  csg->insertIndices(1, cols.size(), cols.data(), numDof);

  cols = {21, 18, 24, 15};
  csg->insertIndices(2, cols.size(), cols.data(), numDof);

  EXPECT_EQ(27u, csg->colIndices.size());
  for(size_t i=0; i<csg->colIndices.size(); ++i) {
    EXPECT_EQ((int)i, csg->colIndices[i]);
  }

  //now insert row-0 and row-1 cols again, then make sure the graph is still correct.
  std::vector<LocalOrdinal> dupCols = {0, 3};
  csg->insertIndices(0, dupCols.size(), dupCols.data(), numDof);
  dupCols = {6, 9, 12};
  csg->insertIndices(1, dupCols.size(), dupCols.data(), numDof);

  EXPECT_EQ(27u, csg->colIndices.size());
  for(size_t i=0; i<csg->colIndices.size(); ++i) {
    EXPECT_EQ((int)i, csg->colIndices[i]);
  }
}

