// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "HypreDeterministicAssembly.h"

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <gtest/gtest.h>
#include <chrono>
#include <stdexcept>
#include <thread>

using sierra::kynema_ugf::HypreDeterministicAssembly;

TEST(HypreDeterministicAssembly, rank_order_and_missing_columns)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 3)
    GTEST_SKIP() << "Requires at least three ranks";

  const int owner = size - 1;
  const HYPRE_Int row = rank;
  const HYPRE_Int target = owner;
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, row, row);
  std::vector<HYPRE_Int> rows{row}, cols{row};
  std::vector<double> values{rank == owner ? 1.e16 : 2.0};
  if (rank != owner) {
    rows.insert(rows.end(), {target, target, target, target});
    cols.insert(cols.end(), {target, 0, 0, 0});
    values.push_back(rank == 0 ? 1.0 : (rank == 1 ? -1.e16 : 0.0));
    // No (target, 0) slot exists on the owner. Preserve duplicate buffer order.
    values.insert(values.end(), {1.e16, 1.0, -1.e16});
  }
  const auto original = values;
  for (int repeat = 0; repeat < 20; ++repeat) {
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(
      std::chrono::milliseconds((rank + repeat) % size));
    auto result = assembly.reduce(
      1, rows.size() - 1, rows.data(), cols.data(), values.data());
    EXPECT_EQ(values, original);
    if (rank == owner) {
      ASSERT_EQ(result.rows.size(), 2u);
      EXPECT_EQ(result.rows, (std::vector<HYPRE_Int>{target, target}));
      EXPECT_EQ(result.cols, (std::vector<HYPRE_Int>{0, target}));
      EXPECT_EQ(result.values, (std::vector<double>{0.0, 0.0}));
    } else {
      EXPECT_EQ(result.rows, (std::vector<HYPRE_Int>{row}));
      EXPECT_EQ(result.values, (std::vector<double>{2.0}));
    }

    HYPRE_IJMatrix matrix;
    ASSERT_EQ(
      HYPRE_IJMatrixCreate(MPI_COMM_WORLD, row, row, row, row, &matrix), 0);
    ASSERT_EQ(HYPRE_IJMatrixSetObjectType(matrix, HYPRE_PARCSR), 0);
    ASSERT_EQ(HYPRE_IJMatrixInitialize(matrix), 0);
    std::vector<HYPRE_Int> counts(result.rows.size(), 1);
    ASSERT_EQ(
      HYPRE_IJMatrixSetValues2(
        matrix, result.rows.size(), nullptr, result.rows.data(), nullptr,
        result.cols.data(), result.values.data()),
      0);
    ASSERT_EQ(HYPRE_IJMatrixAssemble(matrix), 0);
    std::vector<double> actual(result.values.size(), -1.0);
    ASSERT_EQ(
      HYPRE_IJMatrixGetValues(
        matrix, result.rows.size(), counts.data(), result.rows.data(),
        result.cols.data(), actual.data()),
      0);
    EXPECT_EQ(actual, result.values);
    HYPRE_IJMatrixDestroy(matrix);

    // Exercise each segregated RHS component, including an all-zero component.
    for (unsigned component = 0; component < 3; ++component) {
      std::vector<HYPRE_Int> rhsRows{row};
      std::vector<double> rhsValues{
        component == 2 ? 0.0 : (rank == owner ? 1.e16 : 2.0)};
      if (rank != owner) {
        rhsRows.push_back(target);
        rhsValues.push_back(
          component == 2 ? 0.0
          : rank == 0    ? (component == 0 ? 1.0 : 0.5)
          : rank == 1    ? -1.e16
                         : 0.0);
      }
      auto rhs = assembly.reduce(
        1, rhsRows.size() - 1, rhsRows.data(), nullptr, rhsValues.data());
      const double expected = component == 2 || rank == owner ? 0.0 : 2.0;
      ASSERT_EQ(rhs.rows.size(), 1u);
      EXPECT_EQ(rhs.rows[0], row);
      EXPECT_EQ(rhs.values[0], expected);
      HYPRE_IJVector vector;
      ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_WORLD, row, row, &vector), 0);
      ASSERT_EQ(HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJVectorInitialize(vector), 0);
      ASSERT_EQ(
        HYPRE_IJVectorSetValues(
          vector, rhs.rows.size(), rhs.rows.data(), rhs.values.data()),
        0);
      ASSERT_EQ(HYPRE_IJVectorAssemble(vector), 0);
      double actualRhs = -1.0;
      ASSERT_EQ(HYPRE_IJVectorGetValues(vector, 1, &row, &actualRhs), 0);
      EXPECT_EQ(actualRhs, expected);
      HYPRE_IJVectorDestroy(vector);
    }
  }
}

TEST(HypreDeterministicAssembly, empty_partitions_and_no_shared_entries)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int owner = size / 2;
  // Empty partitions before and after the owner have repeated lower bounds.
  HypreDeterministicAssembly assembly(
    MPI_COMM_WORLD, rank <= owner ? 0 : 1, rank < owner ? -1 : 0);
  HYPRE_Int row = 0;
  double value = rank == owner ? 2.0 : 1.0;
  auto result = assembly.reduce(
    rank == owner ? 1 : 0, rank == owner ? 0 : 1, &row, nullptr, &value);
  if (rank == owner) {
    EXPECT_EQ(result.rows, (std::vector<HYPRE_Int>{0}));
    EXPECT_EQ(result.values, (std::vector<double>{size + 1.0}));
  } else {
    EXPECT_TRUE(result.rows.empty());
  }
  result = assembly.reduce(0, 0, nullptr, nullptr, nullptr);
  EXPECT_TRUE(result.rows.empty());
}

TEST(HypreDeterministicAssembly, exchange_in_both_directions)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, rank, rank);
  HYPRE_Int rows[3] = {rank, (rank + 1) % size, (rank + size - 1) % size};
  double values[3] = {1.0, 2.0, 4.0};
  auto result = assembly.reduce(1, 2, rows, nullptr, values);
  EXPECT_EQ(result.rows, (std::vector<HYPRE_Int>{rank}));
  EXPECT_EQ(result.values, (std::vector<double>{7.0}));
}

TEST(HypreDeterministicAssembly, invalid_rows_fail_collectively)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, rank, rank);
  HYPRE_Int invalidRow = size;
  double value = 1.0;
  EXPECT_THROW(
    assembly.reduce(0, rank == 0 ? 1 : 0, &invalidRow, nullptr, &value),
    std::runtime_error);
}

int
main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  HYPRE_Init();
  testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  HYPRE_Finalize();
  MPI_Finalize();
  return result;
}
