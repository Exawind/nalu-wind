// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "HypreDeterministicAssembly.h"

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <map>
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
  assembly.setup(1, rows.size() - 1, rows.data(), cols.data());
  const auto* resultStorage = assembly.entries().values.data();
  std::vector<HYPRE_Int> rhsRows{row};
  if (rank != owner)
    rhsRows.push_back(target);
  HypreDeterministicAssembly rhsAssembly(MPI_COMM_WORLD, row, row);
  rhsAssembly.setup(1, rhsRows.size() - 1, rhsRows.data(), nullptr, 3);
  std::vector<double> rhsValues(rhsRows.size() * 3);
  for (unsigned component = 0; component < 3; ++component) {
    const auto offset = component * rhsRows.size();
    rhsValues[offset] = component == 2 ? 0.0 : (rank == owner ? 1.e16 : 2.0);
    if (rank != owner)
      rhsValues[offset + 1] = component == 2 ? 0.0
                              : rank == 0    ? (component == 0 ? 1.0 : 0.5)
                              : rank == 1    ? -1.e16
                                             : 0.0;
  }
  for (int repeat = 0; repeat < 20; ++repeat) {
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(
      std::chrono::milliseconds((rank + repeat) % size));
    const auto& result = assembly.reduce(values.data());
    EXPECT_EQ(result.values.data(), resultStorage);
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
    auto queryRows = result.rows;
    auto queryCols = result.cols;
    ASSERT_EQ(
      HYPRE_IJMatrixGetValues(
        matrix, result.rows.size(), counts.data(), queryRows.data(),
        queryCols.data(), actual.data()),
      0);
    EXPECT_EQ(actual, result.values);
    HYPRE_IJMatrixDestroy(matrix);

    // All segregated RHS components, including zero values, share one exchange.
    const auto& rhs = rhsAssembly.reduce(rhsValues.data());
    for (unsigned component = 0; component < 3; ++component) {
      const double expected = component == 2 || rank == owner ? 0.0 : 2.0;
      ASSERT_EQ(rhs.rows.size(), 1u);
      EXPECT_EQ(rhs.rows[0], row);
      EXPECT_EQ(rhs.values[component], expected);
      HYPRE_IJVector vector;
      ASSERT_EQ(HYPRE_IJVectorCreate(MPI_COMM_WORLD, row, row, &vector), 0);
      ASSERT_EQ(HYPRE_IJVectorSetObjectType(vector, HYPRE_PARCSR), 0);
      ASSERT_EQ(HYPRE_IJVectorInitialize(vector), 0);
      ASSERT_EQ(
        HYPRE_IJVectorSetValues(
          vector, rhs.rows.size(), rhs.rows.data(),
          rhs.values.data() + component),
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
  assembly.setup(rank == owner ? 1 : 0, rank == owner ? 0 : 1, &row, nullptr);
  auto result = assembly.reduce(&value);
  if (rank == owner) {
    EXPECT_EQ(result.rows, (std::vector<HYPRE_Int>{0}));
    EXPECT_EQ(result.values, (std::vector<double>{size + 1.0}));
  } else {
    EXPECT_TRUE(result.rows.empty());
  }
  assembly.setup(0, 0, nullptr, nullptr);
  result = assembly.reduce(nullptr);
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
  assembly.setup(1, 2, rows, nullptr);
  const auto* storage = assembly.receivedValues().data();
  for (int repeat = 1; repeat <= 3; ++repeat) {
    values[0] = repeat;
    values[1] = 2 * repeat;
    values[2] = 4 * repeat;
    const auto& result = assembly.reduce(values);
    EXPECT_EQ(result.rows, (std::vector<HYPRE_Int>{rank}));
    EXPECT_EQ(result.values, (std::vector<double>{7.0 * repeat}));
    EXPECT_EQ(assembly.receivedValues().data(), storage);
  }
  assembly.setup(1, 0, rows, nullptr);
  EXPECT_EQ(assembly.reduce(values).values, (std::vector<double>{3.0}));
  EXPECT_TRUE(assembly.receivedValues().empty());
}

TEST(HypreDeterministicAssembly, invalid_rows_fail_collectively)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, rank, rank);
  HYPRE_Int invalidRow = size;
  EXPECT_THROW(
    assembly.setup(0, rank == 0 ? 1 : 0, &invalidRow, nullptr),
    std::runtime_error);
  EXPECT_FALSE(assembly.isSetup());
  EXPECT_THROW(
    assembly.setup(rank == 0 ? 1 : 0, 0, &invalidRow, nullptr),
    std::runtime_error);
  // A failed setup must leave the object reusable.
  HYPRE_Int row = rank;
  assembly.setup(1, 0, &row, nullptr);
  double value = 1.0;
  EXPECT_EQ(assembly.reduce(&value).values, (std::vector<double>{1.0}));
}

TEST(HypreDeterministicAssembly, signed_zero_and_duplicate_owned_entries)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, rank, rank);
  HYPRE_Int rows[] = {rank, rank, (rank + 1) % size};
  HYPRE_Int cols[] = {rank, rank, size};
  double values[] = {-0.0, -0.0, -0.0};
  assembly.setup(2, 1, rows, cols);
  const auto& result = assembly.reduce(values);
  ASSERT_EQ(result.values.size(), 2u);
  EXPECT_TRUE(std::signbit(result.values[0]));
  EXPECT_TRUE(std::signbit(result.values[1]));
}

TEST(HypreDeterministicAssembly, strided_batch_matches_original_reduction)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, rank, rank);
  // Each sender has two duplicate entries to every rank, in non-rank order.
  std::vector<HYPRE_Int> rows{rank}, cols{rank};
  for (int dest = size - 1; dest >= 0; --dest) {
    rows.insert(rows.end(), {dest, dest, dest});
    cols.insert(cols.end(), {dest, size, size});
  }
  const size_t stride = rows.size() + 2;
  std::vector<double> values(3 * stride, -999.0);
  assembly.setup(1, rows.size() - 1, rows.data(), cols.data(), 3);
  const auto* storage = assembly.entries().values.data();
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (unsigned c = 0; c < 3; ++c) {
      auto* component = values.data() + c * stride;
      component[0] = c == 2 ? 0.0 : 1.e16;
      for (size_t i = 1; i < rows.size(); i += 3) {
        component[i] = c == 2 ? 0.0 : (rank == 0 ? -1.e16 : repeat + 1.0);
        component[i + 1] = c == 2 ? -0.0 : 1.e16;
        component[i + 2] = c == 2 ? -0.0 : -1.e16;
      }
    }
    // Independent reference: the original map insertion/addition sequence.
    std::vector<double> gathered(size * values.size());
    MPI_Allgather(
      values.data(), values.size(), MPI_DOUBLE, gathered.data(), values.size(),
      MPI_DOUBLE, MPI_COMM_WORLD);
    const auto& result = assembly.reduce(values.data(), stride);
    EXPECT_EQ(result.values.data(), storage);
    ASSERT_EQ(result.rows.size(), 2u);
    for (unsigned c = 0; c < 3; ++c) {
      std::map<HYPRE_Int, double> expected;
      expected.emplace(rank, values[c * stride]);
      for (int source = 0; source < size; ++source) {
        const auto offset = source * values.size() + c * stride;
        const auto entry = 1 + 3 * (size - 1 - rank);
        for (unsigned j = 0; j < 3; ++j) {
          const auto col = j == 0 ? rank : size;
          const double value = gathered[offset + entry + j];
          auto inserted = expected.emplace(col, value);
          if (!inserted.second)
            inserted.first->second += value;
        }
      }
      size_t slot = 0;
      for (const auto& item : expected) {
        EXPECT_EQ(result.cols[slot], item.first);
        const auto actual = result.values[c * result.rows.size() + slot++];
        EXPECT_EQ(actual, item.second);
        EXPECT_EQ(std::signbit(actual), std::signbit(item.second));
      }
    }
  }
}

TEST(HypreDeterministicAssembly, zero_component_batch)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, rank, rank);
  HYPRE_Int rows[] = {rank, rank};
  assembly.setup(1, 1, rows, nullptr, 0);
  EXPECT_TRUE(assembly.reduce(nullptr).values.empty());
}

TEST(HypreDeterministicAssembly, inconsistent_components_fail_collectively)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == 1)
    GTEST_SKIP() << "Requires multiple ranks";
  HypreDeterministicAssembly assembly(MPI_COMM_WORLD, rank, rank);
  HYPRE_Int rows[] = {rank, (rank + 1) % size};
  EXPECT_THROW(
    assembly.setup(1, 1, rows, nullptr, rank == 0 ? 1 : 3), std::runtime_error);
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
