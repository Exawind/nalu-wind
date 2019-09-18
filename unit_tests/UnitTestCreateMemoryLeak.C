/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "gtest/gtest.h"

TEST(CreateMemoryLeak, create_memory_leak)
{
  double *p = new double;
  *p = 1.0;
}
