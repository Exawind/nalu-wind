// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef BucketLoop_h
#define BucketLoop_h

#include <stk_mesh/base/Bucket.hpp>

namespace sierra {
namespace nalu {

template <class LOOP_BODY>
void
bucket_loop(const stk::mesh::BucketVector& buckets, LOOP_BODY inner_loop_body)
{
  for (const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket& bkt = *bptr;
    for (size_t j = 0; j < bkt.size(); ++j) {
      inner_loop_body(bkt[j]);
    }
  }
}

} // namespace nalu
} // namespace sierra

#endif
