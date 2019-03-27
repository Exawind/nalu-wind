#ifndef _UnitTestKokkosUtils_h_
#define _UnitTestKokkosUtils_h_

#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <Kokkos_Core.hpp>

#include <KokkosInterface.h>
#include "UnitTestUtils.h"

template<class OUTER_LOOP_BODY, class INNER_LOOP_BODY>
void bucket_loop_serial_only(const stk::mesh::BucketVector& buckets, const OUTER_LOOP_BODY& outer_loop_body, const INNER_LOOP_BODY& inner_loop_body)
{
    for(const stk::mesh::Bucket* bptr : buckets)
    {   
        const stk::mesh::Bucket& bkt = *bptr;
        stk::topology topo = bkt.topology();
        sierra::nalu::MasterElement* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

        outer_loop_body(topo,*meSCS);

        for(size_t j=0; j<bkt.size(); ++j)
        {
            inner_loop_body(bkt[j], topo, *meSCS);
        }
    }
}

template<class LOOP_BODY>
void kokkos_bucket_loop(const stk::mesh::BucketVector& buckets, LOOP_BODY inner_loop_body)
{
    Kokkos::parallel_for(buckets.size(), [&](const size_t& i)
    {
        const stk::mesh::Bucket& bkt = *buckets[i];
        for(size_t j=0; j<bkt.size(); ++j)
        {
            inner_loop_body(bkt[j]);
        }
    });
}

template<class LOOP_BODY>
void kokkos_thread_team_bucket_loop(const stk::mesh::BucketVector& buckets, LOOP_BODY inner_loop_body)
{
    Kokkos::parallel_for(sierra::nalu::DeviceTeamPolicy(buckets.size(), Kokkos::AUTO), KOKKOS_LAMBDA(const sierra::nalu::TeamHandleType& team)
    {
        const stk::mesh::Bucket& bkt = *buckets[team.league_rank()];
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bkt.size()), [&](const size_t& j)
        {
            inner_loop_body(bkt[j]);
        });
    });
}

template<class LOOP_BODY>
void kokkos_thread_team_bucket_loop_with_topo(const stk::mesh::BucketVector& buckets,
                                    const LOOP_BODY& inner_loop_body)
{
    Kokkos::parallel_for(sierra::nalu::DeviceTeamPolicy(buckets.size(), Kokkos::AUTO), KOKKOS_LAMBDA(const sierra::nalu::TeamHandleType& team)
    {
        const stk::mesh::Bucket& bkt = *buckets[team.league_rank()];
        stk::topology topo = bkt.topology();
        sierra::nalu::MasterElement* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bkt.size()), [&](const size_t& j)
        {
            inner_loop_body(bkt[j], topo, *meSCS);
        });
    });
}

#endif

