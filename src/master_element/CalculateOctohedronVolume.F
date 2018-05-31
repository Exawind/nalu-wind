      subroutine octohedronVolumeByTriangleFacets( volcoords, volume )
c     This subroutine calculates the volume of the tip of a the pyramid
c     element by constructing a polyhedron of triangular facets.
c
c     coorinates of vertices of the pyramid tip octohedron
c     ====================================================
c     double    volcoords  (n_v3d,10)
c
c     output volume
c     =============
c     double    volume
c
      implicit none
      double precision volcoords, volume
      double precision coords
      integer triangularFacetTable
      integer j, k, ncoords, ntriangles

      dimension volcoords(3, 10)
      dimension coords(3,14)
      dimension triangularFacetTable(3,24)
     
      ! this table defines which vertices compose each triangle
      data triangularFacetTable /
     .  1, 3, 10,
     .  2, 10, 3,
     .  2, 9, 10,
     .  10, 9, 1,
     .  4, 3, 11,
     .  3, 1, 11,
     .  11, 1, 5,
     .  4, 11, 5,
     .  1, 12, 5,
     .  1, 7, 12,
     .  12, 7, 6,
     .  5, 12, 6,
     .  9, 8, 13,
     .  13, 8, 7,
     .  13, 7, 1,
     .  9, 13, 1,
     .  4, 5, 0,
     .  5, 6, 0,
     .  6, 7, 0,
     .  7, 8, 0,
     .  0, 8, 9,
     .  0, 9, 2,
     .  0, 2, 3,
     .  0, 3, 4 /

      ! the first ten coordinates are the vertices of the octohedron
      do j = 1, 10
        do k = 1,3
          coords(k,j) = volcoords(k,j)
        end do
      end do
      ! we now add face midpoints only for the four faces that are
      ! not planar
      do k = 1,3
        coords(k,11) = 0.5d0*( volcoords(k,4)
     .    + volcoords(k,10) )
      end do
      do k = 1,3
        coords(k,12) = 0.50d0*( volcoords(k,4)
     .    + volcoords(k,6) )
      end do
      do k = 1,3
        coords(k,13) = 0.50d0*( volcoords(k,6)
     .     + volcoords(k,8) )
      end do
      do k = 1,3
        coords(k,14) = 0.50d0*( volcoords(k,8)
     .     + volcoords(k,10) )
      end do

      ncoords = 14
      ntriangles = 24

      ! compute the volume using the new equivalent polyhedron
      call polyhedralVolumeByFaces(ncoords, coords(1,1), ntriangles,
     .  triangularFacetTable(1,1), volume)
      
      end
