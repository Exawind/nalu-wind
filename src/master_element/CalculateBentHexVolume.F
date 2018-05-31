      subroutine bhexVolumeByTriangleFacets( volcoords, volume )
c     This subroutine calculates the volume of a hexahedron using
c     triangular facets. (see "Efficient computation of volume of
c     Hexahedral Cells", Jeffrey Grandy, LLNL, UCRL-ID-128886, 
c     October 30, 1997.
c
c     Modified for a non-planar bent top face which we have
c     broken into two triangles
c
c
c     coordinates of vertices of hexahedron
c     =====================================
c     double    volcoords  (3,8)
c
c     volume output
c     =============
c     double    volume
c
      implicit none
      double precision volcoords, volume
      double precision coords

      integer triangularFacetTable(3,24)
      integer j, k, ncoords, ntriangles

      dimension volcoords(3, 8)

      dimension coords(3,14)

      ! this table contains the vertices defining each triangular facet
      data triangularFacetTable /
     .  0, 8, 1,
     .  8, 2, 1,
     .  3, 2, 8,
     .  3, 8, 0,
     .  6, 9, 5,
     .  7, 9, 6,
     .  4, 9, 7,
     .  4, 5, 9,
     .  10, 0, 1,
     .  5, 10, 1,
     .  4, 10, 5,
     .  4, 0, 10,
     .  7, 6, 11,
     .  6, 2, 11,
     .  2, 3, 11,
     .  3, 7, 11,
     .  6, 12, 2,
     .  5, 12, 6,
     .  5, 1, 12,
     .  1, 2, 12,
     .  0, 4, 13,
     .  4, 7, 13,
     .  7, 3, 13,
     .  3, 0, 13 /

      ! the first coordinates are equivalent to vertices
      do j = 1, 8
        do k = 1,3
          coords(k,j) = volcoords(k,j)
        end do
      end do
      ! now we add the face midpoints
      do k = 1,3
        coords(k,9) = 0.25d0*( volcoords(k,1)
     .    + volcoords(k,2)
     .    + volcoords(k,3) + volcoords(k,4) )
      end do
      do k = 1,3
        coords(k,10) = 0.5d0*( volcoords(k,6)
     .    + volcoords(k,8) )
      end do
      do k = 1,3
        coords(k,11) = 0.25d0*( volcoords(k,1)
     .     + volcoords(k,2)
     .    + volcoords(k,6) + volcoords(k,5) )
      end do
      do k = 1,3
        coords(k,12) = 0.25d0*( volcoords(k,4)
     .     + volcoords(k,3)
     .    + volcoords(k,7) + volcoords(k,8) )
      end do
      do k = 1,3
        coords(k,13) = 0.25d0*( volcoords(k,2)
     .    + volcoords(k,3)
     .    + volcoords(k,7) + volcoords(k,6) )
      end do
      do k = 1,3
        coords(k,14) = 0.25d0*( volcoords(k,1)
     .    + volcoords(k,4)
     .    + volcoords(k,8) + volcoords(k,5) )
      end do

      ! number of vertices for the equivalent tetrahedron
      ncoords = 14
      ! number of triangular facets
      ntriangles = 24

      ! calculate the volume using the triangular facets
      call polyhedralVolumeByFaces(ncoords, coords(1,1), ntriangles,
     .  triangularFacetTable(1,1), volume)

      end
