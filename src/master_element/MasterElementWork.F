      subroutine hex_scs_det( nelem, npe, nscs,
     .  cordel, area_vec )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 12
c     subcontrol surfaces of the hex.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nscs          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area_vec      real  area vector: A = Ax i + Ay j + Az k
c                                        = Area*(i + j + k)
c
c***********************************************************************
c
      implicit none
      integer nelem, npe, nscs
      double precision cordel, area_vec

      dimension cordel(3, npe, nelem)
      dimension area_vec(3, nelem, nscs)

      double precision coords, scscoords

      integer ielem, j, k, ics, inode, itrianglenode, npf
      ! coordinates of the vertices needed to define the scs
      dimension coords(3,27)
      ! coordinates of the vertices of the scs
      dimension scscoords(3,4)

      double precision half, one4th, one8th

      ! this table defines the scs
      integer HexEdgeFacetTable(4,12)
      data HexEdgeFacetTable /
     .  21, 9, 13, 27,   ! sc face 1 -- points from 1 -> 2
     .  25, 10, 13, 27,  ! sc face 2 -- points from 2 -> 3
     .  11, 13, 27, 24,  ! sc face 3 -- points from 3 -> 4
     .  12, 26, 27, 13,  ! sc face 4 -- points from 1 -> 4
     .  14, 21, 27, 18,  ! sc face 5 -- points from 5 -> 6
     .  18, 15, 25, 27,  ! sc face 6 -- points from 6 -> 7
     .  18, 16, 24, 27,  ! sc face 7 -- points from 7 -> 8
     .  17, 18, 27, 26,  ! sc face 8 -- points from 5 -> 8
     .  20, 21, 27, 26,  ! sc face 9 -- points from 1 -> 5
     .  21, 19, 25, 27,  ! sc face 10 -- points from 2 -> 6
     .  23, 24, 27, 25,  ! sc face 11 -- points from 3 -> 7
     .  22, 26, 27, 24 / ! sc face 12 -- points from 4 -> 8

      half = 0.5d0
      one4th = 0.25d0
      one8th = 0.125d0

      ! loop over elements
      do ielem = 1, nelem
        ! element vertices
        do j = 1,8
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (front)
        ! 4++++11+++3
        ! +         +
        ! +         +
        ! 12   13   10
        ! +         +
        ! +         +
        ! 1++++9++++2

        ! edge midpoints
        do k = 1,3
          coords(k,9) = half*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,10) = half*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,11) = half*(cordel(k,3,ielem) + cordel(k,4,ielem))
        end do
        do k = 1,3
          coords(k,12) = half*(cordel(k,4,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,13) = one4th*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem) + cordel(k,4,ielem))
        end do

        ! face 2 (back)
        ! 8++++16+++7
        ! +         +
        ! +         +
        ! 17   18   15
        ! +         +
        ! +         +
        ! 5++++14+++6

        ! edge midpoints
        do k = 1,3
          coords(k,14) = half*(cordel(k,5,ielem) + cordel(k,6,ielem))
        end do
        do k = 1,3
          coords(k,15) = half*(cordel(k,6,ielem) + cordel(k,7,ielem))
        end do
        do k = 1,3
          coords(k,16) = half*(cordel(k,7,ielem) + cordel(k,8,ielem))
        end do
        do k = 1,3
          coords(k,17) = half*(cordel(k,8,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,18) = one4th*(cordel(k,5,ielem) + cordel(k,6,ielem)
     .      + cordel(k,7,ielem) + cordel(k,8,ielem))
        end do

        ! face 3 (bottom)
        ! 5++++14+++6
        ! +         +
        ! +         +
        ! 20   21   19
        ! +         +
        ! +         +
        ! 1++++9++++2

        ! edge midpoints
        do k = 1,3
          coords(k,19) = half*(cordel(k,2,ielem) + cordel(k,6,ielem))
        end do
        do k = 1,3
          coords(k,20) = half*(cordel(k,1,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,21) = one4th*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,5,ielem) + cordel(k,6,ielem))
        end do

        ! face 4 (top)
        ! 8++++16+++7
        ! +         +
        ! +         +
        ! 22   24   23
        ! +         +
        ! +         +
        ! 4++++11+++3

        ! edge mipdoints
        do k = 1,3
          coords(k,22) = half*(cordel(k,4,ielem) + cordel(k,8,ielem))
        end do
        do k = 1,3
          coords(k,23) = half*(cordel(k,3,ielem) + cordel(k,7,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,24) = one4th*(cordel(k,3,ielem) + cordel(k,4,ielem)
     .      + cordel(k,7,ielem) + cordel(k,8,ielem))
        end do

        ! face 5 (right)
        ! 3++++23+++7
        ! +         +
        ! +         +
        ! 10   25   15
        ! +         +
        ! +         +
        ! 2++++19+++6

        ! face midpoint
        do k = 1,3
          coords(k,25) = one4th*(cordel(k,2,ielem) + cordel(k,3,ielem)
     .      + cordel(k,6,ielem) + cordel(k,7,ielem))
        end do

        ! face 6 (left)
        ! 4++++22+++8
        ! +         +
        ! +         +
        ! 12   26   18
        ! +         +
        ! +         +
        ! 1++++20+++5

        ! face midpoint
        do k = 1,3
          coords(k,26) = one4th*(cordel(k,1,ielem) + cordel(k,4,ielem)
     .      + cordel(k,5,ielem) + cordel(k,8,ielem))
        end do

        ! volume centroid
        do k = 1,3
          coords(k,27) = one8th*(
     .        cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem) + cordel(k,4,ielem)
     .      + cordel(k,5,ielem) + cordel(k,6,ielem)
     .      + cordel(k,7,ielem) + cordel(k,8,ielem) )
        end do

        npf = 4
        ! loop over subcontrol surfaces
        do ics = 1, nscs
          ! loop over vertices of scs
          do inode = 1, npf
            itrianglenode = HexEdgeFacetTable(inode,ics)
            ! set scs coordinates using node table
            do k = 1,3
              scscoords(k,inode) = coords(k,itrianglenode)
            end do
          end do
          ! compute quad area vector using triangle decomposition
          call quadAreaByTriangleFacets( scscoords(1,1),
     .      area_vec(1,ielem,ics) )
        end do

      end do

      return
      end

      subroutine hex_scv_det( nelem, npe, nscv,
     .      cordel, vol, err, nerr )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the volume of each of the 8 subcontrol
c     volumes of the hex element for each element.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nscv          int   number of sub control volumes
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     vol           real  volume of each sub control volume
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      integer nelem, npe, nscv, nerr
      double precision cordel, vol, err
      dimension cordel(3,npe,nelem),
     .          vol(nelem,nscv),
     .          err(nelem)
c
      double precision coords, scvcoords, voltmp
      ! coordinates of each vertex defining the subcontrol volumes
      dimension coords(3,27)
      ! coordinates of the subcontrol volume
      dimension scvcoords(3,8)

      integer ielem, j, k, icv, inode
      ! this table defines the vertices that compose each scv
      integer HexSubcontrolNodeTable(8,8)
      data HexSubcontrolNodeTable /
     .  0, 8, 12, 11, 19, 20, 26, 25,
     .  8, 1, 9, 12, 20, 18, 24, 26,
     .  12, 9, 2, 10, 26, 24, 22, 23,
     .  11, 12, 10, 3, 25, 26, 23, 21,
     .  19, 20, 26, 25, 4, 13, 17, 16,
     .  20, 18, 24, 26, 13, 5, 14, 17,
     .  26, 24, 22, 23, 17, 14, 6, 15,
     .  25, 26, 23, 21, 16, 17, 15, 7 /

      ! loop over elements
      do ielem = 1,nelem
        ! element vertices
        do j = 1,8
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (front)
        ! 4++++11+++3
        ! +         +
        ! +         +
        ! 12   13   10
        ! +         +
        ! +         +
        ! 1++++9++++2

        ! edge midpoints
        do k = 1,3
          coords(k,9) = 0.5d0*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,10) = 0.5d0*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,11) = 0.5d0*(cordel(k,3,ielem) + cordel(k,4,ielem))
        end do
        do k = 1,3
          coords(k,12) = 0.5d0*(cordel(k,4,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,13) = 0.25d0*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem) + cordel(k,4,ielem))
        end do

        ! face 2 (back)
        ! 8++++16+++7
        ! +         +
        ! +         +
        ! 17   18   15
        ! +         +
        ! +         +
        ! 5++++14+++6

        ! edge midpoints
        do k = 1,3
          coords(k,14) = 0.5d0*(cordel(k,5,ielem) + cordel(k,6,ielem))
        end do
        do k = 1,3
          coords(k,15) = 0.5d0*(cordel(k,6,ielem) + cordel(k,7,ielem))
        end do
        do k = 1,3
          coords(k,16) = 0.5d0*(cordel(k,7,ielem) + cordel(k,8,ielem))
        end do
        do k = 1,3
          coords(k,17) = 0.5d0*(cordel(k,8,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,18) = 0.25d0*(cordel(k,5,ielem) + cordel(k,6,ielem)
     .      + cordel(k,7,ielem) + cordel(k,8,ielem))
        end do

        ! face 3 (bottom)
        ! 5++++14+++6
        ! +         +
        ! +         +
        ! 20   21   19
        ! +         +
        ! +         +
        ! 1++++9++++2

        ! edge midpoints
        do k = 1,3
          coords(k,19) = 0.5d0*(cordel(k,2,ielem) + cordel(k,6,ielem))
        end do
        do k = 1,3
          coords(k,20) = 0.5d0*(cordel(k,1,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,21) = 0.25d0*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,5,ielem) + cordel(k,6,ielem))
        end do

        ! face 4 (top)
        ! 8++++16+++7
        ! +         +
        ! +         +
        ! 22   24   23
        ! +         +
        ! +         +
        ! 4++++11+++3

        ! edge mipdoints
        do k = 1,3
          coords(k,22) = 0.5d0*(cordel(k,4,ielem) + cordel(k,8,ielem))
        end do
        do k = 1,3
          coords(k,23) = 0.5d0*(cordel(k,3,ielem) + cordel(k,7,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,24) = 0.25d0*(cordel(k,3,ielem) + cordel(k,4,ielem)
     .      + cordel(k,7,ielem) + cordel(k,8,ielem))
        end do

        ! face 5 (right)
        ! 3++++23+++7
        ! +         +
        ! +         +
        ! 10   25   15
        ! +         +
        ! +         +
        ! 2++++19+++6

        ! face midpoint
        do k = 1,3
          coords(k,25) = 0.25d0*(cordel(k,2,ielem) + cordel(k,3,ielem)
     .      + cordel(k,6,ielem) + cordel(k,7,ielem))
        end do

        ! face 6 (left)
        ! 4++++22+++8
        ! +         +
        ! +         +
        ! 12   26   18
        ! +         +
        ! +         +
        ! 1++++20+++5

        ! face midpoint
        do k = 1,3
          coords(k,26) = 0.25d0*(cordel(k,1,ielem) + cordel(k,4,ielem)
     .      + cordel(k,5,ielem) + cordel(k,8,ielem))
        end do

        ! volume centroid
        do k = 1,3
          coords(k,27) = 0.125d0*(
     .        cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem) + cordel(k,4,ielem)
     .      + cordel(k,5,ielem) + cordel(k,6,ielem)
     .      + cordel(k,7,ielem) + cordel(k,8,ielem) )
        end do

        ! loop over subcontrol volumes
        do icv = 1, nscv
          ! loop over each vertex of scv
          do inode = 1,8
            ! set coordinates of scv vertex using the node table
            do k = 1,3
              scvcoords(k,inode) = coords(k,
     .          HexSubcontrolNodeTable(inode,icv)+1)
            end do
          end do
          ! compute the volume using an equivalent polyhedron
          call hexVolumeByTriangleFacets(scvcoords(1,1),voltmp)
          vol(ielem,icv) = voltmp
          ! check for negative volume
          if ( vol(ielem,icv) < 0.0d0) then
            err(nelem) = 1.0d0
            nerr = ielem
          end if
        end do

      end do

      return
      end

      subroutine tet_scv_det( nelem, npe, nscv,
     .     cordel, vol, err, nerr )
c
c***********************************************************************
c***********************************************************************
c     
c description:
c     This  routine returns the volume of each of the 4 subcontrol
c     volumes of the tetrahedron element for each element.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nscv          int   number of sub control volumes
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     vol           real  volume of each sub control volume
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      integer nelem, npe, nscv, nerr
      double precision cordel, vol, err
      double precision coords, ehexcoords
      double precision one3rd, voltmp
      integer TetSubcontrolNodeTable
      
      integer ielem, j, k, icv, inode
      dimension cordel(3,npe,nelem), vol(nelem,nscv)
      dimension err(nelem)
      dimension TetSubcontrolNodeTable(8,4)

      ! coordinates of vertices needed to define scv
      dimension coords(3,15)
      ! coordinates defining scv
      dimension ehexcoords(3,8)

      ! this table defines the vertices that compose each scv
      data TetSubcontrolNodeTable /
     .  0, 4, 7, 6, 11, 13, 14, 12,
     .  1, 5, 7, 4, 9, 10, 14, 13,
     .  2, 6, 7, 5, 8, 12, 14, 10,
     .  3, 9, 13, 11, 8, 10, 14, 12 /

      one3rd = 1.d0/3.d0

      ! loop over elements
      do ielem = 1,nelem
        ! element vertices
        do j = 1,4
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (tri)

        ! edge midpoints
        do k = 1,3
          coords(k,5) = 0.5d0*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,6) = 0.5d0*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,7) = 0.5d0*(cordel(k,3,ielem) + cordel(k,1,ielem))
        end do

        ! face mipdoint
        do k = 1,3
          coords(k,8) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem))
        end do

        ! face 2 (tri)

        ! edge midpoints
        do k = 1,3
          coords(k,9) = 0.5d0*(cordel(k,3,ielem) + cordel(k,4,ielem))
        end do
        do k = 1,3
          coords(k,10) = 0.5d0*(cordel(k,4,ielem) + cordel(k,2,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,11) = one3rd*(cordel(k,2,ielem) + cordel(k,3,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! face 3 (tri)

        ! edge midpoint
        do k = 1,3
          coords(k,12) = 0.5d0*(cordel(k,1,ielem) + cordel(k,4,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,13) = one3rd*(cordel(k,1,ielem) + cordel(k,3,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! face 4 (tri)

        ! face midpoint
        do k = 1,3
          coords(k,14) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! element centroid
        do k = 1,3
          coords(k,15) = 0.0d0
        end do
        do j = 1, npe
          do k = 1,3
            coords(k,15) = coords(k,15) + 0.25d0*cordel(k,j,ielem)
          end do
        end do

        ! loop over subcontrol volumes
        do icv = 1, nscv
          ! loop over nodes of scv
          do inode = 1,8
            ! define scv coordinates using node table
            do k = 1,3
              ehexcoords(k,inode) = coords(k,
     .          TetSubcontrolNodeTable(inode,icv)+1)
            end do
          end do
          ! compute volume using an equivalent polyhedron
          call hexVolumeByTriangleFacets(ehexcoords(1,1),voltmp)
          vol(ielem,icv) = voltmp
          ! check for negative volume
          if ( vol(ielem,icv) < 0.0d0) then
            err(nelem) = 1.0d0
            nerr = ielem
          end if
        end do

      end do

      return
      end

      subroutine tet_scs_det( nelem, npe, nscs,
     *     cordel, area_vec )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 6
c     subcontrol surfaces of the tetrahedron.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nscs          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area_vec      real  area vector: A = Ax i + Ay j + Az k
c                                        = Area*(i + j + k)
c
c***********************************************************************
c
      implicit none
c
      integer nelem, npe, nscs
      double precision cordel, area_vec
      double precision coords, scscoords
      integer TetEdgeFacetTable, ielem, j, k, ics, inode, itrianglenode
      double precision half, one3rd, one4th

      dimension cordel(3,npe,nelem),
     *     area_vec(3,nelem,nscs)

      ! coordinates of vertices needed to describe scs
      dimension coords(3,15)
      ! coordinates of vertices of scs
      dimension scscoords(3,4)
      dimension TetEdgeFacetTable(4,6)

      data TetEdgeFacetTable /
     .  5, 8, 15, 14,    ! sc face 1 -- points from 1 -> 2
     .  8, 15, 11, 6,    ! sc face 2 -- points from 2 -> 3
     .  7, 13, 15, 8,    ! sc face 3 -- points from 1 -> 3
     .  12, 14, 15, 13,  ! sc face 4 -- points from 1 -> 4
     .  14, 10, 11, 15,  ! sc face 5 -- points from 2 -> 4
     .  11, 9, 13, 15  / ! sc face 6 -- points from 3 -> 4

      half = 0.5d0
      one3rd = 1.d0/3.d0
      one4th = 1.d0/4.d0

      ! loop over elements
      do ielem = 1,nelem
        ! element vertices
        do j = 1,4
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (tri)

        ! edge midpoints
        do k = 1,3
          coords(k,5) = half*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,6) = half*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,7) = half*(cordel(k,3,ielem) + cordel(k,1,ielem))
        end do

        ! face mipdoint
        do k = 1,3
          coords(k,8) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem))
        end do

        ! face 2 (tri)

        ! edge midpoints
        do k = 1,3
          coords(k,9) = half*(cordel(k,3,ielem) + cordel(k,4,ielem))
        end do
        do k = 1,3
          coords(k,10) = half*(cordel(k,4,ielem) + cordel(k,2,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,11) = one3rd*(cordel(k,2,ielem) + cordel(k,3,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! face 3 (tri)

        ! edge midpoint
        do k = 1,3
          coords(k,12) = half*(cordel(k,1,ielem) + cordel(k,4,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,13) = one3rd*(cordel(k,1,ielem) + cordel(k,3,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! face 4 (tri)

        ! face midpoint
        do k = 1,3
          coords(k,14) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! element centroid
        do k = 1,3
          coords(k,15) = 0.0d0
        end do
        do j = 1, npe
          do k = 1,3
            coords(k,15) = coords(k,15) + one4th*cordel(k,j,ielem)
          end do
        end do

        ! loop over subcontrol surface
        do ics = 1, nscs
          ! loop over nodes of scs
          do inode = 1,4
            ! set coordinates of scs vertex using node table
            itrianglenode = TetEdgeFacetTable(inode,ics)
            do k = 1,3
              scscoords(k,inode) = coords(k,itrianglenode)
            end do
          end do
          ! compute area vector using triangle decomposition
          call quadAreaByTriangleFacets( scscoords(1,1),
     .      area_vec(1,ielem,ics) )
        end do

      end do

      return
      end

      subroutine pyr_scs_det( nelem, npe, nscs,
     *    cordel, area_vec )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 7
c     subcontrol surfaces of the pyramid.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area_vec      real  area vector: A = Ax i + Ay j + Az k
c                                        = Area*(i + j + k)
c
c***********************************************************************
c
      implicit none

      integer nelem, npe, nscs
      double precision cordel, area_vec
      double precision coords, scscoords
      integer PyramidEdgeFacetTable, ielem, j, k, ics, inode
      integer itrianglenode
      double precision half, one4th, one3rd

      dimension cordel(3,npe,nelem)
      dimension area_vec(3,nelem,nscs)

      ! coordinates of vertices needed to define scs
      dimension coords(3,19)
      ! coordinates of vertices of scs
      dimension scscoords(3,4)

      ! this table defines the vertices of each scs
      dimension PyramidEdgeFacetTable(4,12)
      data PyramidEdgeFacetTable /
     .  6, 10, 19, 13,    ! sc face 1  -- points from 1 -> 2
     .  7, 10, 19, 15,    ! sc face 2  -- points from 2 -> 3
     .  8, 10, 19, 17,    ! sc face 3  -- points from 3 -> 4
     .  9, 18, 19, 10,    ! sc face 4  -- points from 1 -> 4
     .  13, 13, 19, 18,   ! sc face 5  -- points from 1 -> 5 I
     .  12, 13, 13, 18,   ! sc face 6  -- points from 1 -> 5 O
     .  15, 15, 19, 13,   ! sc face 7  -- points from 2 -> 5 I
     .  11, 15, 15, 13,   ! sc face 8  -- points from 2 -> 5 O
     .  17, 17, 19, 15,   ! sc face 9  -- points from 3 -> 5 I 
     .  14, 17, 17, 15,   ! sc face 10 -- points from 3 -> 5 O
     .  18, 18, 19, 17,   ! sc face 11 -- points from 4 -> 5 I 
     .  16, 18, 18, 17 /  ! sc face 12 -- points from 4 -> 5 O

      half = 0.5d0
      one3rd = 1.d0/3.d0
      one4th = 1.d0/4.d0

      ! loop over elements
      do ielem = 1,nelem
        ! element vertices
        do j = 1,5
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (quad)
        ! 4++++8+++3
        ! +         +
        ! +         +
        ! 9   10    7
        ! +         +
        ! +         +
        ! 1++++6++++2

        ! edge midpoints
        do k = 1,3
          coords(k,6) = half*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,7) = half*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,8) = half*(cordel(k,3,ielem) + cordel(k,4,ielem))
        end do
        do k = 1,3
          coords(k,9) = half*(cordel(k,4,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,10) = one4th*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem) + cordel(k,4,ielem))
        end do

        ! face 2 (tri)
        !
        ! edge midpoints
        do k = 1,3
          coords(k,11) = half*(cordel(k,2,ielem) + cordel(k,5,ielem))
        end do
        do k = 1,3
          coords(k,12) = half*(cordel(k,5,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,13) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,5,ielem))
        end do

        ! face 3 (tri)

        ! edge midpoint
        do k = 1,3
          coords(k,14) = half*(cordel(k,3,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,15) = one3rd*(cordel(k,2,ielem) + cordel(k,3,ielem)
     .      + cordel(k,5,ielem))
        end do

        ! face 4 (tri)

        ! edge midpoint
        do k = 1,3
          coords(k,16) = half*(cordel(k,4,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,17) = one3rd*(cordel(k,4,ielem) + cordel(k,5,ielem)
     .      + cordel(k,3,ielem))
        end do

        ! face 5 (tri)

        ! face midpoint
        do k = 1,3
          coords(k,18) = one3rd*(cordel(k,1,ielem) + cordel(k,5,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! element centroid
        do k = 1, 3
          coords(k,19) = 0.0d0
        end do
        do j = 1,npe
          do k = 1,3
            coords(k,19) = coords(k,19) + 0.2d0*cordel(k,j,ielem)
          end do
        end do

        ! loop over subcontrol surfaces
        do ics = 1, nscs
          ! loop over vertices of scs
          do inode = 1,4
            ! set coordinates of vertices using node table
            itrianglenode = PyramidEdgeFacetTable(inode,ics)
            do k = 1,3
              scscoords(k,inode) = coords(k,itrianglenode)
            end do
          end do
          ! compute area vector using triangle decomposition
          call quadAreaByTriangleFacets( scscoords(1,1),
     .      area_vec(1,ielem,ics) )
        end do

      end do

      return
      end

      subroutine pyr_scv_det( nelem, npe, nscv,
     .     cordel, vol, err, nerr )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the volume of each of the 5 subcontrol
c     volumes of the pyramid element for each element.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nscv          int   number of sub control volumes
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     vol           real  volume of each sub control volume
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      
      integer nelem, npe, nscv, nerr
      double precision cordel, vol, err
      double precision coords, ehexcoords, epyrcoords
      double precision voltmp, one3rd

      integer PyramidSubcontrolNodeTable, ielem, j, k, icv, inode
      
      dimension cordel(3,npe,nelem),
     .          vol(nelem,nscv),
     .          err(nelem)

      ! coordinates of vertices needed to define scv
      dimension coords(3,19)
      ! coordinates defining hexahedral scv
      dimension ehexcoords(3,8)
      ! coordinates defining octohedral scv
      dimension epyrcoords(3,10)

      ! thist table defines the vertices of each scv
      dimension PyramidSubcontrolNodeTable(10,5)
      data PyramidSubcontrolNodeTable /
     .  0,  5,  9,  8, 11, 12, 18, 17, -1, -1,
     .  1,  6,  9,  5, 10, 14, 18, 12, -1, -1,
     .  2,  7,  9,  6, 13, 16, 18, 14, -1, -1,
     .  3,  8,  9,  7, 15, 17, 18, 16, -1, -1,
     .  4, 18, 15, 17, 11, 12, 10, 14, 13, 16  /

      one3rd = 1.d0/3.d0

      ! loop over elements
      do ielem = 1,nelem
        ! element vertices
        do j = 1,5
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (quad)
        ! 4++++8+++3
        ! +         +
        ! +         +
        ! 9   10    7
        ! +         +
        ! +         +
        ! 1++++6++++2

        ! edge midpoints
        do k = 1,3
          coords(k,6) = 0.5d0*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,7) = 0.5d0*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,8) = 0.5d0*(cordel(k,3,ielem) + cordel(k,4,ielem))
        end do
        do k = 1,3
          coords(k,9) = 0.5d0*(cordel(k,4,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,10) = 0.25d0*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem) + cordel(k,4,ielem))
        end do

        ! face 2 (tri)
        !
        ! edge midpoints
        do k = 1,3
          coords(k,11) = 0.5d0*(cordel(k,2,ielem) + cordel(k,5,ielem))
        end do
        do k = 1,3
          coords(k,12) = 0.5d0*(cordel(k,5,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,13) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,5,ielem))
        end do

        ! face 3 (tri)

        ! edge midpoint
        do k = 1,3
          coords(k,14) = 0.5d0*(cordel(k,3,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,15) = one3rd*(cordel(k,2,ielem) + cordel(k,3,ielem)
     .      + cordel(k,5,ielem))
        end do

        ! face 4 (tri)

        ! edge midpoint
        do k = 1,3
          coords(k,16) = 0.5d0*(cordel(k,4,ielem) + cordel(k,5,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,17) = one3rd*(cordel(k,4,ielem) + cordel(k,5,ielem)
     .      + cordel(k,3,ielem))
        end do

        ! face 5 (tri)

        ! face midpoint
        do k = 1,3
          coords(k,18) = one3rd*(cordel(k,1,ielem) + cordel(k,5,ielem)
     .      + cordel(k,4,ielem))
        end do

        ! element centroid
        do k = 1, 3
          coords(k,19) = 0.0d0
        end do
        do j = 1,npe
          do k = 1,3
            coords(k,19) = coords(k,19) + 0.2d0*cordel(k,j,ielem)
          end do
        end do

        ! loop over hexahedral volumes first
        do icv = 1, nscv-1
          ! loop over vertices of hexahedral scv
          do inode = 1,8
            ! set coordinates of scv from node table
            do k = 1,3
              ehexcoords(k,inode) = coords(k,
     .          PyramidSubcontrolNodeTable(inode,icv)+1)
            end do
          end do
          ! compute volume use an equivalent polyhedron
          call bhexVolumeByTriangleFacets(ehexcoords(1,1),voltmp)
          vol(ielem,icv) = voltmp
          if ( vol(ielem,icv) < 0.0d0) then
            err(nelem) = 1.0d0
            nerr = ielem
          end if
        end do
        ! now do octohedron on pyramid tip
        icv = nscv
        ! loop over vertices of octohedral scv
        do inode = 1,10
          ! set coordinates based on node table
          do k = 1,3
            epyrcoords(k,inode) = coords(k,
     .         PyramidSubcontrolNodeTable(inode,icv)+1)
          end do
        end do
        ! compute volume using an equivalent polyhedron
        call octohedronVolumeByTriangleFacets(epyrcoords,voltmp)
        vol(ielem,icv) = voltmp
        if ( vol(ielem,icv) < 0.0d0) then
          err(nelem) = 1.0d0
          nerr = ielem
        end if

      end do

      return
      end

      subroutine wed_scs_det( nelem, npe, nscs,
     *    cordel, area_vec )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 9
c     subcontrol surfaces of the wedge.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nscs          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area_vec      real  area vector: A = Ax i + Ay j + Az k
c                                        = Area*(i + j + k)
c
c***********************************************************************
c
      implicit none
      integer nelem, npe, nscs
      double precision cordel, area_vec
      dimension cordel(3,npe,nelem),
     *          area_vec(3,nelem,nscs)

      double precision coords, scscoords
      ! coordinates of vertices needed to define scs
      dimension coords(3,21)
      ! coordinates of vertices of the scs
      dimension scscoords(3,4)

      integer ielem, k, j, ics, inodeTriangle, inode

      double precision one3rd, one6th

      ! table defining vertices of each scs
      integer WedgeEdgeFacetTable(4,9)
      data WedgeEdgeFacetTable /
     .  7, 10, 21, 17,      ! sc face 1 -- points from 1 -> 2
     .  8, 10, 21, 19,      ! sc face 2 -- points from 2 -> 3
     .  10, 9, 20, 21,      ! sc face 3 -- points from 1 -> 3
     .  11, 17, 21, 14,    ! sc face 4 -- points from 4 -> 5
     .  14, 12, 19, 21,    ! sc face 5 -- points from 5 -> 6
     .  13, 14, 21, 20,    ! sc face 6 -- points from 4 -> 6
     .  16, 17, 21, 20,    ! sc face 7 -- points from 1 -> 4
     .  17, 15, 19, 21,    ! sc face 8 -- points from 2 -> 5
     .  20, 21, 19, 18  /  ! sc face 9 -- points from 3 -> 6

      one3rd = 1.0d0/3.0d0
      one6th = 1.0d0/6.0d0

      ! loop over elements
      do ielem = 1,nelem

        ! the element vertices
        do j = 1,6
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (tri)

        ! edge mipdoints
        do k = 1,3
          coords(k,7) = 0.5d0*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,8) = 0.5d0*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,9) = 0.5d0*(cordel(k,3,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,10) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem))
        end do

        ! face 2 (tri)

        ! edge midpoints
        do k = 1,3
          coords(k,11) = 0.5d0*(cordel(k,4,ielem) + cordel(k,5,ielem))
        end do
        do k = 1,3
          coords(k,12) = 0.5d0*(cordel(k,5,ielem) + cordel(k,6,ielem))
        end do
        do k = 1,3
          coords(k,13) = 0.5d0*(cordel(k,6,ielem) + cordel(k,4,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,14) = one3rd*(cordel(k,4,ielem) + cordel(k,5,ielem)
     .      + cordel(k,6,ielem))
        end do

        ! face 3 (quad)

        ! edge midpoints
        do k = 1,3
          coords(k,15) = 0.5d0*(cordel(k,2,ielem) + cordel(k,5,ielem))
        end do
        do k = 1,3
          coords(k,16) = 0.5d0*(cordel(k,1,ielem) + cordel(k,4,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,17) = 0.25d0*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,5,ielem) + cordel(k,4,ielem))
        end do

        ! face 4 (quad)

        ! edge midpoint
        do k = 1,3
          coords(k,18) = 0.5d0*(cordel(k,3,ielem) + cordel(k,6,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,19) = 0.25d0*(cordel(k,2,ielem) + cordel(k,5,ielem)
     .      + cordel(k,6,ielem) + cordel(k,3,ielem))
        end do

        ! face 5 (quad)

        ! face midpoint
        do k = 1,3
          coords(k,20) = 0.25d0*(cordel(k,6,ielem) + cordel(k,4,ielem)
     .      + cordel(k,1,ielem) + cordel(k,3,ielem))
        end do

        ! element centroid
        do k = 1,3
          coords(k,21) = 0.0d0
        end do
        do j = 1, npe
          do k = 1,3
            coords(k,21) = coords(k,21)
     .        + one6th*cordel(k,j,ielem)
          end do
        end do

        ! loop over subcontrol surfaces
        do ics = 1, nscs
          ! loop over nodes of subcontrol surface
          do inode = 1,4
            ! set coordinates of vertex using node table
            inodetriangle = WedgeEdgeFacetTable(inode,ics)
            do k = 1,3
              scscoords(k,inode) = coords(k,inodetriangle)
            end do
          end do
          ! compute area vector using triangle decomposition
          call quadAreaByTriangleFacets( scscoords(1,1),
     .      area_vec(1,ielem,ics) )
        end do

      end do

      return
      end

      subroutine wed_scv_det( nelem, npe, nscv,
     .     cordel, vol, err, nerr )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the volume of each of the 6 subcontrol
c     volumes of the wedge element for each element.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nscv          int   number of sub control volumes
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     vol           real  volume of each sub control volume
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      integer nelem, npe, nscv, nerr
      double precision cordel, vol, err
      dimension cordel(3,npe,nelem),
     .          vol(nelem,nscv),
     .          err(nelem)

      double precision coords, ehexcoords
      ! coordinates of vertices needed to define scv
      dimension coords(3,21)
      ! coordinates defining scv
      dimension ehexcoords(3,8)

      integer ielem, j, k, icv, inode
      double precision voltmp
      double precision one3rd, one6th
      ! this table defines the vertices that compose each scv
      integer WedgeSubcontrolNodeTable(8,6)
      data WedgeSubcontrolNodeTable /
     .  0, 15, 16, 6, 8, 19, 20, 9,
     .  9, 6, 1, 7, 20, 16, 14, 18,
     .  8, 9, 7, 2, 19, 20, 18, 17,
     .  19, 15, 16, 20, 12, 3, 10, 13,
     .  20, 16, 14, 18, 13, 10, 4, 11,
     .  19, 20, 18, 17, 12, 13, 11, 5 /

      one3rd = 1.0d0/3.0d0
      one6th = 1.0d0/6.0d0

      ! loop over elements
      do ielem = 1,nelem

        ! the element vertices
        do j = 1,6
          do k = 1,3
            coords(k,j) = cordel(k,j,ielem)
          end do
        end do

        ! face 1 (tri)

        ! edge mipdoints
        do k = 1,3
          coords(k,7) = 0.5d0*(cordel(k,1,ielem) + cordel(k,2,ielem))
        end do
        do k = 1,3
          coords(k,8) = 0.5d0*(cordel(k,2,ielem) + cordel(k,3,ielem))
        end do
        do k = 1,3
          coords(k,9) = 0.5d0*(cordel(k,3,ielem) + cordel(k,1,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,10) = one3rd*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,3,ielem))
        end do

        ! face 2 (tri)

        ! edge midpoints
        do k = 1,3
          coords(k,11) = 0.5d0*(cordel(k,4,ielem) + cordel(k,5,ielem))
        end do
        do k = 1,3
          coords(k,12) = 0.5d0*(cordel(k,5,ielem) + cordel(k,6,ielem))
        end do
        do k = 1,3
          coords(k,13) = 0.5d0*(cordel(k,6,ielem) + cordel(k,4,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,14) = one3rd*(cordel(k,4,ielem) + cordel(k,5,ielem)
     .      + cordel(k,6,ielem))
        end do

        ! face 3 (quad)

        ! edge midpoints
        do k = 1,3
          coords(k,15) = 0.5d0*(cordel(k,2,ielem) + cordel(k,5,ielem))
        end do
        do k = 1,3
          coords(k,16) = 0.5d0*(cordel(k,1,ielem) + cordel(k,4,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,17) = 0.25d0*(cordel(k,1,ielem) + cordel(k,2,ielem)
     .      + cordel(k,5,ielem) + cordel(k,4,ielem))
        end do

        ! face 4 (quad)

        ! edge midpoint
        do k = 1,3
          coords(k,18) = 0.5d0*(cordel(k,3,ielem) + cordel(k,6,ielem))
        end do

        ! face midpoint
        do k = 1,3
          coords(k,19) = 0.25d0*(cordel(k,2,ielem) + cordel(k,5,ielem)
     .      + cordel(k,6,ielem) + cordel(k,3,ielem))
        end do

        ! face 5 (quad)

        ! face midpoint
        do k = 1,3
          coords(k,20) = 0.25d0*(cordel(k,6,ielem) + cordel(k,4,ielem)
     .      + cordel(k,1,ielem) + cordel(k,3,ielem))
        end do

        ! element centroid
        do k = 1,3
          coords(k,21) = 0.0d0
        end do
        do j = 1, npe
          do k = 1,3
            coords(k,21) = coords(k,21)
     .        + one6th*cordel(k,j,ielem)
          end do
        end do

        ! loop over subcontrol volumes
        do icv = 1, nscv
          ! loop over vertices of scv
          do inode = 1,8
            ! set coordinates of scv vertices using node table
            do k = 1,3
              ehexcoords(k,inode) = coords(k,
     .          WedgeSubcontrolNodeTable(inode,icv)+1)
            end do
          end do
          ! compute volume using an equivalent polyhedron
          call hexVolumeByTriangleFacets(ehexcoords(1,1),voltmp)
          vol(ielem,icv) = voltmp
          ! check for negative volume
          if ( vol(ielem,icv) < 0.0d0) then
            err(nelem) = 1.0d0
            nerr = ielem
          end if
        end do

      end do

      return
      end

      subroutine quad3d_scs_det( nelem, cordel, area )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 4
c     subcontrol surfaces of the quad.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area          real  area vector: A = Ax i + Ay j + Az k
c                                        = Area*(i + j + k)
c    
c***********************************************************************
c

      implicit none
      
c
      integer nelem
      integer k,i, idim
      double precision cordel, area
      double precision p, e, c, half, one4th
      double precision dx13, dx24, dy13, dy24, dz13, dz24
c
      dimension cordel(3,4,nelem)
      dimension area(3,nelem,4)
c
      dimension p(3,4), e(3,4), c(3)
c
c----------- first executable line of quad3d_scs_det ---------
c
      half = 0.5d0
      one4th = 1.d0/4.d0
c
c ... INITIALIZE FACE DEFINITION
c
      do 100 k=1,nelem
c
        do 20 i = 1, 4
          do 10 idim = 1, 3
            p(idim, i) = cordel(idim, i, k)
 10       continue
 20     continue
c
        do 30 idim = 1, 3
c
          e(idim, 1) = ( p(idim, 1) + p(idim, 2) ) * half
          e(idim, 2) = ( p(idim, 2) + p(idim, 3) ) * half
          e(idim, 3) = ( p(idim, 3) + p(idim, 4) ) * half
          e(idim, 4) = ( p(idim, 4) + p(idim, 1) ) * half
c
          c(idim) = ( p(idim, 1) + p(idim, 2) + 
     *                p(idim, 3) + p(idim, 4)   ) * one4th
c
 30     continue
c
c ... CALCULATE SUBCONTROL VOLUME FACE AREAS ...
c     ... subcontrol volume face 1
c
        dx13 = c(1)    - p(1, 1)
        dx24 = e(1, 4) - e(1, 1)
        dy13 = c(2)    - p(2, 1)
        dy24 = e(2, 4) - e(2, 1)
        dz13 = c(3)    - p(3, 1)
        dz24 = e(3, 4) - e(3, 1)
c
        area(1, k, 1) = half * ( dz24 * dy13 - dz13 * dy24 )
        area(2, k, 1) = half * ( dx24 * dz13 - dx13 * dz24 )
        area(3, k, 1) = half * ( dy24 * dx13 - dy13 * dx24 )
c
c     ... subcontrol volume face 2
c
        dx13 = c(1)    - p(1, 2)
        dx24 = e(1, 1) - e(1, 2)
        dy13 = c(2)    - p(2, 2)
        dy24 = e(2, 1) - e(2, 2)
        dz13 = c(3)    - p(3, 2)
        dz24 = e(3, 1) - e(3, 2)
c
        area(1, k, 2) = half * ( dz24 * dy13 - dz13 * dy24 )
        area(2, k, 2) = half * ( dx24 * dz13 - dx13 * dz24 )
        area(3, k, 2) = half * ( dy24 * dx13 - dy13 * dx24 )
c
c     ... subcontrol volume face 3
c
        dx13 = c(1)    - p(1, 3)
        dx24 = e(1, 2) - e(1, 3)
        dy13 = c(2)    - p(2, 3)
        dy24 = e(2, 2) - e(2, 3)
        dz13 = c(3)    - p(3, 3)
        dz24 = e(3, 2) - e(3, 3)
c
        area(1, k, 3) = half * ( dz24 * dy13 - dz13 * dy24 )
        area(2, k, 3) = half * ( dx24 * dz13 - dx13 * dz24 )
        area(3, k, 3) = half * ( dy24 * dx13 - dy13 * dx24 )
c
c     ... subcontrol volume face 4
c
        dx13 = c(1)    - p(1, 4)
        dx24 = e(1, 3) - e(1, 4)
        dy13 = c(2)    - p(2, 4)
        dy24 = e(2, 3) - e(2, 4)
        dz13 = c(3)    - p(3, 4)
        dz24 = e(3, 3) - e(3, 4)
c
        area(1, k, 4) = half * ( dz24 * dy13 - dz13 * dy24 )
        area(2, k, 4) = half * ( dx24 * dz13 - dx13 * dz24 )
        area(3, k, 4) = half * ( dy24 * dx13 - dy13 * dx24 )
c
  100 continue
c
c---------- last executable line of quad3d_scs_det ----------
c
      return
      end


      subroutine tri3d_scs_det( nelem, npe, nint,
     *    cordel, area )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 3
c     subcontrol surfaces of the triangle.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area          real  area vector: A = Ax i + Ay j + Az k
c                                        = Area*(i + j + k)
c    
c***********************************************************************
c
c
      implicit none
      integer nelem, npe, nint
      double precision cordel, area
c
      integer k, i, idim
      double precision p, e, c
      double precision half, one3rd
      double precision dx13, dx24, dy13, dy24, dz13, dz24
      
      integer, parameter :: nnodes = 3

      dimension cordel(3, npe, nelem)
      dimension area(3, nelem, nint)
      dimension p(3,3), e(3,3), c(3)
c
      half = 0.5d0
      one3rd = 1.d0/3.d0
c
c----------- first executable line of tri3d_scs_det. ---------
c
c ... INITIALIZE FACE DEFINITION
c
      do 100 k=1,nelem
c
        do 20 i = 1, nnodes
          do 10 idim = 1, 3
            p(idim, i) = cordel(idim, i, k)
 10       continue
 20     continue
c
        do 30 idim = 1, 3
c
          e(idim, 1) = ( p(idim, 1) + p(idim, 2) ) * half
          e(idim, 2) = ( p(idim, 2) + p(idim, 3) ) * half
          e(idim, 3) = ( p(idim, 3) + p(idim, 1) ) * half
c
          c(idim) = ( p(idim, 1) + p(idim, 2) + 
     *                p(idim, 3) ) * one3rd
c
 30     continue
c
c ... CALCULATE SUBCONTROL VOLUME FACE AREAS ...
c     ... subcontrol volume face 1
c
        dx13 = c(1)    - p(1, 1)
        dx24 = e(1, 3) - e(1, 1)
        dy13 = c(2)    - p(2, 1)
        dy24 = e(2, 3) - e(2, 1)
        dz13 = c(3)    - p(3, 1)
        dz24 = e(3, 3) - e(3, 1)
c
        area(1, k, 1) = half * ( dz24 * dy13 - dz13 * dy24 )
        area(2, k, 1) = half * ( dx24 * dz13 - dx13 * dz24 )
        area(3, k, 1) = half * ( dy24 * dx13 - dy13 * dx24 )
c
c     ... subcontrol volume face 2
c
        dx13 = c(1)    - p(1, 2)
        dx24 = e(1, 1) - e(1, 2)
        dy13 = c(2)    - p(2, 2)
        dy24 = e(2, 1) - e(2, 2)
        dz13 = c(3)    - p(3, 2)
        dz24 = e(3, 1) - e(3, 2)
c
        area(1, k, 2) = half * ( dz24 * dy13 - dz13 * dy24 )
        area(2, k, 2) = half * ( dx24 * dz13 - dx13 * dz24 )
        area(3, k, 2) = half * ( dy24 * dx13 - dy13 * dx24 )
c
c     ... subcontrol volume face 3
c
        dx13 = c(1)    - p(1, 3)
        dx24 = e(1, 2) - e(1, 3)
        dy13 = c(2)    - p(2, 3)
        dy24 = e(2, 2) - e(2, 3)
        dz13 = c(3)    - p(3, 3)
        dz24 = e(3, 2) - e(3, 3)
c
        area(1, k, 3) = half * ( dz24 * dy13 - dz13 * dy24 )
        area(2, k, 3) = half * ( dx24 * dz13 - dx13 * dz24 )
        area(3, k, 3) = half * ( dy24 * dx13 - dy13 * dx24 )
c
  100 continue
c
c---------- last executable line of tri3d_scs_det ----------
c
      return
      end

      subroutine edge2d_scs_det( nelem, npe, nint,
     *    cordel, area )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 2
c     subcontrol surfaces of the edge.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area          real  area vector: A = Ax i + Ay j + Az k
c                                        = Area*(i + j + k)
c
c***********************************************************************
c
c
      implicit none
      integer nelem, npe, nint
      double precision cordel, area
c
      integer k, i, idim
      double precision p, c
      double precision half
      double precision dx13, dy13
      
      integer, parameter :: nnodes = 2

      dimension cordel(2, npe, nelem)
      dimension area(2, nelem, nint)
      dimension p(2,2), c(2)
c
      half = 0.5d0

c
c----------- first executable line of edge2d_scs_det. ---------
c
c ... INITIALIZE FACE DEFINITION
c
      do k=1,nelem
c
        do i = 1, nnodes
          do idim = 1, 2
            p(idim, i) = cordel(idim, i, k)
          end do
        end do
c
        do idim = 1, 2
          c(idim) = ( p(idim, 1) + p(idim, 2) ) * half
        end do
c
c ... CALCULATE SUBCONTROL VOLUME FACE AREAS ...
c     ... subcontrol volume face 1
c
        dx13 = cordel(1,1,k) - c(1)
        dy13 = cordel(2,1,k) - c(2)
c
        area(1, k, 1) = -dy13
        area(2, k, 1) = dx13
c
c     ... subcontrol volume face 2
c
        dx13 = cordel(1,2,k) - c(1)
        dy13 = cordel(2,2,k) - c(2)
c
        area(1, k, 2) = dy13
        area(2, k, 2) = -dx13
c
      end do
c
c---------- last executable line of edge2d_scs_det ----------
c
      return
      end

c
c     Copyright 2002 Sandia Corporation, Albuquerque, NM.
c
c***********************************************************************
c***********************************************************************
      subroutine quad_scv_det( nelem, npe, nint,
     .                         cordel, vol, err,
     .                         nerr )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the volume of each of the 4 subcontrol
c     volumes of the quad element.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control volumes
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     vol           real  volume of each sub control volume
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c log:
c
c***********************************************************************
c
c
      implicit none
      integer nelem, npe, nint, nerr
      dimension cordel(2,npe,nelem),
     .          vol(nelem,nint),
     .          err(nelem)
      dimension deriv(2,4), shape_fcn(4)
      dimension xi(2,4), xigp(2,4)
      double precision cordel, vol, err
      double precision deriv, shape_fcn
      double precision xi, xigp
c
c     Gaussian quadrature points within an interval [-.25,+.25]
c
      double precision gpp, gpm, cvm, cvp
      parameter (gpp =  0.144337567d0)
      parameter (gpm = -0.144337567d0)
      parameter (cvm = -0.25d0)
      parameter (cvp =  0.25d0)
      double precision dx_ds1, dx_ds2, dy_ds1, dy_ds2
      double precision ximod, etamod, det_j, test
      double precision one16th, one, half, zero
      double precision a1, a2, a3, sum
      parameter (one = 1.0d0)
      parameter (half = 0.5d0)
      parameter (zero = 0.0d0)
      parameter (one16th = 0.0625d0)

      integer ki, ke, kq, kn, kx, ky
c
c     store sub-volume centroids
c
      save xi, xigp
      data xi   / cvm, cvm, cvp, cvm,
     .            cvp, cvp, cvm, cvp /
      data xigp / gpm, gpm, gpp, gpm,
     .            gpp, gpp, gpm, gpp /
c
      kx = 1
      ky = 2
c
c     Cartesian
       a1 = one
       a2 = zero
       a3 = zero

      do ke = 1,nelem
         err(ke) = zero
      end do
c
c
c     2d cartesian, no cross-section area
c
      do ki = 1,nint
        do ke = 1,nelem
c
         vol(ke,ki) = zero
c
         do kq = 1,nint
            dx_ds1 = zero
            dx_ds2 = zero
            dy_ds1 = zero
            dy_ds2 = zero
c
            ximod  = xi(1,ki) + xigp(1,kq)
            etamod = xi(2,ki) + xigp(2,kq)
c
            deriv(1,1) = -(half - etamod)
            deriv(1,2) =  (half - etamod)
            deriv(1,3) =  (half + etamod)
            deriv(1,4) = -(half + etamod)
c
            deriv(2,1) = -(half - ximod)
            deriv(2,2) = -(half + ximod)
            deriv(2,3) =  (half + ximod)
            deriv(2,4) =  (half - ximod)
c
            shape_fcn(1) = (half - ximod)*(half - etamod)
            shape_fcn(2) = (half + ximod)*(half - etamod)
            shape_fcn(3) = (half + ximod)*(half + etamod)
            shape_fcn(4) = (half - ximod)*(half + etamod)
c
c calculate the jacobian at the integration station -
c
            do kn = 1,npe
c
               dx_ds1 = dx_ds1+deriv(1,kn)*cordel(1,kn,ke)
               dx_ds2 = dx_ds2+deriv(2,kn)*cordel(1,kn,ke)
c
               dy_ds1 = dy_ds1+deriv(1,kn)*cordel(2,kn,ke)
               dy_ds2 = dy_ds2+deriv(2,kn)*cordel(2,kn,ke)
c
            end do
c
c calculate the determinate of the jacobian at the integration station -
c
            det_j = (dx_ds1*dy_ds2 - dy_ds1*dx_ds2)
            test = det_j
            if( test .le. zero ) then
               err(ke) = one
            end if
c
            vol(ke,ki) = vol(ke,ki) + det_j*one16th
c
         end do
        end do
      end do
c
c summarize volume error checks -
c
      sum = zero
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if ( sum .ne. zero ) then
c
c flag error -
c
         do ke = 1 , nelem
            if ( err(ke) .ne. zero ) nerr = ke
         end do
      end if
c
      return
      end

c
c     Copyright 2002 Sandia Corporation, Albuquerque, NM.
c
c***********************************************************************
c***********************************************************************
      subroutine quad_scs_det( nelem, npe, nint,
     *    cordel, area_vec )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 4
c     subcontrol surfaces of the quad.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area_vec      real  area vector: A = Ax i + Ay j
c                                        = Area*(i + j)
c
c***********************************************************************
c
      implicit none
c
      integer nelem, npe, nint
      dimension cordel(2,npe,nelem),
     *          area_vec(2,nelem,nint)
      double precision cordel, area_vec
c
      dimension coord_mid_face(2,4)
      double precision coord_mid_face

      integer k, kx, ky
      double precision a1, a2, a3, x1, y1, x2, y2, rr

      double precision zero, one, half, one4th
      parameter (zero = 0.0d0)
      parameter (one = 1.0d0)
      parameter (half = 0.5d0)
      parameter (one4th = 0.25d0)
c
      kx = 1
      ky = 2
c
c     Cartesian
      a1 = one
      a2 = zero
      a3 = zero
c
      do k = 1, nelem
c
c calculate element mid-point coordinates
         x1 = ( cordel(kx,1,k) + cordel(kx,2,k)
     *        + cordel(kx,3,k) + cordel(kx,4,k) ) * one4th
         y1 = ( cordel(ky,1,k) + cordel(ky,2,k)
     *        + cordel(ky,3,k) + cordel(ky,4,k) ) * one4th
c calculate element mid-face coordinates
         coord_mid_face(kx,1) = ( cordel(kx,1,k)+cordel(kx,2,k) )*half
         coord_mid_face(kx,2) = ( cordel(kx,2,k)+cordel(kx,3,k) )*half
         coord_mid_face(kx,3) = ( cordel(kx,3,k)+cordel(kx,4,k) )*half
         coord_mid_face(kx,4) = ( cordel(kx,4,k)+cordel(kx,1,k) )*half
         coord_mid_face(ky,1) = ( cordel(ky,1,k)+cordel(ky,2,k) )*half
         coord_mid_face(ky,2) = ( cordel(ky,2,k)+cordel(ky,3,k) )*half
         coord_mid_face(ky,3) = ( cordel(ky,3,k)+cordel(ky,4,k) )*half
         coord_mid_face(ky,4) = ( cordel(ky,4,k)+cordel(ky,1,k) )*half
c
c Control surface 1
         x2 = coord_mid_face(kx,1)
         y2 = coord_mid_face(ky,1)
c
         rr = a1 + a2*(x1+x2) + a3*(y1+y2)
c
         area_vec(kx, k, 1) = -(y2 - y1)*rr
         area_vec(ky, k, 1) =  (x2 - x1)*rr
c
c Control surface 2
         x2 = coord_mid_face(kx,2)
         y2 = coord_mid_face(ky,2)
c
         rr = a1 + a2*(x1+x2) + a3*(y1+y2)
c
         area_vec(kx, k, 2) = -(y2 - y1)*rr
         area_vec(ky, k, 2) =  (x2 - x1)*rr
c
c Control surface 3
         x2 = coord_mid_face(kx,3)
         y2 = coord_mid_face(ky,3)
c
         rr = a1 + a2*(x1+x2) + a3*(y1+y2)
c
         area_vec(kx, k, 3) = -(y2 - y1)*rr
         area_vec(ky, k, 3) =  (x2 - x1)*rr
c
c Control surface 4
         x2 = coord_mid_face(kx,4)
         y2 = coord_mid_face(ky,4)
c
         rr = a1 + a2*(x1+x2) + a3*(y1+y2)
c
         area_vec(kx, k, 4) =  (y2 - y1)*rr
         area_vec(ky, k, 4) = -(x2 - x1)*rr
c
      end do
c
      return
      end

c***********************************************************************
      subroutine tri_scs_det( nelem, npe, nint,
     *    cordel, area_vec )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the area vectors for each of the 3
c     subcontrol surfaces of the tri.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     area_vec      real  area vector: A = Ax i + Ay j
c                                        = Area*(i + j)
c
c***********************************************************************
c
      implicit none
c
      integer nelem, npe, nint
      dimension cordel(2,npe,nelem),
     *          area_vec(2,nelem,nint)
      double precision cordel, area_vec
c
      dimension coord_mid_face(2,3)
      double precision coord_mid_face

      integer k, kx, ky
      double precision a1, a2, a3, x1, y1, x2, y2, rr
      double precision one, zero, half, one3rd
      parameter (one = 1.0d0)
      parameter (zero = 0.0d0)
      parameter (half = 0.5d0)

      one3rd = 1.0d0/3.0d0
c
      kx = 1
      ky = 2
c
c     Cartesian
      a1 = one
      a2 = zero
      a3 = zero
c
      do k = 1, nelem
c
c calculate element mid-point coordinates
         x1 = ( cordel(kx,1,k) + cordel(kx,2,k)
     *        + cordel(kx,3,k) ) * one3rd
         y1 = ( cordel(ky,1,k) + cordel(ky,2,k)
     *        + cordel(ky,3,k) ) * one3rd
c calculate element mid-face coordinates
         coord_mid_face(kx,1) = ( cordel(kx,1,k)+cordel(kx,2,k) )*half
         coord_mid_face(kx,2) = ( cordel(kx,2,k)+cordel(kx,3,k) )*half
         coord_mid_face(kx,3) = ( cordel(kx,3,k)+cordel(kx,1,k) )*half
         coord_mid_face(ky,1) = ( cordel(ky,1,k)+cordel(ky,2,k) )*half
         coord_mid_face(ky,2) = ( cordel(ky,2,k)+cordel(ky,3,k) )*half
         coord_mid_face(ky,3) = ( cordel(ky,3,k)+cordel(ky,1,k) )*half
c
c Control surface 1
         x2 = coord_mid_face(kx,1)
         y2 = coord_mid_face(ky,1)
c
         rr = a1 + a2*(x1+x2) + a3*(y1+y2)
c
         area_vec(kx, k, 1) = -(y2 - y1)*rr
         area_vec(ky, k, 1) =  (x2 - x1)*rr
c
c Control surface 2
         x2 = coord_mid_face(kx,2)
         y2 = coord_mid_face(ky,2)
c
         rr = a1 + a2*(x1+x2) + a3*(y1+y2)
c
         area_vec(kx, k, 2) = -(y2 - y1)*rr
         area_vec(ky, k, 2) =  (x2 - x1)*rr
c
c Control surface 3
         x2 = coord_mid_face(kx,3)
         y2 = coord_mid_face(ky,3)
c
         rr = a1 + a2*(x1+x2) + a3*(y1+y2)
c
         area_vec(kx, k, 3) =  (y2 - y1)*rr
         area_vec(ky, k, 3) = -(x2 - x1)*rr
c
      end do
c
      return
      end

c
c     Copyright 2002 Sandia Corporation, Albuquerque, NM.
c
c***********************************************************************
c***********************************************************************
      subroutine tri_scv_det( nelem, npe, nint,
     .                        cordel, vol, err,
     .                        nerr )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the volume of each of the 3 subcontrol
c     volumes of the tri element.
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control volumes
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     vol           real  volume of each sub control volume
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c log:
c
c***********************************************************************
c
      implicit none
c
      integer nelem, npe, nint, nerr
      dimension cordel(2,npe,nelem),
     .          vol(nelem,nint),
     .          err(nelem)
      double precision cordel, vol, err

      dimension deriv(2,4), shape_fcn(4)
      dimension xyval(2,4,3)
      dimension xigp(2,4)

      double precision deriv, shape_fcn, xyval, xigp, gpp, gpm
c
c     Gaussian quadrature points within an interval [-.5,+.5]
c
      parameter (gpp =  0.288675134d0)
      parameter (gpm = -0.288675134d0)

      integer ke, kx, ky, kn, ki, kq
      double precision a1, a2, a3, xc, yc, det_j, test, etamod, ximod
      double precision zero, one, half, one3rd, one4th
      double precision dx_ds1, dx_ds2, dy_ds1, dy_ds2, sum

      parameter (zero = 0.0d0)
      parameter (one = 1.0d0)
      parameter (half = 0.5d0)
      parameter (one4th = 0.25d0)

c
c     store sub-volume centroids
c
      save xigp
      data xigp / gpm, gpm, gpp, gpm,
     .            gpp, gpp, gpm, gpp /
c
      one3rd = 1.0d0/3.0d0
      kx = 1
      ky = 2
c
c     Cartesian
      a1 = one
      a2 = zero
      a3 = zero
c
      do ke = 1,nelem
         err(ke) = zero
      end do
c
c
c      2d cartesian, no cross-section area
c
       do ke = 1,nelem
c
        xc = one3rd*(cordel(kx,1,ke)+cordel(kx,2,ke)+cordel(kx,3,ke))
        yc = one3rd*(cordel(ky,1,ke)+cordel(ky,2,ke)+cordel(ky,3,ke))
c
c       sub-volume 1
c
        xyval(kx,1,1) = cordel(kx,1,ke)
        xyval(kx,2,1) = half*(cordel(kx,1,ke)+cordel(kx,2,ke))
        xyval(kx,3,1) = xc
        xyval(kx,4,1) = half*(cordel(kx,3,ke)+cordel(kx,1,ke))
c
        xyval(ky,1,1) = cordel(ky,1,ke)
        xyval(ky,2,1) = half*(cordel(ky,1,ke)+cordel(ky,2,ke))
        xyval(ky,3,1) = yc
        xyval(ky,4,1) = half*(cordel(ky,3,ke)+cordel(ky,1,ke))
c
c       sub-volume 2
c
        xyval(kx,1,2) = cordel(kx,2,ke)
        xyval(kx,2,2) = half*(cordel(kx,2,ke)+cordel(kx,3,ke))
        xyval(kx,3,2) = xc
        xyval(kx,4,2) = half*(cordel(kx,1,ke)+cordel(kx,2,ke))
c
        xyval(ky,1,2) = cordel(ky,2,ke)
        xyval(ky,2,2) = half*(cordel(ky,2,ke)+cordel(ky,3,ke))
        xyval(ky,3,2) = yc
        xyval(ky,4,2) = half*(cordel(ky,1,ke)+cordel(ky,2,ke))
c
c       sub-volume 3
c
        xyval(kx,1,3) = cordel(kx,3,ke)
        xyval(kx,2,3) = half*(cordel(kx,3,ke)+cordel(kx,1,ke))
        xyval(kx,3,3) = xc
        xyval(kx,4,3) = half*(cordel(kx,2,ke)+cordel(kx,3,ke))
c
        xyval(ky,1,3) = cordel(ky,3,ke)
        xyval(ky,2,3) = half*(cordel(ky,3,ke)+cordel(ky,1,ke))
        xyval(ky,3,3) = yc
        xyval(ky,4,3) = half*(cordel(ky,2,ke)+cordel(ky,3,ke))
c
        do ki = 1,nint
c
         vol(ke,ki) = zero
c
         do kq = 1,4
            dx_ds1 = zero
            dx_ds2 = zero
            dy_ds1 = zero
            dy_ds2 = zero
c
            ximod  = xigp(1,kq)
            etamod = xigp(2,kq)
c
            deriv(1,1) = -(half - etamod)
            deriv(1,2) =  (half - etamod)
            deriv(1,3) =  (half + etamod)
            deriv(1,4) = -(half + etamod)
c
            deriv(2,1) = -(half - ximod)
            deriv(2,2) = -(half + ximod)
            deriv(2,3) =  (half + ximod)
            deriv(2,4) =  (half - ximod)
c
            shape_fcn(1) = (half - ximod)*(half - etamod)
            shape_fcn(2) = (half + ximod)*(half - etamod)
            shape_fcn(3) = (half + ximod)*(half + etamod)
            shape_fcn(4) = (half - ximod)*(half + etamod)
c
c calculate the jacobian at the integration station -
c
            do kn = 1,4
c
               dx_ds1 = dx_ds1+deriv(1,kn)*xyval(kx,kn,ki)
               dx_ds2 = dx_ds2+deriv(2,kn)*xyval(kx,kn,ki)
c
               dy_ds1 = dy_ds1+deriv(1,kn)*xyval(ky,kn,ki)
               dy_ds2 = dy_ds2+deriv(2,kn)*xyval(ky,kn,ki)
c
            end do
c
c calculate the determinate of the jacobian at the integration station -
c
            det_j = (dx_ds1*dy_ds2 - dy_ds1*dx_ds2)
            test = det_j
            if( test .le. zero ) then
               err(ke) = one
            end if
c
            vol(ke,ki) = vol(ke,ki) + det_j*one4th
c
         end do
        end do
       end do
c
c
c summarize volume error checks -
c
      sum = zero
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if ( sum .ne. zero ) then
c
c flag error -
c
         do ke = 1 , nelem
            if ( err(ke) .ne. zero ) nerr = ke
         end do
      end if
c
      return
      end

      subroutine hex_shape_fcn( npts, par_coord, shape_fcn )
c
c***********************************************************************
c***********************************************************************
c
c formal parameters - input:
c     npts          int   number of points to evaluate (usually
c                         the number of Gauss Points)
c     par_coord     real  parametric coordinates of the points to be
c                         evaluated (typically, the gauss pts) -1:1
c
c formal parameters - output:
c     shape_fcn     real  shape functions evaluated at the evaluation
c                         points
c
c***********************************************************************
      implicit none

      integer npts

      double precision par_coord
      double precision shape_fcn

      dimension par_coord(3,npts)
      dimension shape_fcn(8,npts)

      integer j
      double precision s1, s2, s3, half, one4th, one8th
c
      half = 1.d0/2.d0
      one4th = 1.d0/4.d0
      one8th = 1.d0/8.d0
c
      do j = 1,npts
         s1 = par_coord(1,j)
         s2 = par_coord(2,j)
         s3 = par_coord(3,j)
c
         shape_fcn(1,j) = one8th + one4th*(-s1 - s2 - s3)
     *                  + half*( s2*s3 + s3*s1 + s1*s2 ) - s1*s2*s3
         shape_fcn(2,j) = one8th + one4th*( s1 - s2 - s3)
     *                  + half*( s2*s3 - s3*s1 - s1*s2 ) + s1*s2*s3
         shape_fcn(3,j) = one8th + one4th*( s1 + s2 - s3)
     *                  + half*(-s2*s3 - s3*s1 + s1*s2 ) - s1*s2*s3
         shape_fcn(4,j) = one8th + one4th*(-s1 + s2 - s3)
     *                  + half*(-s2*s3 + s3*s1 - s1*s2 ) + s1*s2*s3
         shape_fcn(5,j) = one8th + one4th*(-s1 - s2 + s3)
     *                  + half*(-s2*s3 - s3*s1 + s1*s2 ) + s1*s2*s3
         shape_fcn(6,j) = one8th + one4th*( s1 - s2 + s3)
     *                  + half*(-s2*s3 + s3*s1 - s1*s2 ) - s1*s2*s3
         shape_fcn(7,j) = one8th + one4th*( s1 + s2 + s3)
     *                  + half*( s2*s3 + s3*s1 + s1*s2 ) + s1*s2*s3
         shape_fcn(8,j) = one8th + one4th*(-s1 + s2 + s3)
     *                  + half*( s2*s3 - s3*s1 - s1*s2 ) - s1*s2*s3
c
      end do
c
      return
      end     

      subroutine hex_derivative( npts, par_coord, deriv )
c
c***********************************************************************
c***********************************************************************
c
c formal parameters - input:
c     npts          int   number of points to evaluate (usually
c                         the number of Gauss Points)
c     ncoord        int   number of parametric coordinates in element
c     par_coord     real  parametric coordinates of the points to be
c                         evaluated (typically, the gauss pts)
c
c formal parameters - output:
c     deriv         real  shape function derivatives evaluated at
c                         evaluation points.
c
c***********************************************************************
c
c
      implicit none

      integer npts
      double precision par_coord, deriv

      dimension par_coord(3,npts)
      dimension deriv(3,8,npts)
c
      integer j

      double precision s1, s2, s3

      double precision s1s2
      double precision s2s3
      double precision s1s3
c
      double precision half, one4th
      half = 1.d0/2.d0
      one4th = 1.d0/4.d0
c
      do j = 1,npts
         s1 = par_coord(1,j)
         s2 = par_coord(2,j)
         s3 = par_coord(3,j)
c
         s1s2 = s1*s2
         s2s3 = s2*s3
         s1s3 = s1*s3
c shape function derivative in the s1 direction -
         deriv(1,1,j) = half*( s3 + s2 ) - s2s3 - one4th
         deriv(1,2,j) = half*(-s3 - s2 ) + s2s3 + one4th
         deriv(1,3,j) = half*(-s3 + s2 ) - s2s3 + one4th
         deriv(1,4,j) = half*(+s3 - s2 ) + s2s3 - one4th
         deriv(1,5,j) = half*(-s3 + s2 ) + s2s3 - one4th
         deriv(1,6,j) = half*(+s3 - s2 ) - s2s3 + one4th
         deriv(1,7,j) = half*(+s3 + s2 ) + s2s3 + one4th
         deriv(1,8,j) = half*(-s3 - s2 ) - s2s3 - one4th
c
c shape function derivative in the s2 direction -
         deriv(2,1,j) = half*( s3 + s1 ) - s1s3 - one4th
         deriv(2,2,j) = half*( s3 - s1 ) + s1s3 - one4th
         deriv(2,3,j) = half*(-s3 + s1 ) - s1s3 + one4th
         deriv(2,4,j) = half*(-s3 - s1 ) + s1s3 + one4th
         deriv(2,5,j) = half*(-s3 + s1 ) + s1s3 - one4th
         deriv(2,6,j) = half*(-s3 - s1 ) - s1s3 - one4th
         deriv(2,7,j) = half*( s3 + s1 ) + s1s3 + one4th
         deriv(2,8,j) = half*( s3 - s1 ) - s1s3 + one4th
c
c shape function derivative in the s3 direction -
         deriv(3,1,j) = half*( s2 + s1 ) - s1s2 - one4th
         deriv(3,2,j) = half*( s2 - s1 ) + s1s2 - one4th
         deriv(3,3,j) = half*(-s2 - s1 ) - s1s2 - one4th
         deriv(3,4,j) = half*(-s2 + s1 ) + s1s2 - one4th
         deriv(3,5,j) = half*(-s2 - s1 ) + s1s2 + one4th
         deriv(3,6,j) = half*(-s2 + s1 ) - s1s2 + one4th
         deriv(3,7,j) = half*( s2 + s1 ) + s1s2 + one4th
         deriv(3,8,j) = half*( s2 - s1 ) - s1s2 + one4th
c
      end do
c
      return
      end

      subroutine quad_derivative( npts, par_coord, deriv )
c***********************************************************************
c
c formal parameters - input:
c     npts          int   number of points to evaluate (usually
c                         the number of Gauss Points)
c     ncoord        int   number of parametric coordinates in element
c     par_coord     real  parametric coordinates of the points to be
c                         evaluated (typically, the gauss pts)
c
c formal parameters - output:
c     deriv         real  shape function derivatives evaluated at
c                         evaluation points.
c
c***********************************************************************
c
c
      implicit none

      integer npts
      double precision par_coord, deriv

      dimension par_coord(2,npts)
      dimension deriv(2,4,npts)

      integer j

      double precision s1, s2

      double precision half

      half = 0.5d0
c
      do j = 1,npts
         s1 = par_coord(1,j)
         s2 = par_coord(2,j)
c shape function derivative in the s1 direction -
         deriv(1,1,j) = - half + s2  
         deriv(1,2,j) =   half - s2 
         deriv(1,3,j) =   half + s2 
         deriv(1,4,j) = - half - s2 
c
c shape function derivative in the s2 direction -
         deriv(2,1,j) = - half + s1
         deriv(2,2,j) = - half - s1
         deriv(2,3,j) =   half + s1
         deriv(2,4,j) =   half - s1
c
      end do
c
      return
      end

      subroutine tet_derivative( npts, deriv )
c
c***********************************************************************
c***********************************************************************
c
c formal parameters - input:
c     npts          int   number of points to evaluate (usually
c                         the number of Gauss Points)
c
c formal parameters - output:
c     deriv         real  shape function derivatives evaluated at
c                         evaluation points.
c
c***********************************************************************
c
c
      implicit none

      integer npts
      double precision deriv
      dimension deriv(3,4,npts)
c
      integer j

      do j = 1, npts
         deriv(1,1,j) = -1.0
         deriv(2,1,j) = -1.0
         deriv(3,1,j) = -1.0
         
         deriv(1,2,j) = 1.0
         deriv(2,2,j) = 0.0
         deriv(3,2,j) = 0.0
         
         deriv(1,3,j) = 0.0
         deriv(2,3,j) = 1.0
         deriv(3,3,j) = 0.0
         
         deriv(1,4,j) = 0.0
         deriv(2,4,j) = 0.0
         deriv(3,4,j) = 1.0
      end do
      
      return;
      end

      subroutine tri_derivative( npts, deriv )
c
c***********************************************************************
c***********************************************************************
c
c formal parameters - input:
c     npts          int   number of points to evaluate (usually
c                         the number of Gauss Points)
c
c formal parameters - output:
c     deriv         real  shape function derivatives evaluated at
c                         evaluation points.
c
c***********************************************************************
c
c
      implicit none

      integer npts
      double precision deriv

      dimension deriv(2,3,npts)
c
      integer j

      do j = 1, npts
         deriv(1,1,j) = -1.0
         deriv(1,2,j) =  1.0
         deriv(1,3,j) =  0.0
         
         deriv(2,1,j) = -1.0
         deriv(2,2,j) =  0.0
         deriv(2,3,j) =  1.0
         
      end do
      
      return;
      end

      subroutine hex_gradient_operator( nelem, npe, 
     *    nint, deriv, cordel, gradop, det_j, err, nerr )
c       
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the gradient operator, determinate of 
c     the Jacobian, and error count for an element workset of 3D 
c     subcontrol surface elements The gradient operator and the 
c     determinate of the jacobians are computed at the center of
c     each control surface (the locations for the integration rule
c     are at the center of each control surface).
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     deriv         real  shape function derivatives evaluated at the
c                         integration stations
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     gradop        real  element gradient operator at each integration
c                         station
c     det_j         real  determinate of the jacobian at each integration
c                         station
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      
      integer nelem, npe, nint, nerr
      double precision deriv, cordel, gradop, det_j, err
c
      dimension deriv(3,npe,nint),
     *     cordel(3,npe,nelem),
     *     gradop(3,npe,nelem,nint),
     *     det_j(nelem,nint),
     *     err(nelem)
c
      integer ke, ki, kn
      double precision dx_ds1, dx_ds2, dx_ds3
      double precision dy_ds1, dy_ds2, dy_ds3
      double precision dz_ds1, dz_ds2, dz_ds3
      double precision ds1_dx, ds1_dy, ds1_dz
      double precision ds2_dx, ds2_dy, ds2_dz
      double precision ds3_dx, ds3_dy, ds3_dz

      double precision test, denom, sum, realmin
      realmin = 2.2250738585072014d-308
c
      do ke = 1,nelem
         err(ke) = 0.d0
      end do
c
      do ki = 1,nint
         do ke = 1,nelem
            dx_ds1 = 0.d0
            dx_ds2 = 0.d0
            dx_ds3 = 0.d0
            dy_ds1 = 0.d0
            dy_ds2 = 0.d0
            dy_ds3 = 0.d0
            dz_ds1 = 0.d0
            dz_ds2 = 0.d0
            dz_ds3 = 0.d0
c 
c calculate the jacobian at the integration station -
            do kn = 1,npe              
c
               dx_ds1 = dx_ds1+deriv(1,kn,ki)*cordel(1,kn,ke)
               dx_ds2 = dx_ds2+deriv(2,kn,ki)*cordel(1,kn,ke)
               dx_ds3 = dx_ds3+deriv(3,kn,ki)*cordel(1,kn,ke)
c                                                           
               dy_ds1 = dy_ds1+deriv(1,kn,ki)*cordel(2,kn,ke)
               dy_ds2 = dy_ds2+deriv(2,kn,ki)*cordel(2,kn,ke)
               dy_ds3 = dy_ds3+deriv(3,kn,ki)*cordel(2,kn,ke)
c                                             
               dz_ds1 = dz_ds1+deriv(1,kn,ki)*cordel(3,kn,ke)
               dz_ds2 = dz_ds2+deriv(2,kn,ki)*cordel(3,kn,ke)
               dz_ds3 = dz_ds3+deriv(3,kn,ki)*cordel(3,kn,ke)
c
            end do
c
c calculate the determinate of the jacobian at the integration station -
            det_j(ke,ki) = dx_ds1*( dy_ds2*dz_ds3 - dz_ds2*dy_ds3 )
     *                   + dy_ds1*( dz_ds2*dx_ds3 - dx_ds2*dz_ds3 )
     *                   + dz_ds1*( dx_ds2*dy_ds3 - dy_ds2*dx_ds3 )
c
c protect against a negative or small value for the determinate of the 
c jacobian. The value of real_min (set in precision.par) represents 
c the smallest Real value (based upon the precision set for this 
c compilation) which the machine can represent - 
            test = det_j(ke,ki)
            if( test .le. 1.d6*realmin ) then
               test = 1.d0
               err(ke) = 1.d0
            end if
            denom = 1.d0/test
c
c compute the gradient operators at the integration station -
c
            ds1_dx = denom*(dy_ds2*dz_ds3 - dz_ds2*dy_ds3)
            ds2_dx = denom*(dz_ds1*dy_ds3 - dy_ds1*dz_ds3)
            ds3_dx = denom*(dy_ds1*dz_ds2 - dz_ds1*dy_ds2)
c
            ds1_dy = denom*(dz_ds2*dx_ds3 - dx_ds2*dz_ds3)
            ds2_dy = denom*(dx_ds1*dz_ds3 - dz_ds1*dx_ds3)
            ds3_dy = denom*(dz_ds1*dx_ds2 - dx_ds1*dz_ds2)
c
            ds1_dz = denom*(dx_ds2*dy_ds3 - dy_ds2*dx_ds3)
            ds2_dz = denom*(dy_ds1*dx_ds3 - dx_ds1*dy_ds3)
            ds3_dz = denom*(dx_ds1*dy_ds2 - dy_ds1*dx_ds2)
c
            do kn = 1,npe 
c
              gradop(1,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dx
     *           + deriv(2,kn,ki)*ds2_dx
     *           + deriv(3,kn,ki)*ds3_dx
c       
              gradop(2,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dy
     *           + deriv(2,kn,ki)*ds2_dy
     *           + deriv(3,kn,ki)*ds3_dy
c       
              gradop(3,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dz
     *           + deriv(2,kn,ki)*ds2_dz
     *           + deriv(3,kn,ki)*ds3_dz
c
            end do
         end do
      end do
c
c summarize volume error checks - 
      sum = 0.d0
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if( sum .ne. 0.d0 ) then
c flag error -
         do ke = 1 , nelem
            if ( err(ke) .ne. 0.d0 ) nerr = ke
         end do
      end if
c     
      return
      end

      subroutine quad_gradient_operator( nelem, npe, 
     *    nint, deriv, cordel, gradop, det_j, err, nerr )
c       
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the gradient operator, determinate of 
c     the Jacobian, and error count for an element workset of 2D 
c     subcontrol surface elements The gradient operator and the 
c     determinate of the jacobians are computed at the center of
c     each control surface (the locations for the integration rule
c     are at the center of each control surface).
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     deriv         real  shape function derivatives evaluated at the
c                         integration stations
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     gradop        real  element gradient operator at each integration
c                         station
c     det_j         real  determinate of the jacobian at each integration
c                         station
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      
      integer nelem, npe, nint, nerr
      double precision deriv, cordel, gradop, det_j, err

      dimension deriv(2,npe,nint),
     *          cordel(2,npe,nelem),
     *          gradop(2,npe,nelem,nint),
     *          det_j(nelem,nint),
     *          err(nelem)
c
      integer ke, ki, kn
      double precision dx_ds1, dx_ds2
      double precision dy_ds1, dy_ds2
      double precision ds1_dx, ds1_dy
      double precision ds2_dx, ds2_dy

      double precision test, denom, sum, realmin
      realmin = 2.2250738585072014d-308

      do ke = 1,nelem
         err(ke) = 0.d0
      end do
c
      do ki = 1,nint
         do ke = 1,nelem
            dx_ds1 = 0.d0
            dx_ds2 = 0.d0
            dy_ds1 = 0.d0
            dy_ds2 = 0.d0
c 
c calculate the jacobian at the integration station -
            do kn = 1,npe              
c
               dx_ds1 = dx_ds1+deriv(1,kn,ki)*cordel(1,kn,ke)
               dx_ds2 = dx_ds2+deriv(2,kn,ki)*cordel(1,kn,ke)
c                                                           
               dy_ds1 = dy_ds1+deriv(1,kn,ki)*cordel(2,kn,ke)
               dy_ds2 = dy_ds2+deriv(2,kn,ki)*cordel(2,kn,ke)
c
            end do
c
c calculate the determinate of the jacobian at the integration station -
            det_j(ke,ki) = dx_ds1*dy_ds2 - dy_ds1*dx_ds2
c
c protect against a negative or small value for the determinate of the 
c jacobian. The value of real_min (set in precision.par) represents 
c the smallest Real value (based upon the precision set for this 
c compilation) which the machine can represent - 
            test = det_j(ke,ki)
            if( test .le. 1.d6*realmin ) then
               test = 1.d0
               err(ke) = 1.d0
            end if
            denom = 1.d0/test
c
c compute the gradient operators at the integration station -
c
            ds1_dx =  denom*dy_ds2
            ds2_dx = -denom*dy_ds1
c
            ds1_dy = -denom*dx_ds2
            ds2_dy =  denom*dx_ds1
c
            do kn = 1,npe 
c
              gradop(1,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dx
     *           + deriv(2,kn,ki)*ds2_dx
c       
              gradop(2,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dy
     *           + deriv(2,kn,ki)*ds2_dy
c       
            end do
         end do
      end do
c
c summarize volume error checks - 
      sum = 0.d0
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if( sum .ne. 0.d0 ) then
c flag error -
         do ke = 1 , nelem
            if ( err(ke) .ne. 0.d0 ) nerr = ke
         end do
      end if
c
      return
      end

      subroutine twod_gij( npe, 
     *    nint, deriv, cordel, gupperij, glowerij )
c       
c***********************************************************************
c
      implicit none
      
      integer npe, nint
      double precision deriv, cordel, gupperij, glowerij

      dimension deriv(2,npe,nint),
     *          cordel(2,npe),
     *          gupperij(2,2,nint),
     *          glowerij(2,2,nint)
c
      integer ki, kn, i, j
      double precision dx_ds, ds_dx
      dimension dx_ds(2,2), ds_dx(2,2)
      double precision det_j, denom, realmin
      realmin = 2.2250738585072014d-308

      do ki = 1,nint
c     zero out
         dx_ds(1,1) = 0.d0
         dx_ds(1,2) = 0.d0
         dx_ds(2,1) = 0.d0
         dx_ds(2,2) = 0.d0
c 
c calculate the jacobian at the integration station -
         do kn = 1,npe              
c
            dx_ds(1,1) = dx_ds(1,1)+deriv(1,kn,ki)*cordel(1,kn)
            dx_ds(1,2) = dx_ds(1,2)+deriv(2,kn,ki)*cordel(1,kn)
c     
            dx_ds(2,1) = dx_ds(2,1)+deriv(1,kn,ki)*cordel(2,kn)
            dx_ds(2,2) = dx_ds(2,2)+deriv(2,kn,ki)*cordel(2,kn)
c     
         end do

c calculate the determinate of the jacobian at the integration station -
         det_j = dx_ds(1,1)*dx_ds(2,2) - dx_ds(2,1)*dx_ds(1,2)

c clip
         if ( det_j .le. 1.d6*realmin ) then
            denom = 1.d0
         else
            denom = 1.d0/det_j
         endif

c compute the inverse jacobian
         ds_dx(1,1) =  dx_ds(2,2)*denom 
         ds_dx(1,2) = -dx_ds(1,2)*denom
         ds_dx(2,1) = -dx_ds(2,1)*denom
         ds_dx(2,2) =  dx_ds(1,1)*denom
c         
         do i = 1, 2
            do j = 1, 2
               gupperij(i,j,ki) = 
     $              dx_ds(i,1)*dx_ds(j,1)+dx_ds(i,2)*dx_ds(j,2)
               glowerij(i,j,ki) = 
     $              ds_dx(1,i)*ds_dx(1,j)+ds_dx(2,i)*ds_dx(2,j)
            end do
         end do
c     
      end do
c      
      return
      end

      subroutine threed_gij( npe, 
     *    nint, deriv, cordel, gupperij, glowerij )
c       
c***********************************************************************
c
      implicit none
      
      integer npe, nint
      double precision deriv, cordel, gupperij, glowerij

      dimension deriv(3,npe,nint),
     *          cordel(3,npe),
     *          gupperij(3,3,nint),
     *          glowerij(3,3,nint)
c
      integer ki, kn, i, j
      double precision dx_ds, ds_dx
      dimension dx_ds(3,3), ds_dx(3,3)
      double precision det_j, denom, realmin
      realmin = 2.2250738585072014d-308

      do ki = 1,nint
c     zero out
         dx_ds(1,1) = 0.d0
         dx_ds(1,2) = 0.d0
         dx_ds(1,3) = 0.d0
         dx_ds(2,1) = 0.d0
         dx_ds(2,2) = 0.d0
         dx_ds(2,3) = 0.d0
         dx_ds(3,1) = 0.d0
         dx_ds(3,2) = 0.d0
         dx_ds(3,3) = 0.d0
c 
c calculate the jacobian at the integration station -
         do kn = 1,npe              
c
            dx_ds(1,1) = dx_ds(1,1)+deriv(1,kn,ki)*cordel(1,kn)
            dx_ds(1,2) = dx_ds(1,2)+deriv(2,kn,ki)*cordel(1,kn)
            dx_ds(1,3) = dx_ds(1,3)+deriv(3,kn,ki)*cordel(1,kn)
c     
            dx_ds(2,1) = dx_ds(2,1)+deriv(1,kn,ki)*cordel(2,kn)
            dx_ds(2,2) = dx_ds(2,2)+deriv(2,kn,ki)*cordel(2,kn)
            dx_ds(2,3) = dx_ds(2,3)+deriv(3,kn,ki)*cordel(2,kn)
c     
            dx_ds(3,1) = dx_ds(3,1)+deriv(1,kn,ki)*cordel(3,kn)
            dx_ds(3,2) = dx_ds(3,2)+deriv(2,kn,ki)*cordel(3,kn)
            dx_ds(3,3) = dx_ds(3,3)+deriv(3,kn,ki)*cordel(3,kn)
c     
         end do

c calculate the determinate of the Jacobian at the integration station -
         det_j= dx_ds(1,1)*(dx_ds(2,2)*dx_ds(3,3)-dx_ds(3,2)*dx_ds(2,3))
     *        + dx_ds(2,1)*(dx_ds(3,2)*dx_ds(1,3)-dx_ds(1,2)*dx_ds(3,3))
     *        + dx_ds(3,1)*(dx_ds(1,2)*dx_ds(2,3)-dx_ds(2,2)*dx_ds(1,3))

c clip
         if ( det_j .le. 1.d6*realmin ) then
            denom = 1.d0
         else
            denom = 1.d0/det_j
         endif

c calculate the inverse Jacobian
         ds_dx(1,1)= denom*(dx_ds(2,2)*dx_ds(3,3)-dx_ds(3,2)*dx_ds(2,3))
         ds_dx(1,2)= denom*(dx_ds(3,2)*dx_ds(1,3)-dx_ds(1,2)*dx_ds(3,3))
         ds_dx(1,3)= denom*(dx_ds(1,2)*dx_ds(2,3)-dx_ds(2,2)*dx_ds(1,3))
c     
         ds_dx(2,1)= denom*(dx_ds(3,1)*dx_ds(2,3)-dx_ds(2,1)*dx_ds(3,3))
         ds_dx(2,2)= denom*(dx_ds(1,1)*dx_ds(3,3)-dx_ds(3,1)*dx_ds(1,3))
         ds_dx(2,3)= denom*(dx_ds(2,1)*dx_ds(1,3)-dx_ds(1,1)*dx_ds(2,3))
c
         ds_dx(3,1)= denom*(dx_ds(2,1)*dx_ds(3,2)-dx_ds(3,1)*dx_ds(2,2))
         ds_dx(3,2)= denom*(dx_ds(3,1)*dx_ds(1,2)-dx_ds(1,1)*dx_ds(3,2))
         ds_dx(3,3)= denom*(dx_ds(1,1)*dx_ds(2,2)-dx_ds(2,1)*dx_ds(1,2))
C
         do i = 1, 3
            do j = 1, 3
               gupperij(i,j,ki) = 
     $              dx_ds(i,1)*dx_ds(j,1)+dx_ds(i,2)*dx_ds(j,2)
     $              +dx_ds(i,3)*dx_ds(j,3)
               glowerij(i,j,ki) = 
     $              ds_dx(1,i)*ds_dx(1,j)+ds_dx(2,i)*ds_dx(2,j)
     $              +ds_dx(3,i)*ds_dx(3,j)
            end do
         end do
c     
      end do
c      
      return
      end

      subroutine tet_gradient_operator( nelem, npe,
     *    nint, deriv, cordel, gradop, det_j, err, nerr )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the gradient operator, determinate of
c     the Jacobian, and error count for an element workset of 3D
c     subcontrol surface elements The gradient operator and the
c     determinate of the jacobians are computed at the center of
c     each control surface (the locations for the integration rule
c     are at the center of each control surface).
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     deriv         real  shape function derivatives evaluated at the
c                         integration stations
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     gradop        real  element gradient operator at each integration
c                         station
c     det_j         real  determinate of the jacobian at each integration
c                         station
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      
      integer nelem, npe, nint, nerr
      double precision deriv, cordel, gradop, det_j, err

      dimension deriv(3,npe,nint),
     *          cordel(3,npe,nelem),
     *          gradop(3,npe,nelem,nint),
     *          det_j(nelem,nint),
     *          err(nelem)
c
      integer ke, ki, kn
      double precision dx_ds1, dx_ds2, dx_ds3
      double precision dy_ds1, dy_ds2, dy_ds3
      double precision dz_ds1, dz_ds2, dz_ds3
      double precision ds1_dx, ds1_dy, ds1_dz
      double precision ds2_dx, ds2_dy, ds2_dz
      double precision ds3_dx, ds3_dy, ds3_dz

      double precision test, denom, sum, realmin
      realmin = 2.2250738585072014d-308

      do ke = 1,nelem
         err(ke) = 0.d0
      end do
c
      do ki = 1,nint
         do ke = 1,nelem
            dx_ds1 = 0.d0
            dx_ds2 = 0.d0
            dx_ds3 = 0.d0
            dy_ds1 = 0.d0
            dy_ds2 = 0.d0
            dy_ds3 = 0.d0
            dz_ds1 = 0.d0
            dz_ds2 = 0.d0
            dz_ds3 = 0.d0
c
c calculate the jacobian at the integration station -
c
            do kn = 1,npe
c
               dx_ds1 = dx_ds1+deriv(1,kn,ki)*cordel(1,kn,ke)
               dx_ds2 = dx_ds2+deriv(2,kn,ki)*cordel(1,kn,ke)
               dx_ds3 = dx_ds3+deriv(3,kn,ki)*cordel(1,kn,ke)
c
               dy_ds1 = dy_ds1+deriv(1,kn,ki)*cordel(2,kn,ke)
               dy_ds2 = dy_ds2+deriv(2,kn,ki)*cordel(2,kn,ke)
               dy_ds3 = dy_ds3+deriv(3,kn,ki)*cordel(2,kn,ke)
c
               dz_ds1 = dz_ds1+deriv(1,kn,ki)*cordel(3,kn,ke)
               dz_ds2 = dz_ds2+deriv(2,kn,ki)*cordel(3,kn,ke)
               dz_ds3 = dz_ds3+deriv(3,kn,ki)*cordel(3,kn,ke)
c
            end do
c
c calculate the determinate of the jacobian at the integration station -
c
            det_j(ke,ki) = dx_ds1*( dy_ds2*dz_ds3 - dz_ds2*dy_ds3 )
     *                   + dy_ds1*( dz_ds2*dx_ds3 - dx_ds2*dz_ds3 )
     *                   + dz_ds1*( dx_ds2*dy_ds3 - dy_ds2*dx_ds3 )
c
c protect against a negative or small value for the determinate of the
c jacobian. The value of real_min represents
c the smallest Real value (based upon the precision set for this
c compilation) which the machine can represent -
c
            test = det_j(ke,ki)
            if( test .le. 1.d6*realmin ) then
               test = 1.d0
               err(ke) = 1.d0
            end if
            denom = 1.d0/test
c
c compute the gradient operators at the integration station -
c
            ds1_dx = denom*(dy_ds2*dz_ds3 - dz_ds2*dy_ds3)
            ds2_dx = denom*(dz_ds1*dy_ds3 - dy_ds1*dz_ds3)
            ds3_dx = denom*(dy_ds1*dz_ds2 - dz_ds1*dy_ds2)
c
            ds1_dy = denom*(dz_ds2*dx_ds3 - dx_ds2*dz_ds3)
            ds2_dy = denom*(dx_ds1*dz_ds3 - dz_ds1*dx_ds3)
            ds3_dy = denom*(dz_ds1*dx_ds2 - dx_ds1*dz_ds2)
c
            ds1_dz = denom*(dx_ds2*dy_ds3 - dy_ds2*dx_ds3)
            ds2_dz = denom*(dy_ds1*dx_ds3 - dx_ds1*dy_ds3)
            ds3_dz = denom*(dx_ds1*dy_ds2 - dy_ds1*dx_ds2)
c
            do kn = 1,npe
c
              gradop(1,kn,ke,ki) =
     *             deriv(1,kn,ki)*ds1_dx
     *           + deriv(2,kn,ki)*ds2_dx
     *           + deriv(3,kn,ki)*ds3_dx
c
              gradop(2,kn,ke,ki) =
     *             deriv(1,kn,ki)*ds1_dy
     *           + deriv(2,kn,ki)*ds2_dy
     *           + deriv(3,kn,ki)*ds3_dy
c
              gradop(3,kn,ke,ki) =
     *             deriv(1,kn,ki)*ds1_dz
     *           + deriv(2,kn,ki)*ds2_dz
     *           + deriv(3,kn,ki)*ds3_dz
c
            end do
         end do
      end do
c
c summarize volume error checks -
c
      sum = 0.d0
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if( sum .ne. 0.d0 ) then
c flag error -
         do ke = 1 , nelem
            if ( err(ke) .ne. 0.d0 ) nerr = ke
         end do
      end if
c
      return
      end

      subroutine tri_gradient_operator( nelem, npe,
     *    nint, deriv, cordel, gradop, det_j, err, nerr )
c
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the gradient operator, determinate of
c     the Jacobian, and error count for an element workset of 3D
c     subcontrol surface elements The gradient operator and the
c     determinate of the jacobians are computed at the center of
c     each control surface (the locations for the integration rule
c     are at the center of each control surface).
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     deriv         real  shape function derivatives evaluated at the
c                         integration stations
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     gradop        real  element gradient operator at each integration
c                         station
c     det_j         real  determinate of the jacobian at each integration
c                         station
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none
      
      integer nelem, npe, nint, nerr
      double precision deriv, cordel, gradop, det_j, err

      dimension deriv(2,npe,nint),
     *          cordel(2,npe,nelem),
     *          gradop(2,npe,nelem,nint),
     *          det_j(nelem,nint),
     *          err(nelem)
c
      integer ke, ki, kn
      double precision dx_ds1, dx_ds2
      double precision dy_ds1, dy_ds2
      double precision ds1_dx, ds1_dy
      double precision ds2_dx, ds2_dy

      double precision test, denom, sum, realmin
      realmin = 2.2250738585072014d-308
      
      do ke = 1,nelem
         err(ke) = 0.d0
      end do
c     
      do ki = 1,nint
         do ke = 1,nelem
            
            dx_ds1 = 0.d0
            dx_ds2 = 0.d0
            dy_ds1 = 0.d0
            dy_ds2 = 0.d0
c     
c     calculate the jacobian at the integration station -
            do kn = 1,npe              
c     
               dx_ds1 = dx_ds1+deriv(1,kn,ki)*cordel(1,kn,ke)
               dx_ds2 = dx_ds2+deriv(2,kn,ki)*cordel(1,kn,ke)
c     
               dy_ds1 = dy_ds1+deriv(1,kn,ki)*cordel(2,kn,ke)
               dy_ds2 = dy_ds2+deriv(2,kn,ki)*cordel(2,kn,ke)
c     
            end do
c     
c     calculate the determinate of the jacobian at the integration station -
            det_j(ke,ki) = dx_ds1*dy_ds2 - dy_ds1*dx_ds2
c     
c     protect against a negative or small value for the determinate of the 
c     jacobian. The value of real_min (set in precision.par) represents 
c     the smallest Real value (based upon the precision set for this 
c     compilation) which the machine can represent - 
            test = det_j(ke,ki)
            if( test .le. 1.d6*realmin ) then
               test = 1.d0
               err(ke) = 1.d0
            end if
            denom = 1.d0/test
c     
c     compute the gradient operators at the integration station -
c     
            ds1_dx =  denom*dy_ds2
            ds2_dx = -denom*dy_ds1
c     
            ds1_dy = -denom*dx_ds2
            ds2_dy =  denom*dx_ds1
c     
            do kn = 1,npe 
c     
               gradop(1,kn,ke,ki) = 
     *              deriv(1,kn,ki)*ds1_dx
     *              + deriv(2,kn,ki)*ds2_dx
c     
               gradop(2,kn,ke,ki) = 
     *              deriv(1,kn,ki)*ds1_dy
     *              + deriv(2,kn,ki)*ds2_dy   
c     
            end do
         end do
      end do
c     
c     summarize volume error checks -
c     
      sum = 0.d0
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if( sum .ne. 0.d0 ) then
c     flag error -
         do ke = 1 , nelem
            if ( err(ke) .ne. 0.d0 ) nerr = ke
         end do
      end if
c     
      return
      end

      subroutine pyr_gradient_operator( nelem, npe, 
     *    nint, deriv, cordel, gradop, det_j, err, nerr )
c       
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the gradient operator, determinate of 
c     the Jacobian, and error count for an element workset of 3D 
c     subcontrol surface elements The gradient operator and the 
c     determinate of the jacobians are computed at the center of
c     each control surface (the locations for the integration rule
c     are at the center of each control surface).
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     deriv         real  shape function derivatives evaluated at the
c                         integration stations
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     gradop        real  element gradient operator at each integration
c                         station
c     det_j         real  determinate of the jacobian at each integration
c                         station
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none

      integer nelem, npe, nint, nerr
      double precision deriv, cordel, gradop, det_j, err
c     
      dimension deriv(3,npe,nint),
     *          cordel(3,npe,nelem),
     *          gradop(3,npe,nelem,nint),
     *          det_j(nelem,nint),
     *          err(nelem)
c
      integer ke, ki, kn
      double precision dx_ds1, dx_ds2, dx_ds3
      double precision dy_ds1, dy_ds2, dy_ds3
      double precision dz_ds1, dz_ds2, dz_ds3
      double precision ds1_dx, ds1_dy, ds1_dz
      double precision ds2_dx, ds2_dy, ds2_dz
      double precision ds3_dx, ds3_dy, ds3_dz

      double precision test, denom, sum, realmin
      realmin = 2.2250738585072014d-308

      do ke = 1,nelem
         err(ke) = 0.d0
      end do
c
      do ki = 1,nint
         do ke = 1,nelem
            dx_ds1 = 0.d0
            dx_ds2 = 0.d0
            dx_ds3 = 0.d0
            dy_ds1 = 0.d0
            dy_ds2 = 0.d0
            dy_ds3 = 0.d0
            dz_ds1 = 0.d0
            dz_ds2 = 0.d0
            dz_ds3 = 0.d0
c 
c calculate the jacobian at the integration station -
            do kn = 1,npe              
c
               dx_ds1 = dx_ds1+deriv(1,kn,ki)*cordel(1,kn,ke)
               dx_ds2 = dx_ds2+deriv(2,kn,ki)*cordel(1,kn,ke)
               dx_ds3 = dx_ds3+deriv(3,kn,ki)*cordel(1,kn,ke)
c                                                          
               dy_ds1 = dy_ds1+deriv(1,kn,ki)*cordel(2,kn,ke)
               dy_ds2 = dy_ds2+deriv(2,kn,ki)*cordel(2,kn,ke)
               dy_ds3 = dy_ds3+deriv(3,kn,ki)*cordel(2,kn,ke)
c                                             
               dz_ds1 = dz_ds1+deriv(1,kn,ki)*cordel(3,kn,ke)
               dz_ds2 = dz_ds2+deriv(2,kn,ki)*cordel(3,kn,ke)
               dz_ds3 = dz_ds3+deriv(3,kn,ki)*cordel(3,kn,ke)
c
            end do
c
c calculate the determinate of the jacobian at the integration station -
            det_j(ke,ki) = dx_ds1*( dy_ds2*dz_ds3 - dz_ds2*dy_ds3 )
     *                   + dy_ds1*( dz_ds2*dx_ds3 - dx_ds2*dz_ds3 )
     *                   + dz_ds1*( dx_ds2*dy_ds3 - dy_ds2*dx_ds3 )
c
c protect against a negative or small value for the determinate of the 
c jacobian. The value of real_min (set in precision.par) represents 
c the smallest Real value (based upon the precision set for this 
c compilation) which the machine can represent - 
            test = det_j(ke,ki)
            if( test .le. 1.d6*realmin ) then
               test = 1.d0
               err(ke) = 1.d0
            end if
            denom = 1.d0/test
c
c compute the gradient operators at the integration station -
c
            ds1_dx = denom*(dy_ds2*dz_ds3 - dz_ds2*dy_ds3)
            ds2_dx = denom*(dz_ds1*dy_ds3 - dy_ds1*dz_ds3)
            ds3_dx = denom*(dy_ds1*dz_ds2 - dz_ds1*dy_ds2)
c
            ds1_dy = denom*(dz_ds2*dx_ds3 - dx_ds2*dz_ds3)
            ds2_dy = denom*(dx_ds1*dz_ds3 - dz_ds1*dx_ds3)
            ds3_dy = denom*(dz_ds1*dx_ds2 - dx_ds1*dz_ds2)
c
            ds1_dz = denom*(dx_ds2*dy_ds3 - dy_ds2*dx_ds3)
            ds2_dz = denom*(dy_ds1*dx_ds3 - dx_ds1*dy_ds3)
            ds3_dz = denom*(dx_ds1*dy_ds2 - dy_ds1*dx_ds2)
c
            do kn = 1,npe 
c
              gradop(1,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dx
     *           + deriv(2,kn,ki)*ds2_dx
     *           + deriv(3,kn,ki)*ds3_dx
c       
              gradop(2,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dy
     *           + deriv(2,kn,ki)*ds2_dy
     *           + deriv(3,kn,ki)*ds3_dy
c      
              gradop(3,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dz
     *           + deriv(2,kn,ki)*ds2_dz
     *           + deriv(3,kn,ki)*ds3_dz
c
            end do
         end do
      end do
c
c summarize volume error checks - 
      sum = 0.d0
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if( sum .ne. 0.d0 ) then
c flag error -
         do ke = 1 , nelem
            if ( err(ke) .ne. 0.d0 ) nerr = ke
         end do
      end if
c
      return
      end

      subroutine wed_gradient_operator( nelem, npe, 
     *    nint, deriv, cordel, gradop, det_j, err, nerr )
c       
c***********************************************************************
c***********************************************************************
c
c description:
c     This  routine returns the gradient operator, determinate of 
c     the Jacobian, and error count for an element workset of 3D 
c     subcontrol surface elements The gradient operator and the 
c     determinate of the jacobians are computed at the center of
c     each control surface (the locations for the integration rule
c     are at the center of each control surface).
c
c formal parameters - input:
c     nelem         int   number of elements in the workset
c     npe           int   number of nodes per element
c     nint          int   number of sub control surfaces
c     deriv         real  shape function derivatives evaluated at the
c                         integration stations
c     cordel        real  element local coordinates
c
c formal parameters - output:
c     gradop        real  element gradient operator at each integration
c                         station
c     det_j         real  determinate of the jacobian at each integration
c                         station
c     err           real  positive volume check (0. = no error, 1. = error)
c     nerr          int   element number which fails positive volume check
c
c***********************************************************************
c
      implicit none

      integer nelem, npe, nint, nerr
      double precision deriv, cordel, gradop, det_j, err
c
      dimension deriv(3,npe,nint),
     *          cordel(3,npe,nelem),
     *          gradop(3,npe,nelem,nint),
     *          det_j(nelem,nint),
     *          err(nelem)
c
      integer ke, ki, kn      
      double precision dx_ds1, dx_ds2, dx_ds3
      double precision dy_ds1, dy_ds2, dy_ds3
      double precision dz_ds1, dz_ds2, dz_ds3
      double precision ds1_dx, ds1_dy, ds1_dz
      double precision ds2_dx, ds2_dy, ds2_dz
      double precision ds3_dx, ds3_dy, ds3_dz

      double precision test, denom, sum, realmin
      realmin = 2.2250738585072014d-308

      do ke = 1,nelem
         err(ke) = 0.d0
      end do
c
      do ki = 1,nint
         do ke = 1,nelem
            dx_ds1 = 0.d0
            dx_ds2 = 0.d0
            dx_ds3 = 0.d0
            dy_ds1 = 0.d0
            dy_ds2 = 0.d0
            dy_ds3 = 0.d0
            dz_ds1 = 0.d0
            dz_ds2 = 0.d0
            dz_ds3 = 0.d0
c 
c calculate the jacobian at the integration station -
            do kn = 1,npe              
c
               dx_ds1 = dx_ds1+deriv(1,kn,ki)*cordel(1,kn,ke)
               dx_ds2 = dx_ds2+deriv(2,kn,ki)*cordel(1,kn,ke)
               dx_ds3 = dx_ds3+deriv(3,kn,ki)*cordel(1,kn,ke)
c                                                           
               dy_ds1 = dy_ds1+deriv(1,kn,ki)*cordel(2,kn,ke)
               dy_ds2 = dy_ds2+deriv(2,kn,ki)*cordel(2,kn,ke)
               dy_ds3 = dy_ds3+deriv(3,kn,ki)*cordel(2,kn,ke)
c                                             
               dz_ds1 = dz_ds1+deriv(1,kn,ki)*cordel(3,kn,ke)
               dz_ds2 = dz_ds2+deriv(2,kn,ki)*cordel(3,kn,ke)
               dz_ds3 = dz_ds3+deriv(3,kn,ki)*cordel(3,kn,ke)
c
            end do
c
c calculate the determinate of the jacobian at the integration station -
            det_j(ke,ki) = dx_ds1*( dy_ds2*dz_ds3 - dz_ds2*dy_ds3 )
     *                   + dy_ds1*( dz_ds2*dx_ds3 - dx_ds2*dz_ds3 )
     *                   + dz_ds1*( dx_ds2*dy_ds3 - dy_ds2*dx_ds3 )
c
c protect against a negative or small value for the determinate of the 
c jacobian. The value of realmmin represents 
c the smallest Real value (based upon the precision set for this 
c compilation) which the machine can represent - 
            test = det_j(ke,ki)
            if( test .le. 1.d6*realmin ) then
               test = 1.d0
               err(ke) = 1.d0
            end if
            denom = 1.d0/test
c
c compute the gradient operators at the integration station -
c
            ds1_dx = denom*(dy_ds2*dz_ds3 - dz_ds2*dy_ds3)
            ds2_dx = denom*(dz_ds1*dy_ds3 - dy_ds1*dz_ds3)
            ds3_dx = denom*(dy_ds1*dz_ds2 - dz_ds1*dy_ds2)
c
            ds1_dy = denom*(dz_ds2*dx_ds3 - dx_ds2*dz_ds3)
            ds2_dy = denom*(dx_ds1*dz_ds3 - dz_ds1*dx_ds3)
            ds3_dy = denom*(dz_ds1*dx_ds2 - dx_ds1*dz_ds2)
c
            ds1_dz = denom*(dx_ds2*dy_ds3 - dy_ds2*dx_ds3)
            ds2_dz = denom*(dy_ds1*dx_ds3 - dx_ds1*dy_ds3)
            ds3_dz = denom*(dx_ds1*dy_ds2 - dy_ds1*dx_ds2)
c
            do kn = 1,npe 
c
              gradop(1,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dx
     *           + deriv(2,kn,ki)*ds2_dx
     *           + deriv(3,kn,ki)*ds3_dx
c       
              gradop(2,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dy
     *           + deriv(2,kn,ki)*ds2_dy
     *           + deriv(3,kn,ki)*ds3_dy
c       
              gradop(3,kn,ke,ki) = 
     *             deriv(1,kn,ki)*ds1_dz
     *           + deriv(2,kn,ki)*ds2_dz
     *           + deriv(3,kn,ki)*ds3_dz
c
            end do
         end do
      end do
c
c summarize volume error checks - 
      sum = 0.d0
      do ke = 1 , nelem
         sum = sum + err(ke)
      end do
      if( sum .ne. 0.d0 ) then
c flag error -
         do ke = 1 , nelem
            if ( err(ke) .ne. 0.d0 ) nerr = ke
         end do
      end if
c
      return
      end
      
      subroutine quad3d_shape_fcn( npts, 
     *     par_coord, shape_fcn )
c       
c***********************************************************************
c***********************************************************************
c
c formal parameters - input:
c     npts          int   number of points to evaluate (usually 
c                         the number of Gauss Points)
c     par_coord     real  parametric coordinates of the points to be
c                         evaluated (typically, the gauss pts)
c
c formal parameters - output:
c     shape_fcn     real  shape functions evaluated at the evaluation
c                         points
c
c***********************************************************************
c
      implicit none
c
      integer npts
      double precision par_coord, shape_fcn

      dimension par_coord(2,npts)
      dimension shape_fcn(4,npts)
c
      integer j
      double precision s1, s2, one4th, half
c
      one4th = 1.d0/4.d0
      half = 1.d0/2.d0
c
      do j = 1,npts
c
         s1 = par_coord(1,j)
         s2 = par_coord(2,j)
c
         shape_fcn(1,j) = one4th + half*(-s1 - s2) + s1*s2
         shape_fcn(2,j) = one4th + half*( s1 - s2) - s1*s2
         shape_fcn(3,j) = one4th + half*( s1 + s2) + s1*s2
         shape_fcn(4,j) = one4th + half*(-s1 + s2) - s1*s2
c
      end do
c
      return
      end

      subroutine quad92d_derivative( npts, par_coord, deriv )
!       
!***********************************************************************
!***********************************************************************
!
! formal parameters - input:
!     npts          int   number of points to evaluate (usually 
!                         the number of Gauss Points)
!     par_coord     real  parametric coordinates of the points to be
!                         evaluated (typically, the gauss pts)
!
! formal parameters - output:
!     deriv         real  shape function dirivatives evaluated at 
!                         evaluation points.
!
!***********************************************************************
!
!
      implicit none
!
      integer npts
      double precision par_coord, deriv
c
      dimension par_coord(2,npts)
      dimension deriv(2,9,npts)

      integer j
      double precision s, t, t2, s2
      double precision one4th, half, one, two
!
      one4th = 1.d0/4.d0
      half = 1.d0/2.d0
      one = 1.d0
      two = 2.d0

      do j = 1,npts
         s = par_coord(1,j)
         t = par_coord(2,j)

         t2 = t*t
         s2 = s*s
!
! shape function derivative in the s direction -
!
        deriv(1,1,j) =  one4th * (two * s  * t2 - two*s*t-t2+t)
        deriv(1,2,j) =  one4th * (two * s  * t2 - two*s*t+t2-t)
        deriv(1,3,j) =  one4th * (two * s  * t2 + two*s*t+t2+t)
        deriv(1,4,j) =  one4th * (two * s  * t2 + two*s*t-t2-t)
        deriv(1,5,j) =   -half * (two * s  * t2 - two*s*t)
        deriv(1,6,j) =   -half * (two * s  * t2 + t2 - two*s-one)
        deriv(1,7,j) =   -half * (two * s  * t2 + two*s*t)
        deriv(1,8,j) =   -half * (two * s  * t2 - t2 - two*s+one)
        deriv(1,9,j) =            two * s  * t2      - two*s
!
! shape function derivative in the t direction -
!
        deriv(2,1,j) =  one4th * (two * s2 * t  - two*s*t-s2+s)
        deriv(2,2,j) =  one4th * (two * s2 * t  + two*s*t-s2-s)
        deriv(2,3,j) =  one4th * (two * s2 * t  + two*s*t+s2+s)
        deriv(2,4,j) =  one4th * (two * s2 * t  - two*s*t+s2-s)
        deriv(2,5,j) =   -half * (two * s2 * t  - s2 - two*t+one)
        deriv(2,6,j) =   -half * (two * s2 * t  + two*s*t)
        deriv(2,7,j) =   -half * (two * s2 * t  + s2 - two*t-one)
        deriv(2,8,j) =   -half * (two * s2 * t  - two*s*t)
        deriv(2,9,j) =            two * s2 * t       - two*t

      end do
!
      return
      end
