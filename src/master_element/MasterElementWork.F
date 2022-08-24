
c
c     Copyright 2002 Sandia Corporation, Albuquerque, NM.
c
c***********************************************************************

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
