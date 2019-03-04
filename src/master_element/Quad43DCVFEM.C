/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "master_element/MasterElement.h"
#include "master_element/Quad43DCVFEM.h"

#include "FORTRAN_Proto.h"

namespace sierra{
namespace nalu{


//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Quad3DSCS::Quad3DSCS()  
  : MasterElement(),
    elemThickness_(0.1)
{
  ndim(AlgTraits::nDim_);
  nodesPerElement_ = 4;
  numIntPoints_ = 4;
  scaleToStandardIsoFac_ = 2.0;

  // define ip node mappings; ordinal size = 1
  ipNodeMap_.resize(4);
  ipNodeMap_[0] = 0;
  ipNodeMap_[1] = 1;
  ipNodeMap_[2] = 2;
  ipNodeMap_[3] = 3;

  // standard integration location
  intgLoc_.resize(8);    
  intgLoc_[0]  = -0.25; intgLoc_[1] = -0.25; // surf 1
  intgLoc_[2]  =  0.25; intgLoc_[3] = -0.25; // surf 2
  intgLoc_[4]  =  0.25; intgLoc_[5] =  0.25; // surf 3
  intgLoc_[6]  = -0.25; intgLoc_[7] =  0.25; // surf 4

  // shifted
  intgLocShift_.resize(8);    
  intgLocShift_[0]  = -0.50; intgLocShift_[1] = -0.50; // surf 1
  intgLocShift_[2]  =  0.50; intgLocShift_[3] = -0.50; // surf 2
  intgLocShift_[4]  =  0.50; intgLocShift_[5] =  0.50; // surf 3
  intgLocShift_[6]  = -0.50; intgLocShift_[7] =  0.50; // surf 4  
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Quad3DSCS::~Quad3DSCS()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Quad3DSCS::ipNodeMap(
  int /*ordinal*/)
{
  // define ip->node mappings for each face (single ordinal); 
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void Quad3DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  int lerr = 0;

  SIERRA_FORTRAN(quad3d_scs_det)
    ( &nelem, coords, areav );

  // fake check
  *error = (lerr == 0) ? 0.0 : 1.0;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(quad3d_shape_fcn)
    (&numIntPoints_,&intgLoc_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::shifted_shape_fcn(double *shpfc)
{
  SIERRA_FORTRAN(quad3d_shape_fcn)
    (&numIntPoints_,&intgLocShift_[0],shpfc);
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  quad4_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- shifted_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::shifted_shape_fcn(SharedMemView<DoubleType**> &shpfc)
{
  quad4_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc);
}

//--------------------------------------------------------------------------
//-------- quad4_shape_fcn --------------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::quad4_shape_fcn(
  const int  & /* npts */,
  const double *isoParCoord,
  SharedMemView<DoubleType**> &shape_fcn)
{
  // -1/2:1/2 isoparametric range
  const DoubleType half = 0.50;
  const DoubleType one4th = 0.25;
  for ( int j = 0; j < numIntPoints_; ++j ) {

    const DoubleType s1 = isoParCoord[j*2];
    const DoubleType s2 = isoParCoord[j*2+1];

    shape_fcn(j,0) = one4th + half*(-s1 - s2) + s1*s2;
    shape_fcn(j,1) = one4th + half*( s1 - s2) - s1*s2;
    shape_fcn(j,2) = one4th + half*( s1 + s2) + s1*s2;
    shape_fcn(j,3) = one4th + half*(-s1 + s2) - s1*s2;
  }
}

//--------------------------------------------------------------------------
//-------- isInElement -----------------------------------------------------
//--------------------------------------------------------------------------
double
Quad3DSCS::isInElement(
  const double *elemNodalCoord,
  const double *pointCoord,
  double *isoParCoord )
{
  // square of the desired norm, 1.0e-8
  const double isInElemConverged = 1.0e-16;
  const int maxNonlinearIter = 20;

  // Translate element so that (x,y,z) coordinates of the first node are (0,0,0)

  double x[3] = { elemNodalCoord[1] - elemNodalCoord[0],
		elemNodalCoord[2] - elemNodalCoord[0],
		elemNodalCoord[3] - elemNodalCoord[0] };

  double y[3] = { elemNodalCoord[5] - elemNodalCoord[4],
		elemNodalCoord[6] - elemNodalCoord[4],
		elemNodalCoord[7] - elemNodalCoord[4] };

  double z[3] = { elemNodalCoord[9]  - elemNodalCoord[8],
		elemNodalCoord[10] - elemNodalCoord[8],
		elemNodalCoord[11] - elemNodalCoord[8] };

  // (xp,yp,zp) is the point at which we're searching for (xi,eta,d)
  // (must translate this also)
  // d = (scaled) distance in (x,y,z) space from point (xp,yp,zp) to the
  //     surface defined by the face element (the distance is scaled by
  //     the length of the non-unit normal vector; rescaling of d is done
  //     following the NR iteration below).

  double xp = pointCoord[0] - elemNodalCoord[0];
  double yp = pointCoord[1] - elemNodalCoord[4];
  double zp = pointCoord[2] - elemNodalCoord[8];


  // Newton-Raphson iteration for (xi,eta,d)

  double jdet;
  double j[9];
  double gn[3];
  double xcur[3];          // current (x,y,z) point on element surface
  double normal[3];        // (non-unit) normal computed at xcur

  // Solution vector solcur[3] = {xi,eta,d}
  double solcur[3] = {-0.5,-0.5,-0.5};     // initial guess
  double deltasol[] = {1.0,1.0, 1.0};

  int i = 0;
  do
  {
    // Update guess vector
    solcur[0] += deltasol[0];
    solcur[1] += deltasol[1];
    solcur[2] += deltasol[2];

    interpolatePoint(3,solcur,elemNodalCoord,xcur);

    // Translate xcur ((x,y,z) point corresponding
    // to current (xi,eta) guess)

    xcur[0] -= elemNodalCoord[0];
    xcur[1] -= elemNodalCoord[4];
    xcur[2] -= elemNodalCoord[8];

    non_unit_face_normal(solcur,elemNodalCoord,normal);

    gn[0] = xcur[0] - xp + solcur[2] * normal[0];
    gn[1] = xcur[1] - yp + solcur[2] * normal[1];
    gn[2] = xcur[2] - zp + solcur[2] * normal[2];

    // Mathematica-generated code for the jacobian

    j[0]=0.125*(-2.00*(-1.00+solcur[1])*x[0]
			    +(2.00*(1.00+solcur[1])*(x[1]-x[2])+solcur[2]
			      *(-(y[1]*z[0])+y[2]*z[0]+y[0]*z[1]-y[0]*z[2])));

    j[1]=0.125*(-2.00*(1.00+solcur[0])*x[0]
			    +2.00*(1.00+solcur[0])*x[1]-2.00
			    *(-1.00+solcur[0])*x[2]+(solcur[2]*(y[2]*(z[0]-z[1])+(-y[0]+y[1])*z[2])));

    j[2]= normal[0];

    j[3]=0.125*(-2.00*(-1.00+solcur[1])*y[0]
			    +(2.00*(1.00+solcur[1])*(y[1]-y[2])
			      +solcur[2]*(x[1]*z[0]-x[2]*z[0]-x[0]*z[1]+x[0]*z[2])));

    j[4]=0.125*(-2.00*(1.00+solcur[0])*y[0]
			    +2.00*(1.00+solcur[0])*y[1]
			    -2.00*(-1.00+solcur[0])*y[2]+(solcur[2]*(x[2]*(-z[0]+z[1])+(x[0]-x[1])*z[2])));

    j[5]= normal[1];

    j[6]=0.125*((solcur[2]*(-(x[1]*y[0])+x[2]*y[0]+x[0]*y[1]-x[0]*y[2]))
			    -2.00*((-1.00+solcur[1])*z[0]
					       -(1.00+solcur[1])*(z[1]-z[2])));

    j[7]=0.125*((solcur[2]*(x[2]*(y[0]-y[1])+(-x[0]+x[1])*y[2]))
			    -2.00*(1.00+solcur[0])*z[0]+2.00
			    *(1.00+solcur[0])*z[1]-2.00*(-1.00+solcur[0])*z[2]);
    
    j[8]= normal[2];
    
    jdet=-(j[2]*j[4]*j[6])+j[1]*j[5]*j[6]+j[2]*j[3]*j[7]-
      j[0]*j[5]*j[7]-j[1]*j[3]*j[8]+j[0]*j[4]*j[8];


    // Solve linear system (j*deltasol = -gn) for deltasol at step n+1
    
    deltasol[0] = (gn[2]*(j[2]*j[4]-j[1]*j[5])+gn[1]*(-(j[2]*j[7])+
		  j[1]*j[8])+gn[0]*(j[5]*j[7]-j[4]*j[8]))/jdet;
    deltasol[1] = (gn[2]*(-(j[2]*j[3])+j[0]*j[5])+gn[1]*(j[2]*j[6]-
		  j[0]*j[8])+gn[0]*(-(j[5]*j[6])+j[3]*j[8]))/jdet;
    deltasol[2] = (gn[2]*(j[1]*j[3]-j[0]*j[4])+gn[1]*(-(j[1]*j[6])+
		  j[0]*j[7])+gn[0]*(j[4]*j[6]-j[3]*j[7]))/jdet;

  } while ( !within_tolerance( vector_norm_sq(deltasol,3), isInElemConverged)
	    && ++i < maxNonlinearIter );

  // Fill in solution vector; only include the distance (in the third
  // solution vector slot) if npar_coord = 3 (this is how the user
  // requests it)

  isoParCoord[0] = std::numeric_limits<double>::max();
  isoParCoord[1] = std::numeric_limits<double>::max();
  isoParCoord[2] = std::numeric_limits<double>::max();
  double dist = std::numeric_limits<double>::max();

  if (i < maxNonlinearIter) {
    isoParCoord[0] = solcur[0] + deltasol[0];
    isoParCoord[1] = solcur[1] + deltasol[1];
    // Rescale the distance vector by the length of the (non-unit) normal vector,
    // which was used above in the NR iteration.
    const double area   = std::sqrt(vector_norm_sq(normal,3));
    const double length = std::sqrt(area);
    
    const double par_coor_2 = (solcur[2] + deltasol[2]) * length;
    //if ( npar_coord == 3 ) par_coor[2] = par_coor_2;
    isoParCoord[2] = par_coor_2;

    std::vector<double> xtmp(3);
    xtmp[0] = isoParCoord[0];
    xtmp[1] = isoParCoord[1];
    xtmp[2] = isoParCoord[2];
    dist = parametric_distance(xtmp);
  }
  return dist;
}

void
Quad3DSCS::non_unit_face_normal(
  const double * isoParCoord,            // (2)
  const double * elem_nodal_coor,        // (4,3)
	double * normal_vector )         // (3)
{
  double xi  = isoParCoord[0];
  double eta = isoParCoord[1];

  // Translate element so that node 0 is at (x,y,z) = (0,0,0)

  double x[3] = { elem_nodal_coor[1] - elem_nodal_coor[0],
                  elem_nodal_coor[2] - elem_nodal_coor[0],
                  elem_nodal_coor[3] - elem_nodal_coor[0] };
  
  double y[3] = { elem_nodal_coor[5] - elem_nodal_coor[4],
                  elem_nodal_coor[6] - elem_nodal_coor[4],
                  elem_nodal_coor[7] - elem_nodal_coor[4] };
  
  double z[3] = { elem_nodal_coor[9]  - elem_nodal_coor[8],
                  elem_nodal_coor[10] - elem_nodal_coor[8],
                  elem_nodal_coor[11] - elem_nodal_coor[8] };
  
  // Mathematica-generated and simplified code for the normal vector

  double n0 = 0.125*(xi*y[2]*z[0]+y[0]*z[1]+xi*y[0]*z[1]-y[2]*z[1]-
			       xi*y[0]*z[2]+y[1]*(-((1.00+xi)*z[0])+
	   (1.00+eta)*z[2])+eta*(y[2]*z[0]-y[2]*z[1]-y[0]*z[2]));

  double n1 = 0.125*(-(xi*x[2]*z[0])-x[0]*z[1]-xi*x[0]*z[1]+x[2]*z[1]+
				 xi*x[0]*z[2]+x[1]*((1.00+xi)*z[0]-
	   (1.00+eta)*z[2])+eta*(-(x[2]*z[0])+x[2]*z[1]+x[0]*z[2]));

  double n2 = 0.125*(xi*x[2]*y[0]+x[0]*y[1]+xi*x[0]*y[1]-x[2]*y[1]-
			       xi*x[0]*y[2]+x[1]*(-((1.00+xi)*y[0])+
	   (1.00+eta)*y[2])+eta*(x[2]*y[0]-x[2]*y[1]-x[0]*y[2]));

  normal_vector[0] = n0;
  normal_vector[1] = n1;
  normal_vector[2] = n2;

}

double 
Quad3DSCS::parametric_distance(const std::vector<double> &x)
{
  const int NCOORD   = 3;
  std::vector<double> y(NCOORD);
  
  for (int i=0; i<NCOORD; ++i) {
    y[i] = std::abs(x[i]);
  }

  double d = y[0];
  if (d < y[1]) d = y[1];
  if (elemThickness_ < y[2] && d < 1+y[2]) d = 1+y[2];
  return d;
}

//--------------------------------------------------------------------------
//-------- interpolatePoint ------------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result )
{
  // this is the same as the 2D implementation... Consider consolidation
  const double xi   = isoParCoord[0];
  const double eta  = isoParCoord[1];

  for ( int i = 0; i < nComp; i++ )
  {
    // Base 'field array' index for ith component
    int b = 4*i;

    result[i] = 0.250 * (
      (1.00-eta) * (1.00-xi ) * field[b+0] +
      (1.00-eta) * (1.00+xi ) * field[b+1] +
      (1.00+eta) * (1.00+xi ) * field[b+2] +
      (1.00+eta) * (1.00-xi ) * field[b+3] ) ;
  }
}

//--------------------------------------------------------------------------
//-------- general_shape_fcn -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::general_shape_fcn(
  const int numIp,
  const double *isoParCoord,
  double *shpfc)
{
  // -1:1 isoparametric range
  const double npe = nodesPerElement_;
  for ( int ip = 0; ip < numIp; ++ip ) {
    
    const int rowIpc = 2*ip;
    const int rowSfc = npe*ip;
    
    const double s1 = isoParCoord[rowIpc];
    const double s2 = isoParCoord[rowIpc+1];
    shpfc[rowSfc  ] = 0.25*(1.0-s1)*(1.0-s2);
    shpfc[rowSfc+1] = 0.25*(1.0+s1)*(1.0-s2);
    shpfc[rowSfc+2] = 0.25*(1.0+s1)*(1.0+s2);
    shpfc[rowSfc+3] = 0.25*(1.0-s1)*(1.0+s2);
 
  }
}

//--------------------------------------------------------------------------
//-------- general_normal --------------------------------------------------
//--------------------------------------------------------------------------
void
Quad3DSCS::general_normal(
  const double *isoParCoord,
  const double *coords,
  double *normal)
{
  const int nDim = 3;

  const double psi0Xi = -0.25 * (1.0 - isoParCoord[1]);
  const double psi1Xi =  0.25 * (1.0 - isoParCoord[1]);
  const double psi2Xi =  0.25 * (1.0 + isoParCoord[1]);
  const double psi3Xi = -0.25 * (1.0 + isoParCoord[1]);
  
  const double psi0Eta =-0.25 * (1.0 - isoParCoord[0]);
  const double psi1Eta =-0.25 * (1.0 + isoParCoord[0]);
  const double psi2Eta = 0.25 * (1.0 + isoParCoord[0]);
  const double psi3Eta = 0.25 * (1.0 - isoParCoord[0]);
  
  const double DxDxi = coords[0*nDim+0]*psi0Xi +
    coords[1*nDim+0]*psi1Xi +
    coords[2*nDim+0]*psi2Xi +
    coords[3*nDim+0]*psi3Xi;  
      
  const double DyDxi = coords[0*nDim+1]*psi0Xi +
    coords[1*nDim+1]*psi1Xi +
    coords[2*nDim+1]*psi2Xi +
    coords[3*nDim+1]*psi3Xi;
  
  const double DzDxi = coords[0*nDim+2]*psi0Xi +
    coords[1*nDim+2]*psi1Xi +
    coords[2*nDim+2]*psi2Xi +
    coords[3*nDim+2]*psi3Xi;
  
  const double DxDeta = coords[0*nDim+0]*psi0Eta +
    coords[1*nDim+0]*psi1Eta +
    coords[2*nDim+0]*psi2Eta +
    coords[3*nDim+0]*psi3Eta;

  const double DyDeta = coords[0*nDim+1]*psi0Eta +
    coords[1*nDim+1]*psi1Eta +
    coords[2*nDim+1]*psi2Eta +
    coords[3*nDim+1]*psi3Eta;
  
  const double DzDeta = coords[0*nDim+2]*psi0Eta +
    coords[1*nDim+2]*psi1Eta +
    coords[2*nDim+2]*psi2Eta +
    coords[3*nDim+2]*psi3Eta;
  
  const double detXY =  DxDxi*DyDeta - DxDeta*DyDxi;
  const double detYZ =  DyDxi*DzDeta - DyDeta*DzDxi;
  const double detXZ = -DxDxi*DzDeta + DxDeta*DzDxi;
  
  const double det = std::sqrt( detXY*detXY + detYZ*detYZ + detXZ*detXZ );
    
  normal[0] = detYZ / det;
  normal[1] = detXZ / det;
  normal[2] = detXY / det;
}

} // namespace nalu
} // namespace sierra
