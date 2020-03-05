#ifndef FSITURBINEN_H
#define FSITURBINEN_H

#include "OpenFAST.H"

#include <SurfaceFMPostProcessing.h>

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/Field.hpp"


#include <vector>
#include <string>
#include <array>

#include "yaml-cpp/yaml.h"

namespace sierra {

namespace nalu {

typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
typedef stk::mesh::Field<double> ScalarFieldType;
typedef stk::mesh::Field<int> ScalarIntFieldType;
typedef stk::mesh::Field<double, stk::mesh::SimpleArrayTag>  GenericFieldType;
typedef stk::mesh::Field<int, stk::mesh::SimpleArrayTag>  GenericIntFieldType;

class fsiTurbine
{

public:

    fsiTurbine(int iTurb, const YAML::Node &, stk::mesh::MetaData & meta, stk::mesh::BulkData & bulk);

    virtual ~fsiTurbine();

    void setup(SurfaceFMPostProcessing *sfm_pp = nullptr);

    void initialize();

    //! Convert pressure and viscous/turbulent stress on the turbine surface CFD mesh into a "fsiForce" field on the turbine surface CFD mesh
    void computeFSIforce() ;

    //! Map loads from the "fsiForce" field on the turbine surface CFD mesh into point load array that gets transferred to openfast
    void mapLoads();

    //! Transfer the deflections from the openfast nodes to the turbine surface CFD mesh. Will call 'computeDisplacement' for each node on the turbine surface CFD mesh.
    void mapDisplacements();

    //! Map each node on the turbine surface CFD mesh to blade beam mesh
    void computeMapping();

    bool exists_in_mesh(std::vector<double>&, double);
    
    //! Map each sub-control surface on the turbine surface CFD mesh to blade beam mesh
    void computeLoadMapping();
    
    //! Compute divergence of mesh velocity for use in "gcl" source term of
    //  conservation equations
    void compute_div_mesh_velocity();

    //! Set displacement corresponding to rotation at a constant rpm on the OpenFAST mesh before mapping to the turbine blade surface mesh
    void setRotationDisplacement(std::array<double,3> axis, double omega, double curTime);
    
    //! Set sample displacement on the OpenFAST mesh before mapping to the turbine blade surface mesh
    void setSampleDisplacement(double curTime=0.0);

    //! Set reference displacement on the turbine blade surface mesh, for comparison with Sample displacement set in setSampleDisplacement
    void setRefDisplacement(double curTime=0.0);

    //! Set the processor containing the turbine
    void setProc(int turbProc) {turbineProc_ = turbProc;}
    //! Get the processor containing the turbine
    int getProc() {return turbineProc_;}

    //! Get the part vector containing all the parts with mesh displacement
    stk::mesh::PartVector & getPartVec() { return partVec_;}

    //! Prepare netcdf file to write deflections and loads
    void prepare_nc_file(const int nTwrPts, const int nBlades, const int nTotBldPts);

    //! Write reference positions to netcdf file
    void write_nc_ref_pos();

    //! Write deflections and loads to netcdf file
    void write_nc_def_loads(const size_t tStep_, const double curTime);

    fast::turbineDataType params_;
    fast::turbBRfsiDataType brFSIdata_;
    std::vector<double> bld_dr_; 
    std::vector<std::array<double,2>> bld_rmm_; //Min-Max r for each node along blade

    //! Map of `{variableName : netCDF_ID}` obtained from the NetCDF C interface
    std::unordered_map<std::string, int> ncVarIDs_;

private:

    fsiTurbine() = delete;
    fsiTurbine(const fsiTurbine&) = delete;

    //! Populate the parts from the names for a given vector and put the fields into the parts
    void populateParts(std::vector<std::string> & partNames,
                       stk::mesh::PartVector & partVec,
                       stk::mesh::PartVector & allPartVec,
                       const std::string & turbinePart);

    //! Populate the boundary parts from the names for a given vector and put the fields into the parts
    void populateBndyParts(std::vector<std::string> & partNames,
                       stk::mesh::PartVector & partVec,
                       stk::mesh::PartVector & allPartVec,
                       const std::string & turbinePart);
    
    /** Project a point 'pt' onto a line from 'lStart' to 'lEnd' and return the non-dimensional location of the projected point along the line in [0-1] coordinates
        \f[
        nonDimCoord = \frac{ (\vec{pt} - \vec{lStart}) \cdot ( \vec{lEnd} - \vec{lStart} ) }{ (\vec{lEnd} - \vec{lStart}) \cdot (\vec{lEnd} - \vec{lStart}) }
        \f]
    */
    double projectPt2Line(std::vector<double> & pt, std::vector<double> & lStart, std::vector<double> & lEnd);

    /** Project a point 'pt' onto a line from 'lStart' to 'lEnd' and return the non-dimensional distance of 'pt' from the line w.r.t the distance from 'lStart' to 'lEnd'
        \f[
        \vec{perp} &= (\vec{pt} - \vec{lStart}) - \frac{ (\vec{pt} - \vec{lStart}) \cdot ( \vec{lEnd} - \vec{lStart} ) }{ (\vec{lEnd} - \vec{lStart}) \cdot (\vec{lEnd} - \vec{lStart}) } ( \vec{lEnd} - \vec{lStart} ) \\
        nonDimPerpDist = \frac{\lvert \vec{perp} \rvert}{ \lvert  (\vec{lEnd} - \vec{lStart}) \rvert }
        \f]
    */
    double perpProjectDist_Pt2Line(std::vector<double> & pt, std::vector<double> & lStart, std::vector<double> & lEnd);

    //! Compute the effective force and moment at the OpenFAST mesh node for a given force at the CFD surface mesh node
    void computeEffForceMoment(double *forceCFD, double *xyzCFD, double *forceMomOF, double *xyzOF);

    //! Compute the effective force and moment at the hub (can be any point) from a given mesh part vector
    void computeHubForceMomentForPart(std::vector<double> & hubForceMoment, std::vector<double> & hubPos, stk::mesh::PartVector part);


    //! Linearly interpolate dispInterp = dispStart + interpFac * (dispEnd - dispStart). Special considerations for Wiener-Milenkovic parameters
    void linInterpTotDisplacement(double *dispStart, double *dispEnd, double interpFac, double * dispInterp);

    //! Linearly interpolate velInterp = velStart + interpFac * (velEnd - velStart).
    void linInterpTotVelocity(double *velStart, double *velEnd, double interpFac, double * velInterp);
    //! Linearly interpolate between 3-dimensional vectors 'a' and 'b' with interpolating factor 'interpFac'
    void linInterpVec(double * a, double * b, double interpFac, double * aInterpb);

    /* Linearly interpolate the Wiener-Milenkovic parameters between 'qStart' and 'qEnd' into 'qInterp' with an interpolating factor 'interpFac'
       see O.A.Bauchau, 2011, Flexible Multibody Dynamics p. 649, section 17.2, Algorithm 1'
    */
    void linInterpRotation(double * qStart, double * qEnd, double interpFac, double * qInterp);

    //! Compose Wiener-Milenkovic parameters 'p' and 'q' into 'pPlusq'. If a transpose of 'p' is required, set tranposeP to '-1', else leave blank or set to '+1'
    void composeWM(double * p, double * q, double * pPlusq, double transposeP=1.0, double transposeQ=1.0);

    //! Convert one array of 6 deflections (transX, transY, transZ, wmX, wmY, wmZ) into one vector of translational displacement at a given node on the turbine surface CFD mesh.
    void computeDisplacement(double *totDispNode, double * xyzOF,  double *transDispNode, double * xyzCFD);

    //! Convert one array of 6 velocities (transX, transY, transZ, wmX, wmY, wmZ) into one vector of translational velocity at a given node on the turbine surface CFD mesh.
    void computeMeshVelocity(double *totVelNode, double * totDispNode, double * totPosOF,  double *transVelNode, double * xyzCFD);

    //! Split a force and moment into the surrounding 'left' and 'right' nodes in a variationally consistent manner using interpFac
    void splitForceMoment(double *totForceMoment, double interpFac, double *leftForceMoment, double *rightForceMoment);

    //! Apply a Wiener-Milenkovic rotation 'wm' to a vector 'r' into 'rRot'
    void applyWMrotation(double * wm, double * r, double *rRot, double transpose=1.0);

    //! Calculate the distance between 3-dimensional vectors 'a' and 'b'
    double calcDistanceSquared(double * a, double * b);

    //! Return the dot product of 3-dimensional vectors 'a' and 'b'
    double dot(double * a, double * b);

    //! Compute the cross product of 3-dimensional vectors 'a' and 'b' into 'aCrossb'
    void cross(double * a, double * b, double * aCrossb);

    //! Compute the error norm between two fields for a given part vector
    void compute_error_norm(VectorFieldType * vec, VectorFieldType * vec_ref, stk::mesh::PartVector partVec, std::vector<double> & err);

    int iTurb_; // Global turbine number

    stk::mesh::MetaData& meta_;
    stk::mesh::BulkData& bulk_;

    int turbineProc_; // The MPI rank containing the OpenFAST instance of the turbine
    bool turbineInProc_; // A boolean flag to determine if the processor contains any part of the turbine

    GenericIntFieldType * loadMap_; // Maps every node on the tower surface to the closest node of the openfast tower mesh element
    GenericFieldType * loadMapInterp_; // Maps every node on the tower surface to the closest node of the openfast tower mesh element
    ScalarIntFieldType * dispMap_; // Maps every node on the tower surface to the lower node of the openfast mesh element containing the projection of the tower surface node on to the openfast mesh tower element
    ScalarFieldType * dispMapInterp_; // The location of the CFD surface mesh node projected along the OpenFAST mesh element in non-dimensional [0,1] co-ordinates.
    int nBlades_; // Number of blades in the turbine

    //! Fields containing the FSI force at all nodes on the turbine surface
    VectorFieldType *pressureForceSCS_;
    VectorFieldType *tauWallSCS_;

    // Volume mesh parts and part names
    //! Part name of the tower
    std::vector<std::string> twrPartNames_;
    //! Pointer to tower part
    stk::mesh::PartVector twrParts_;
    //! Part name of the hub
    std::vector<std::string> hubPartNames_;
    //! Pointer to hub part
    stk::mesh::PartVector hubParts_;
    //! Part name of the nacelle
    std::vector<std::string> nacellePartNames_;
    //! Pointer to the nacelle part
    stk::mesh::PartVector nacelleParts_;
    //! Part names of the blades
    std::vector<std::vector<std::string>> bladePartNames_;
    //! Pointers to the blade parts
    std::vector<stk::mesh::PartVector> bladeParts_;
    //! Part vector over all parts applying a mesh displacement
    stk::mesh::PartVector partVec_;

    // Boundary mesh parts and part names
    //! Part name of the tower
    std::vector<std::string> twrBndyPartNames_;
    //! Pointer to tower part
    stk::mesh::PartVector twrBndyParts_;
    //! Part name of the hub
    std::vector<std::string> hubBndyPartNames_;
    //! Pointer to hub part
    stk::mesh::PartVector hubBndyParts_;
    //! Part name of the nacelle
    std::vector<std::string> nacelleBndyPartNames_;
    //! Pointer to the nacelle part
    stk::mesh::PartVector nacelleBndyParts_;
    //! Part names of the blades
    std::vector<std::vector<std::string>> bladeBndyPartNames_;
    //! Pointers to the blade parts
    std::vector<stk::mesh::PartVector> bladeBndyParts_;
    //! Part vector over all wall boundary parts applying a mesh displacement
    stk::mesh::PartVector bndyPartVec_;

};

} // nalu

} // sierra

#endif /* FSITURBINE_H */
