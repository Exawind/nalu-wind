#include "FSIturbine.h"

#include "SurfaceFMPostProcessing.h"
#include <nalu_make_unique.h>
#include "utils/ComputeVectorDivergence.h"

#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/Field.hpp"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"

#include "netcdf.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace sierra {

namespace nalu {

inline void check_nc_error(int code, std::string msg) {
    if (code != 0)
        throw std::runtime_error("BdyLayerStatistics:: NetCDF error: " + msg);
}
    
fsiTurbine::fsiTurbine(
    int iTurb,
    const YAML::Node & node,
    stk::mesh::MetaData & meta,
    stk::mesh::BulkData & bulk
) :
iTurb_(iTurb),
meta_(meta),
bulk_(bulk),
turbineProc_(-1),
turbineInProc_(false),
loadMap_(NULL),
dispMap_(NULL),
dispMapInterp_(NULL),
pressureForceSCS_(NULL),
tauWallSCS_(NULL)
{

    if(node["tower_parts"]) {
        const auto& tparts = node["tower_parts"];
        twrPartNames_ = tparts.as<std::vector<std::string>>();
    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Tower part name(s) not specified for turbine " << iTurb_ << std::endl;
    }
    if(node["nacelle_parts"]) {
        const auto& nparts = node["nacelle_parts"];
        nacellePartNames_ = nparts.as<std::vector<std::string>>();
    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Nacelle part name(s) not specified for turbine " << iTurb_ << std::endl;
    }
    if(node["hub_parts"]) {
        const auto& hparts = node["hub_parts"];
        hubPartNames_ = hparts.as<std::vector<std::string>>();
    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Hub part name(s) not specified for turbine " << iTurb_ << std::endl;
    }
    if(node["blade_parts"]) {
        const auto& bparts = node["blade_parts"];
        nBlades_ = bparts.size();
        bladePartNames_.resize(nBlades_);
        bladeParts_.resize(nBlades_);
        for(int iBlade = 0; iBlade < nBlades_; iBlade++){
            const auto& bpart = bparts[iBlade];
            bladePartNames_[iBlade]= bpart.as<std::vector<std::string>>();
         }

    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Blade part names not specified for turbine " << iTurb_ << std::endl;
    }

    if(node["tower_boundary_parts"]) {
        const auto& tparts = node["tower_boundary_parts"];
        twrBndyPartNames_ = tparts.as<std::vector<std::string>>();
    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Tower part name(s) not specified for turbine " << iTurb_ << std::endl;
    }
    if(node["nacelle_boundary_parts"]) {
        const auto& nparts = node["nacelle_boundary_parts"];
        nacelleBndyPartNames_ = nparts.as<std::vector<std::string>>();
    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Nacelle boundary part name(s) not specified for turbine " << iTurb_ << std::endl;
    }
    if(node["hub_boundary_parts"]) {
        const auto& hparts = node["hub_boundary_parts"];
        hubBndyPartNames_ = hparts.as<std::vector<std::string>>();
    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Hub boundary part name(s) not specified for turbine " << iTurb_ << std::endl;
    }
    if(node["blade_boundary_parts"]) {
        const auto& bparts = node["blade_boundary_parts"];
        nBlades_ = bparts.size();
        bladeBndyPartNames_.resize(nBlades_);
        bladeBndyParts_.resize(nBlades_);
        for(int iBlade = 0; iBlade < nBlades_; iBlade++){
            const auto& bpart = bparts[iBlade];
            bladeBndyPartNames_[iBlade]= bpart.as<std::vector<std::string>>();
         }

    } else if (! bulk_.parallel_rank() ) {
        std::cout << "Blade boundary part names not specified for turbine " << iTurb_ << std::endl;
    }

    loadMap_ = meta_.get_field<GenericIntFieldType>(
        meta_.side_rank(), "load_map");
    if (loadMap_ == NULL)
        loadMap_ =  &(meta_.declare_field<GenericIntFieldType>(
                          meta_.side_rank(), "load_map"));

    loadMapInterp_ = meta_.get_field<GenericFieldType>(
        meta_.side_rank(), "load_map_interp");
    if (loadMapInterp_ == NULL)
        loadMapInterp_ =  &(meta_.declare_field<GenericFieldType>(
                          meta_.side_rank(), "load_map_interp"));
    
    dispMap_ = meta_.get_field<ScalarIntFieldType>(
         stk::topology::NODE_RANK, "disp_map");
    if (dispMap_ == NULL)
        dispMap_ =  &(meta_.declare_field<ScalarIntFieldType>(
                          stk::topology::NODE_RANK, "disp_map"));

    dispMapInterp_ = meta_.get_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "disp_map_interp");
    if (dispMapInterp_ == NULL)
        dispMapInterp_ =  &(meta_.declare_field<ScalarFieldType>(
                                stk::topology::NODE_RANK, "disp_map_interp"));


    ScalarFieldType * div_mesh_vel = meta_.get_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "div_mesh_velocity");
    if (div_mesh_vel == NULL)
        div_mesh_vel = &(meta_.declare_field<ScalarFieldType>(
                             stk::topology::NODE_RANK, "div_mesh_velocity"));

    pressureForceSCS_ = meta_.get_field<VectorFieldType>(
        meta_.side_rank(), "pressure_force_scs");
    if (pressureForceSCS_ == NULL)
        pressureForceSCS_ = &(meta_.declare_field<VectorFieldType>(
                                  meta_.side_rank(), "pressure_force_scs"));

    tauWallSCS_ = meta_.get_field<VectorFieldType>(
       meta.side_rank(), "tau_wall_scs");
    if (tauWallSCS_ == NULL) //Still null, declare your own field
        tauWallSCS_ =  &(meta_.declare_field<VectorFieldType>(
                             meta_.side_rank(), "tau_wall_scs"));

}

fsiTurbine::~fsiTurbine() {

    //Nothing to do here so far

}

void fsiTurbine::populateParts(std::vector<std::string> & partNames,
                               stk::mesh::PartVector & partVec,
                               stk::mesh::PartVector & allPartVec,
                               const std::string & turbinePart) {

    for (auto pName: partNames) {
        stk::mesh::Part* part = meta_.get_part(pName);
        if (nullptr == part) {
            throw std::runtime_error("fsiTurbine:: No part found for " + turbinePart + " mesh part corresponding to " + pName);
        } else {
            partVec.push_back(part);
            allPartVec.push_back(part);
        }

        if ( ! bulk_.parallel_rank() )
            std::cout << "Adding part " << pName << " to " << turbinePart
                      << std::endl ;

        stk::mesh::put_field_on_mesh(*dispMap_, *part, 1, nullptr);
        stk::mesh::put_field_on_mesh(*dispMapInterp_, *part, 1, nullptr);
    }

}

void fsiTurbine::populateBndyParts(std::vector<std::string> & partNames,
                                   stk::mesh::PartVector & partVec,
                                   stk::mesh::PartVector & allPartVec,
                                   const std::string & turbinePart) {

    for (auto pName: partNames) {
        stk::mesh::Part* part = meta_.get_part(pName);
        if (nullptr == part) {
            throw std::runtime_error("fsiTurbine:: No part found for " + turbinePart + " mesh part corresponding to " + pName);
        } else {
            partVec.push_back(part);
            allPartVec.push_back(part);
        }

        if ( ! bulk_.parallel_rank() )
            std::cout << "Adding part " << pName << " to " << turbinePart
                      << std::endl ;

        stk::mesh::put_field_on_mesh(*loadMap_, *part, 1, nullptr);
        stk::mesh::put_field_on_mesh(*loadMapInterp_, *part, 1, nullptr);        
    }
    
}

void fsiTurbine::setup(SurfaceFMPostProcessing* sfm_pp) {

    //TODO: Check if any part of the turbine surface is on this processor and set turbineInProc_ to True/False

    //TODO:: Figure out a way to check the consistency between the number of blades specified in the Nalu input file and the number of blades in the OpenFAST model.

    ScalarFieldType * div_mesh_vel = meta_.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity");
    stk::mesh::put_field_on_mesh(*div_mesh_vel, meta_.universal_part(), 1, nullptr);

    populateParts(twrPartNames_, twrParts_, partVec_, "Tower");
    populateParts(nacellePartNames_, nacelleParts_, partVec_, "Nacelle");
    populateParts(hubPartNames_, hubParts_, partVec_, "Hub");
    for (int i=0; i < nBlades_; i++)
        populateParts(bladePartNames_[i], bladeParts_[i], partVec_, "Blade "+std::to_string(i));

    populateBndyParts(twrBndyPartNames_, twrBndyParts_, bndyPartVec_, "Tower");
    populateBndyParts(nacelleBndyPartNames_, nacelleBndyParts_,
                  bndyPartVec_, "Nacelle");
    populateBndyParts(hubBndyPartNames_, hubBndyParts_, bndyPartVec_, "Hub");
    for (int i=0; i < nBlades_; i++)
        populateBndyParts(bladeBndyPartNames_[i], bladeBndyParts_[i],
                      bndyPartVec_, "Blade "+std::to_string(i));

    if (sfm_pp != nullptr) {
        std::vector<std::string> allPartNames;
        allPartNames.insert(allPartNames.end(),
                            twrBndyPartNames_.begin(), twrBndyPartNames_.end());
        allPartNames.insert(allPartNames.end(),
                            nacelleBndyPartNames_.begin(), nacelleBndyPartNames_.end());
        allPartNames.insert(allPartNames.end(),
                            hubBndyPartNames_.begin(), hubBndyPartNames_.end());
        for (int i=0; i < nBlades_; i++)
            allPartNames.insert(allPartNames.end(),
                                bladeBndyPartNames_[i].begin(), bladeBndyPartNames_[i].end());
        
        SurfaceFMData sfm_pp_data(allPartNames,
                                  "turbine"+std::to_string(iTurb_)+"_forces",
                                  iTurb_);
        sfm_pp->register_surface_pp(sfm_pp_data);
    }

}

void fsiTurbine::initialize() {

    //Allocate memory for loads and deflections data

    int nTwrPts = params_.nBRfsiPtsTwr;
    int nBlades = params_.numBlades;
    int nTotBldPts = 0;
    for (int i=0; i < nBlades; i++)
        nTotBldPts += params_.nBRfsiPtsBlade[i];
    brFSIdata_.twr_ref_pos.resize(6*nTwrPts);
    brFSIdata_.twr_def.resize(6*nTwrPts);
    brFSIdata_.twr_vel.resize(6*nTwrPts);
    brFSIdata_.twr_ld.resize(6*nTwrPts);
    brFSIdata_.bld_rloc.resize(nTotBldPts);
    brFSIdata_.bld_chord.resize(nTotBldPts);
    brFSIdata_.bld_ref_pos.resize(6*nTotBldPts);
    brFSIdata_.bld_def.resize(6*nTotBldPts);
    brFSIdata_.bld_vel.resize(6*nTotBldPts);
    brFSIdata_.bld_ld.resize(6*nTotBldPts);
    brFSIdata_.hub_ref_pos.resize(6);
    brFSIdata_.hub_def.resize(6);
    brFSIdata_.hub_vel.resize(6);
    brFSIdata_.nac_ref_pos.resize(6);
    brFSIdata_.nac_def.resize(6);
    brFSIdata_.nac_vel.resize(6);

    bld_dr_.resize(nTotBldPts);
    bld_rmm_.resize(nTotBldPts);
    
}

void fsiTurbine::prepare_nc_file(const int nTwrPts, const int nBlades, const int nTotBldPts) {

    const int iproc = bulk_.parallel_rank();
    if ( iproc != turbineProc_) return;

    int ncid, n_dim, n_tsteps, n_twr_nds, n_blds, n_bld_nds, varid;
    int ierr;

    int nBldPts = nTotBldPts/nBlades;
    
    //Create the file
    std::stringstream defloads_fstream;
    defloads_fstream << "turb_" ;
    defloads_fstream << std::setfill('0') << std::setw(2) << iTurb_;
    defloads_fstream << "_deflloads.nc";
    std::string defloads_filename = defloads_fstream.str();
    ierr = nc_create(defloads_filename.c_str(), NC_CLOBBER, &ncid);
    check_nc_error(ierr, "nc_create");

    //Define dimensions
    ierr = nc_def_dim(ncid, "n_dim", 3, &n_dim);
    ierr = nc_def_dim(ncid, "n_tsteps", NC_UNLIMITED, &n_tsteps);
    ierr = nc_def_dim(ncid, "n_twr_nds", nTwrPts, &n_twr_nds);
    ierr = nc_def_dim(ncid,"n_blds", nBlades, &n_blds);
    ierr = nc_def_dim(ncid, "n_bld_nds", nBldPts, &n_bld_nds);

    const std::vector<int> twrRefDims{n_dim, n_twr_nds};
    const std::vector<int> twrDefLoadsDims{n_tsteps, n_dim, n_twr_nds};
    //const std::vector<int> bldRootRefDims{n_dim, n_blds};
    //const std::vector<int> bldRootDefDim{n_tsteps, n_dim, n_blds};
    const std::vector<int> bldParamDims{n_blds, n_bld_nds};
    const std::vector<int> bldRefDims{n_blds, n_dim, n_bld_nds};
    const std::vector<int> bldDefLoadsDims{n_tsteps, n_blds, n_dim, n_bld_nds};
    const std::vector<int> ptRefDims{n_dim};
    const std::vector<int> ptDefLoadsDims{n_tsteps, n_dim};

    //Now define variables
    ierr = nc_def_var(ncid, "time", NC_DOUBLE, 1, &n_tsteps, &varid);
    ncVarIDs_["time"] = varid;

    ierr = nc_def_var(ncid, "twr_ref_pos", NC_DOUBLE, 2, twrRefDims.data(), &varid);
    ncVarIDs_["twr_ref_pos"] = varid;
    ierr = nc_def_var(ncid, "twr_ref_orient", NC_DOUBLE, 2, twrRefDims.data(), &varid);
    ncVarIDs_["twr_ref_orient"] = varid;
    ierr = nc_def_var(ncid, "bld_chord", NC_DOUBLE, 2, bldParamDims.data(), &varid);
    ncVarIDs_["bld_chord"] = varid;
    ierr = nc_def_var(ncid, "bld_rloc", NC_DOUBLE, 2, bldParamDims.data(), &varid);
    ncVarIDs_["bld_rloc"] = varid;
    ierr = nc_def_var(ncid, "bld_ref_pos", NC_DOUBLE, 3, bldRefDims.data(), &varid);
    ncVarIDs_["bld_ref_pos"] = varid;
    ierr = nc_def_var(ncid, "bld_ref_orient", NC_DOUBLE, 3, bldRefDims.data(), &varid);
    ncVarIDs_["bld_ref_orient"] = varid;
    ierr = nc_def_var(ncid, "hub_ref_pos", NC_DOUBLE, 1, ptRefDims.data(), &varid);
    ncVarIDs_["hub_ref_pos"] = varid;
    ierr = nc_def_var(ncid, "hub_ref_orient", NC_DOUBLE, 1, ptRefDims.data(), &varid);
    ncVarIDs_["hub_ref_orient"] = varid;
    ierr = nc_def_var(ncid, "nac_ref_pos", NC_DOUBLE, 1, ptRefDims.data(), &varid);
    ncVarIDs_["nac_ref_pos"] = varid;
    ierr = nc_def_var(ncid, "nac_ref_orient", NC_DOUBLE, 1, ptRefDims.data(), &varid);
    ncVarIDs_["nac_ref_orient"] = varid;

    ierr = nc_def_var(ncid, "twr_disp", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
    ncVarIDs_["twr_disp"] = varid;
    ierr = nc_def_var(ncid, "twr_orient", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
    ncVarIDs_["twr_orient"] = varid;
    ierr = nc_def_var(ncid, "twr_vel", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
    ncVarIDs_["twr_vel"] = varid;
    ierr = nc_def_var(ncid, "twr_rotvel", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
    ncVarIDs_["twr_rotvel"] = varid;
    ierr = nc_def_var(ncid, "twr_ld", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
    ncVarIDs_["twr_ld"] = varid;
    ierr = nc_def_var(ncid, "twr_moment", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
    ncVarIDs_["twr_moment"] = varid;

    ierr = nc_def_var(ncid, "bld_disp", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
    ncVarIDs_["bld_disp"] = varid;
    ierr = nc_def_var(ncid, "bld_orient", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
    ncVarIDs_["bld_orient"] = varid;
    ierr = nc_def_var(ncid, "bld_vel", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
    ncVarIDs_["bld_vel"] = varid;
    ierr = nc_def_var(ncid, "bld_rotvel", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
    ncVarIDs_["bld_rotvel"] = varid;
    ierr = nc_def_var(ncid, "bld_ld", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
    ncVarIDs_["bld_ld"] = varid;
    ierr = nc_def_var(ncid, "bld_ld_loc", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
    ncVarIDs_["bld_ld_loc"] = varid;
    ierr = nc_def_var(ncid, "bld_moment", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
    ncVarIDs_["bld_moment"] = varid;
    
    ierr = nc_def_var(ncid, "hub_disp", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
    ncVarIDs_["hub_disp"] = varid;
    ierr = nc_def_var(ncid, "hub_orient", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
    ncVarIDs_["hub_orient"] = varid;
    ierr = nc_def_var(ncid, "hub_vel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
    ncVarIDs_["hub_vel"] = varid;
    ierr = nc_def_var(ncid, "hub_rotvel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
    ncVarIDs_["hub_rotvel"] = varid;
    
    ierr = nc_def_var(ncid, "nac_disp", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);    
    ncVarIDs_["nac_disp"] = varid;
    ierr = nc_def_var(ncid, "nac_orient", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);    
    ncVarIDs_["nac_orient"] = varid;
    ierr = nc_def_var(ncid, "nac_vel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
    ncVarIDs_["nac_vel"] = varid;
    ierr = nc_def_var(ncid, "nac_rotvel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
    ncVarIDs_["nac_rotvel"] = varid;

    //! Indicate that we are done defining variables, ready to write data
    ierr = nc_enddef(ncid);
    check_nc_error(ierr, "nc_enddef");
    ierr = nc_close(ncid);
    check_nc_error(ierr, "nc_close");
    
}

void fsiTurbine::write_nc_ref_pos() {

    const int iproc = bulk_.parallel_rank();
    if ( iproc != turbineProc_) return;

    int nTwrPts = params_.nBRfsiPtsTwr;
    int nBlades = params_.numBlades;
    int nTotBldPts = 0;
    for (int i=0; i < nBlades; i++) {
        nTotBldPts += params_.nBRfsiPtsBlade[i];
    }
    int nBldPts = nTotBldPts/nBlades;
    int ncid, ierr;

    std::stringstream defloads_fstream;
    defloads_fstream << "turb_" ;
    defloads_fstream << std::setfill('0') << std::setw(2) << iTurb_;
    defloads_fstream << "_deflloads.nc";
    std::string defloads_filename = defloads_fstream.str();
    ierr = nc_open(defloads_filename.c_str(), NC_WRITE, &ncid);
    check_nc_error(ierr, "nc_open");
    ierr = nc_enddef(ncid);

    std::vector<double> tmpArray;


    tmpArray.resize(nTwrPts);
    {
        std::vector<size_t> count_dim{1,nTwrPts};
        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_ref_pos[i*6+idim] ;
            std::vector<size_t> start_dim{idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_ref_pos"], start_dim.data(), count_dim.data(), tmpArray.data());
        }
        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_ref_pos[i*6+3+idim] ;
            std::vector<size_t> start_dim{idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_ref_orient"], start_dim.data(), count_dim.data(), tmpArray.data());
        }
    }

    tmpArray.resize(nBldPts);
    {
        std::vector<size_t> count_dim{1,1,nBldPts};
        for (auto iDim=0;iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_ref_pos[(iStart*6)+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_ref_pos"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }
        for (auto iDim=0;iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_ref_pos[(iStart*6)+iDim+3];
                    iStart++;
                }
                std::vector<size_t> start_dim{iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_ref_orient"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }

        std::vector<size_t> param_count_dim{1,nBldPts};
        int iStart = 0 ;
        for (auto iBlade=0; iBlade < nBlades; iBlade++) {
            for (auto i=0; i < nBldPts; i++) {
                tmpArray[i] = brFSIdata_.bld_chord[iStart];
                iStart++;
            }
            std::vector<size_t> start_dim{iBlade,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_chord"], start_dim.data(), param_count_dim.data(), tmpArray.data());
        }
        iStart = 0 ;
        for (auto iBlade=0; iBlade < nBlades; iBlade++) {
            for (auto i=0; i < nBldPts; i++) {
                tmpArray[i] = brFSIdata_.bld_rloc[iStart];
                iStart++;
            }
            std::vector<size_t> start_dim{iBlade,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_rloc"], start_dim.data(), param_count_dim.data(), tmpArray.data());
        }
    }

    ierr = nc_put_var_double(ncid, ncVarIDs_["nac_ref_pos"], &brFSIdata_.nac_ref_pos[0]);
    ierr = nc_put_var_double(ncid, ncVarIDs_["nac_ref_orient"], &brFSIdata_.nac_ref_pos[3]);
    
    ierr = nc_put_var_double(ncid, ncVarIDs_["hub_ref_pos"], &brFSIdata_.hub_ref_pos[0]);
    ierr = nc_put_var_double(ncid, ncVarIDs_["hub_ref_orient"], &brFSIdata_.hub_ref_pos[3]);
    
    ierr = nc_close(ncid);
    
}
 

void fsiTurbine::write_nc_def_loads(const size_t tStep, const double curTime) {

    const int iproc = bulk_.parallel_rank();
    if ( iproc != turbineProc_) return;

    size_t nTwrPts = params_.nBRfsiPtsTwr;
    size_t nBlades = params_.numBlades;
    size_t nTotBldPts = 0;
    for (auto i=0; i < nBlades; i++)
        nTotBldPts += params_.nBRfsiPtsBlade[i];
    size_t nBldPts = nTotBldPts/nBlades;
    
    int ncid, ierr;

    std::stringstream defloads_fstream;
    defloads_fstream << "turb_" ;
    defloads_fstream << std::setfill('0') << std::setw(2) << iTurb_;
    defloads_fstream << "_deflloads.nc";
    std::string defloads_filename = defloads_fstream.str();
    ierr = nc_open(defloads_filename.c_str(), NC_WRITE, &ncid);
    check_nc_error(ierr, "nc_open");
    ierr = nc_enddef(ncid);

    int iStart = 0;
    for (int iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i=1 ; i < nBldPts-1; i++) {
            brFSIdata_.bld_ld[(i + iStart)*6+4] = (0.5 * (brFSIdata_.bld_rloc[i + iStart + 1] - brFSIdata_.bld_rloc[i + iStart - 1]) );
        }
        brFSIdata_.bld_ld[(iStart)*6+4] = (0.5 * (brFSIdata_.bld_rloc[iStart + 1] - brFSIdata_.bld_rloc[iStart]));
        brFSIdata_.bld_ld[(iStart + nBldPts-1)*6+4] = (0.5 * (brFSIdata_.bld_rloc[iStart + nBldPts - 1] - brFSIdata_.bld_rloc[iStart + nBldPts - 2]));
        iStart += nBldPts;
    }
    
    std::ofstream csvOut;
    csvOut.open("defloads.csv", std::ofstream::out);
    csvOut << "rloc, x, y, z, ld_x, ld_y, ld_z, area, chord, dr" << std::endl;
    for (auto i=0; i < nTotBldPts; i++) {
        csvOut << brFSIdata_.bld_rloc[i] << ", " << brFSIdata_.bld_ref_pos[i*6] + brFSIdata_.bld_def[i*6] << ", " << brFSIdata_.bld_ref_pos[i*6+1] + brFSIdata_.bld_def[i*6+1] << ", " << brFSIdata_.bld_ref_pos[i*6+2] + brFSIdata_.bld_def[i*6+2] << ", " ;
        csvOut << brFSIdata_.bld_ld[i*6] << ", " << brFSIdata_.bld_ld[i*6+1] << ", " << brFSIdata_.bld_ld[i*6+2] << ", " << brFSIdata_.bld_ld[i*6+3] << ", " << brFSIdata_.bld_chord[i] << ", " << brFSIdata_.bld_ld[i*6+4] << std::endl ;
    }
    csvOut.close();
    
    size_t count0=1;
    const std::vector<size_t> twrDefLoadsDims{1, 6*nTwrPts};
    //const std::vector<size_t> bldRootDefLoadsDims{1, 3*6*nBlades};
    const std::vector<size_t> bldDefLoadsDims{1, 6*nTotBldPts};
    const std::vector<size_t> ptDefLoadsDims{1, 6};
    
    ierr = nc_put_vara_double(ncid, ncVarIDs_["time"], &tStep, &count0, &curTime);

    std::vector<double> tmpArray;

    tmpArray.resize(nTwrPts);
    {
        std::vector<size_t> count_dim{1,1,nTwrPts};
        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_def[i*6+idim] ;
            std::vector<size_t> start_dim{tStep,idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_disp"], start_dim.data(), count_dim.data(), tmpArray.data());
        }
        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_def[i*6+3+idim] ;
            std::vector<size_t> start_dim{tStep,idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_orient"], start_dim.data(), count_dim.data(), tmpArray.data());
        }
        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_vel[i*6+idim] ;
            std::vector<size_t> start_dim{tStep,idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_vel"], start_dim.data(), count_dim.data(), tmpArray.data());
        }
        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_def[i*6+3+idim] ;
            std::vector<size_t> start_dim{tStep,idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_rotvel"], start_dim.data(), count_dim.data(), tmpArray.data());
        }

        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_ld[i*6+idim] ;
            std::vector<size_t> start_dim{tStep,idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_ld"], start_dim.data(), count_dim.data(), tmpArray.data());
        }
        for (auto idim=0;idim < 3; idim++) {
            for (auto i=0; i < nTwrPts; i++)
                tmpArray[i] = brFSIdata_.twr_ld[i*6+3+idim];
            std::vector<size_t> start_dim{tStep,idim,0};
            ierr = nc_put_vara_double(ncid, ncVarIDs_["twr_moment"], start_dim.data(), count_dim.data(), tmpArray.data());
        }
    }

    tmpArray.resize(nBldPts);
    {
        std::vector<size_t> count_dim{1,1,1,nBldPts};
        for (auto iDim=0;iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_def[(iStart*6)+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{tStep,iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_disp"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }
        for (auto iDim=0;iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_def[(iStart*6)+3+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{tStep,iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_orient"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }
        for (auto iDim=0;iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_vel[(iStart*6)+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{tStep,iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_vel"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }
        for (auto iDim=0; iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_vel[(iStart*6)+3+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{tStep,iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_rotvel"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }
        for (auto iDim=0;iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_ld[(iStart*6)+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{tStep,iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_ld"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }

        std::vector<double> ld_loc(3*nTotBldPts,0.0);
        for (auto i=0; i < nTotBldPts; i++) {
            applyWMrotation(&brFSIdata_.bld_def[i*6+3], &brFSIdata_.bld_ld[i*6], &ld_loc[i*3]);
        }
        for (auto iDim=0;iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = ld_loc[iStart*3+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{tStep,iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_ld_loc"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }

        for (auto iDim=0; iDim < 3; iDim++) {
            int iStart = 0 ;
            for (auto iBlade=0; iBlade < nBlades; iBlade++) {
                for (auto i=0; i < nBldPts; i++) {
                    tmpArray[i] = brFSIdata_.bld_ld[(iStart*6)+3+iDim];
                    iStart++;
                }
                std::vector<size_t> start_dim{tStep,iBlade,iDim,0};
                ierr = nc_put_vara_double(ncid, ncVarIDs_["bld_moment"], start_dim.data(), count_dim.data(), tmpArray.data());
            }
        }
        
    }
    
    std::vector<size_t> start_dim{tStep, 0};
    std::vector<size_t> count_dim{1,3};
    ierr = nc_put_vara_double(ncid, ncVarIDs_["hub_disp"], start_dim.data(), count_dim.data(), &brFSIdata_.hub_def[0]);
    ierr = nc_put_vara_double(ncid, ncVarIDs_["hub_orient"], start_dim.data(), count_dim.data(), &brFSIdata_.hub_def[3]);
    ierr = nc_put_vara_double(ncid, ncVarIDs_["hub_vel"], start_dim.data(), count_dim.data(), &brFSIdata_.hub_vel[0]);
    ierr = nc_put_vara_double(ncid, ncVarIDs_["hub_rotvel"], start_dim.data(), ptDefLoadsDims.data(), &brFSIdata_.hub_vel[3]);

    ierr = nc_put_vara_double(ncid, ncVarIDs_["nac_disp"], start_dim.data(), count_dim.data(), &brFSIdata_.nac_def[0]);
    ierr = nc_put_vara_double(ncid, ncVarIDs_["nac_orient"], start_dim.data(), count_dim.data(), &brFSIdata_.nac_def[3]);
    ierr = nc_put_vara_double(ncid, ncVarIDs_["nac_vel"], start_dim.data(), count_dim.data(), &brFSIdata_.nac_vel[0]);
    ierr = nc_put_vara_double(ncid, ncVarIDs_["nac_rotvel"], start_dim.data(), ptDefLoadsDims.data(), &brFSIdata_.nac_vel[3]);
    
    ierr = nc_close(ncid);
    
}

//! Convert pressure and viscous/turbulent stress on the turbine surface CFD mesh into a "fsiForce" field on the turbine surface CFD mesh
void fsiTurbine::computeFSIforce() {


}

//! Map loads from the "fsiForce" field on the turbine surface CFD mesh into point load array that gets transferred to openfast
void fsiTurbine::mapLoads() {

    //To implement this function - assume that 'loadMap_' field contains the node id along the blade or the tower that will accumulate the load corresponding to the node on the CFD surface mesh

    //First zero out forces on the OpenFAST mesh
    for (size_t i=0; i < params_.nBRfsiPtsTwr; i++) {
        for (size_t j=0; j < 6; j++)
            brFSIdata_.twr_ld[i*6+j] = 0.0;
    }

    int nBlades = params_.numBlades;
    int iRunTot = 0;
    for (size_t iBlade=0; iBlade < nBlades; iBlade++) {
        int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
        for (size_t i=0; i < nPtsBlade; i++) {
            for (size_t j=0; j < 6; j++)
                brFSIdata_.bld_ld[iRunTot*6+j] = 0.0;
            iRunTot++;
        }

    }

    // Now map loads
    const int ndim = meta_.spatial_dimension();
    VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");
    VectorFieldType* meshDisp = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_displacement");
            
    // nodal fields to gather and store at ip's
    std::vector<double> ws_face_shape_function;
    
    std::vector<double> ws_coordinates;
    std::vector<double> coord_bip(3,0.0);
    std::vector<double> coordref_bip(3,0.0);
    
    std::array<double,3> face_center;
    std::array<double,3> face_area;

    std::array<double,3> tforce_bip;
        
    std::vector<double> tmpNodePos(3,0.0); // Vector to temporarily store a position vector
    std::vector<double> tmpNodeDisp(3,0.0); // Vector to temporarily store a displacement vector

    // Do the tower first
    stk::mesh::Selector sel(meta_.locally_owned_part() & stk::mesh::selectUnion(twrBndyParts_) );
    const auto& bkts =  bulk_.get_buckets( meta_.side_rank(), sel );
    for (auto b: bkts) {
        // face master element
        MasterElement *meFC =
            MasterElementRepo::get_surface_master_element(b->topology());
        const int nodesPerFace = meFC->nodesPerElement_;
        const int numScsBip = meFC->num_integration_points();

        // mapping from ip to nodes for this ordinal;
        // face perspective (use with face_node_relations)
        const int *faceIpNodeMap = meFC->ipNodeMap();

        ws_face_shape_function.resize(numScsBip*nodesPerFace);
        meFC->shape_fcn(ws_face_shape_function.data());
        
        ws_coordinates.resize(ndim*nodesPerFace);
        
        for (size_t in=0; in < b->size(); in++) {
            // get face
            stk::mesh::Entity face = (*b)[in];
            // face node relations
            stk::mesh::Entity const * face_node_rels = bulk_.begin_nodes(face);
            // gather nodal data off of face
            for ( int ni = 0; ni < nodesPerFace; ++ni ) {
                stk::mesh::Entity node = face_node_rels[ni];
                // gather coordinates
                const double* xyz = stk::mesh::field_data(*modelCoords, node);
                const double* xyz_disp = stk::mesh::field_data(*meshDisp, node);
                for (auto i=0; i < ndim; i++) {
                    ws_coordinates[ni*ndim+i] = xyz[i] + xyz_disp[i];
                }
            }
            
            // Get reference to load map and loadMapInterp at all ips on this face
            const int* loadMapFace = stk::mesh::field_data(*loadMap_, face);
            const double* loadMapInterpFace = stk::mesh::field_data(*loadMapInterp_, face);
            const double* pforce = stk::mesh::field_data(*pressureForceSCS_, face);
            const double* vforce = stk::mesh::field_data(*tauWallSCS_, face);
            
            for ( int ip = 0; ip < numScsBip; ++ip ) {
                //Get coordinates and pressure force at this ip
                for (auto i=0; i < ndim; i++) {
                    coord_bip[i] = 0.0;
                }
                for (int ni = 0; ni < nodesPerFace; ni++) {
                    const double r = ws_face_shape_function[ip*nodesPerFace + ni];
                    for (int i=0; i < ndim; i++) {
                        coord_bip[i] += r * ws_coordinates[ni*ndim+i];
                    }
                }

                const int loadMap_bip = loadMapFace[ip];
                const double loadMapInterp_bip = loadMapInterpFace[ip];

                for (auto idim=0; idim < 3; idim++)
                    tforce_bip[idim] = pforce[ip*3+idim] + vforce[ip*3+idim];

                //Find the interpolated reference position first
                linInterpVec(&brFSIdata_.twr_ref_pos[(loadMap_bip)*6],
                             &brFSIdata_.twr_ref_pos[(loadMap_bip + 1)*6],
                             loadMapInterp_bip, tmpNodePos.data());
                //Find the interpolated linear displacement 
                linInterpVec(&brFSIdata_.twr_def[(loadMap_bip)*6],
                             &brFSIdata_.twr_def[(loadMap_bip + 1)*6],
                             loadMapInterp_bip, tmpNodeDisp.data());
                //Add displacement to find actual position
                for (auto idim=0; idim < 3; idim++)
                    tmpNodePos[idim] += tmpNodeDisp[idim];

                // Temporarily store total force and moment as (fX, fY, fZ, mX, mY, mZ)
                std::vector<double> tmpForceMoment(6,0.0);
                //Now compute the force and moment on the interpolated reference
                //position
                computeEffForceMoment(tforce_bip.data(), coord_bip.data(),
                                      tmpForceMoment.data(), tmpNodePos.data());
                //Split the force and moment into the two surrounding nodes in a
                //variationally consistent manner using the interpolation factor
                splitForceMoment(tmpForceMoment.data(), loadMapInterp_bip,
                                 &(brFSIdata_.twr_ld[(loadMap_bip)*6]),
                                 &(brFSIdata_.twr_ld[(loadMap_bip+1)*6]));
            }
        }
    }
        
    // Now the blades
    int iStart = 0;
    for (int iBlade=0; iBlade < nBlades; iBlade++) {
        int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
        stk::mesh::Selector sel(meta_.locally_owned_part() & stk::mesh::selectUnion(bladeBndyParts_[iBlade]) );
        const auto& bkts =  bulk_.get_buckets( meta_.side_rank(), sel );
        for (auto b: bkts) {
            // face master element
            MasterElement *meFC =
                MasterElementRepo::get_surface_master_element(b->topology());
            const int nodesPerFace = meFC->nodesPerElement_;
            const int numScsBip = meFC->num_integration_points();
            
            // mapping from ip to nodes for this ordinal;
            // face perspective (use with face_node_relations)
            const int *faceIpNodeMap = meFC->ipNodeMap();
            
            ws_face_shape_function.resize(numScsBip*nodesPerFace);
            meFC->shape_fcn(ws_face_shape_function.data());
            
            ws_coordinates.resize(ndim*nodesPerFace);
            
            for (size_t in=0; in < b->size(); in++) {
                // get face
                stk::mesh::Entity face = (*b)[in];
                // face node relations
                stk::mesh::Entity const * face_node_rels = bulk_.begin_nodes(face);
                
                for (auto i=0; i < ndim; i++)
                    face_center[i] = 0.0;
                // gather nodal data off of face
                for ( int ni = 0; ni < nodesPerFace; ++ni ) {
                    stk::mesh::Entity node = face_node_rels[ni];
                    // gather coordinates
                    const double* xyz = stk::mesh::field_data(*modelCoords, node);
                    const double* xyz_disp = stk::mesh::field_data(*meshDisp, node);
                    for (auto i=0; i < ndim; i++) {
                        ws_coordinates[ni*ndim+i] = xyz[i] + xyz_disp[i];
                        face_center[i] += xyz[i];
                    }                    
                }
                for (auto i=0; i < ndim; i++)
                    face_center[i] /= nodesPerFace;

                // Get reference to load map and loadMapInterp at all ips on this face
                const int* loadMapFace = stk::mesh::field_data(*loadMap_, face);
                const double* loadMapInterpFace = stk::mesh::field_data(*loadMapInterp_, face);
                const double* pforce = stk::mesh::field_data(*pressureForceSCS_, face);
                const double* vforce = stk::mesh::field_data(*tauWallSCS_, face);
                
                for ( int ip = 0; ip < numScsBip; ++ip ) {
                    
                    //Get coordinates and pressure force at this ip
                    for (auto i=0; i < ndim; i++)
                        coord_bip[i] = 0.0;
                    for (int ni = 0; ni < nodesPerFace; ni++) {
                        const double r = ws_face_shape_function[ip*nodesPerFace + ni];
                        for (int i=0; i < ndim; i++)
                            coord_bip[i] += r * ws_coordinates[ni*ndim+i];
                    }

                    int loadMap_bip = loadMapFace[ip];
                    //Radial location of scs center projected onto blade beam mesh
                    double interpFac = loadMapInterpFace[ip];
                    double r_n = brFSIdata_.bld_rloc[iStart + loadMap_bip];
                    double r_np1 = brFSIdata_.bld_rloc[iStart + loadMap_bip + 1];
                    double rloc_proj = r_n + interpFac * (r_np1 - r_n);

                    if (interpFac < 0.0) {
                        std::cerr << "rloc_proj = " << rloc_proj
                                  << ", r_n = " << r_n
                                  << ", r_np1 = " << r_np1
                                  << ", interpFac = " << interpFac
                                  << ", loadMap_bip = " << loadMap_bip
                                  << std::endl ;
                    }
                
                    //Find the interpolated reference position first
                    linInterpVec(&brFSIdata_.bld_ref_pos[(loadMap_bip + iStart)*6],
                                 &brFSIdata_.bld_ref_pos[(loadMap_bip + iStart + 1)*6],
                                 interpFac, tmpNodePos.data());
                        
                    //Find the interpolated linear displacement 
                    linInterpVec(&brFSIdata_.bld_def[(loadMap_bip + iStart)*6],
                                 &brFSIdata_.bld_def[(loadMap_bip + iStart + 1)*6],
                                 interpFac, tmpNodeDisp.data());
                        
                    //Add displacement to find actual position
                    for (auto idim=0; idim < 3; idim++)
                        tmpNodePos[idim] += tmpNodeDisp[idim];

                    for (auto idim=0; idim < 3; idim++)
                        tforce_bip[idim] = pforce[ip*3+idim] + vforce[ip*3+idim];
                    
                    // Temporarily store total force and moment as (fX, fY, fZ,
                    // mX, mY, mZ)
                    std::vector<double> tmpForceMoment(6,0.0); 
                    //Now compute the force and moment on the interpolated
                    //reference position
                    computeEffForceMoment(tforce_bip.data(), coord_bip.data(),
                                          tmpForceMoment.data(), tmpNodePos.data());
                    
                    // Calculate suport length for this scs along local blade direction
                    const int nfNode = faceIpNodeMap[ip];
                    stk::mesh::Entity nnode = face_node_rels[nfNode];
                    const double* xyz = stk::mesh::field_data(*modelCoords, nnode);
                    double sl = 0.0;
                    for(auto i=0; i < ndim; i++) {
                        sl +=
                            (xyz[i] - face_center[i]) *
                            (brFSIdata_.bld_ref_pos[(loadMap_bip + iStart + 1)*6+i]
                             - brFSIdata_.bld_ref_pos[(loadMap_bip + iStart)*6+i]);
                    }
                    sl = std::abs(sl / (r_np1 - r_n));
                    double sl_ratio = sl/(r_np1 - r_n);

                    if ( (loadMap_bip < 1) || (loadMap_bip > (nPtsBlade-3)) ) {
                        //if (true) {
                        //Now split the force and moment on the interpolated
                        //reference position into the 'left' and 'right' nodes
                        splitForceMoment(tmpForceMoment.data(), interpFac,
                                         &(brFSIdata_.bld_ld[(loadMap_bip + iStart)*6]),
                                         &(brFSIdata_.bld_ld[(loadMap_bip + iStart + 1)*6]) );
                    } else {

                        //Split into 2 and distribute to nearest element
                        for(auto i=0;i < 6;i++)
                             tmpForceMoment[i] *= 0.5;
                        
                        splitForceMoment(tmpForceMoment.data(), interpFac,
                                         &(brFSIdata_.bld_ld[(loadMap_bip + iStart)*6]),
                                         &(brFSIdata_.bld_ld[(loadMap_bip + iStart + 1)*6]) );

                        for(auto i=0;i < 6;i++)
                            tmpForceMoment[i] *= 0.5;

                        if ((interpFac - 0.5*sl_ratio) < 0.0) {
                            double r_nm1 = brFSIdata_.bld_rloc[iStart + loadMap_bip - 1];
                            double l_interpFac = (rloc_proj - 0.5*sl - r_nm1)/(r_n - r_nm1);
                            splitForceMoment(tmpForceMoment.data(), l_interpFac,
                                             &(brFSIdata_.bld_ld[(loadMap_bip - 1 + iStart)*6]),
                                             &(brFSIdata_.bld_ld[(loadMap_bip + iStart)*6]) );
                            
                        } else {
                            double l_interpFac = interpFac - 0.5*sl_ratio;
                            splitForceMoment(tmpForceMoment.data(), l_interpFac,
                                             &(brFSIdata_.bld_ld[(loadMap_bip + iStart)*6]),
                                             &(brFSIdata_.bld_ld[(loadMap_bip + iStart + 1)*6]) );
                        }

                        if ((interpFac + 0.5*sl_ratio) > 1.0) {
                            double r_np2 = brFSIdata_.bld_rloc[iStart + loadMap_bip + 2];
                            double r_interpFac = (rloc_proj + 0.5*sl - r_np1)/(r_np2 - r_np1);
                            splitForceMoment(tmpForceMoment.data(), r_interpFac,
                                             &(brFSIdata_.bld_ld[(loadMap_bip + iStart + 1)*6]),
                                             &(brFSIdata_.bld_ld[(loadMap_bip + iStart + 2)*6]) );
                        } else {
                            double r_interpFac = interpFac + 0.5*sl_ratio;
                            splitForceMoment(tmpForceMoment.data(), r_interpFac,
                                             &(brFSIdata_.bld_ld[(loadMap_bip + iStart)*6]),
                                             &(brFSIdata_.bld_ld[(loadMap_bip + iStart + 1)*6]) );
                        }
                        
                        //Bin directly to nodes
                        // double totfrac = 0.0;
                        // for (auto iNode = loadMap_bip-1 ; iNode < loadMap_bip+2; iNode++) {
                        //     double x_lower = std::max(bld_rmm_[iStart+iNode][0],rloc_proj-0.5*sl);
                        //     double x_higher = std::min(bld_rmm_[iStart+iNode][1],rloc_proj+0.5*sl);
                        //     if (x_higher > x_lower) {
                        //         double locfrac = (x_higher - x_lower)/sl;
                        //         totfrac += locfrac;
                        //         for (auto i=0; i < 6; i++)
                        //             brFSIdata_.bld_ld[(iStart+iNode)*6+i] +=
                        //                 tmpForceMoment[i]*locfrac;
                        //     }
                            
                        // }

                        //Split to elements and distribute
                        // std::array<double,6> locTmpForceMoment;
                        // double totfrac = 0.0;
                        // for (auto iNode = loadMap_bip-1 ; iNode < loadMap_bip+2; iNode++) {
                        //     double x_lower = std::max(brFSIdata_.bld_rloc[iStart+iNode],rloc_proj-0.5*sl);
                        //     double x_higher = std::min(brFSIdata_.bld_rloc[iStart+iNode+1],rloc_proj+0.5*sl);
                        //     if (x_higher > x_lower) {
                        //         double locfrac = (x_higher - x_lower)/sl;
                        //         totfrac += locfrac;
                        //         double locInterpFac = (0.5*(x_higher + x_lower) - brFSIdata_.bld_rloc[iStart+iNode])
                        //             /(brFSIdata_.bld_rloc[iStart+iNode+1]-brFSIdata_.bld_rloc[iStart+iNode]);

                        //         for(auto i=0; i<6; i++)
                        //             locTmpForceMoment[i] = locfrac*tmpForceMoment[i];
                                
                        //         splitForceMoment(locTmpForceMoment.data(), locInterpFac,
                        //                          &(brFSIdata_.bld_ld[(iNode + iStart)*6]),
                        //                          &(brFSIdata_.bld_ld[(iNode + iStart + 1)*6]) );
                               
                        //     }
                        // }
                        // if (totfrac < 0.999) {
                        //     std::cerr << "iNode = " << loadMap_bip << ", rloc_proj = " << rloc_proj << ", sl = " << sl << std::endl ;
                        // }

                    }
                }
            }
        }

        // //Compute the total force and moment at the hub from this blade
        // std::vector<double> hubForceMoment(6,0.0);
        // computeHubForceMomentForPart(hubForceMoment, brFSIdata_.hub_ref_pos,
        //                              bladeBndyParts_[iBlade]);
        
        // //Now compute total force and moment at the hub from the loads mapped to the
        // std::vector<double> l_hubForceMomentMapped(6,0.0);
        // std::array<double,3> tmpNodePos {0.0,0.0,0.0}; // Vector to temporarily store a position vector
        // for (size_t i=0 ; i < nPtsBlade; i++) {
        //     for (int idim = 0; idim < 3; idim++)
        //         tmpNodePos[idim] = brFSIdata_.bld_ref_pos[(i+iStart)*6 + idim]
        //             + brFSIdata_.bld_def[(i+iStart)*6 + idim];
        //     computeEffForceMoment(
        //         &(brFSIdata_.bld_ld[(i+iStart)*6]), tmpNodePos.data(),
        //         l_hubForceMomentMapped.data(), brFSIdata_.hub_ref_pos.data() );
        //     for(size_t j=0; j < ndim; j++) // Add the moment manually
        //         l_hubForceMomentMapped[3+j]
        //             += brFSIdata_.bld_ld[(i+iStart)*6+3+j];
        // }
        // std::vector<double> hubForceMomentMapped(6,0.0);
        // stk::all_reduce_sum(bulk_.parallel(),
        //                     l_hubForceMomentMapped.data(),
        //                     hubForceMomentMapped.data(), 6);
        
        
        // if (bulk_.parallel_rank() == turbineProc_) {
            
        //     std::cout << "Total force moment on the hub due to blade " << iBlade << std::endl;
        //     std::cout << "Force = (";
        //     for(size_t j=0; j < ndim; j++)
        //         std::cout << hubForceMoment[j] << ", ";
        //     std::cout << ") Moment = (" ;
        //     for(size_t j=0; j < ndim; j++)
        //         std::cout << hubForceMoment[3+j] << ", ";
        //     std::cout << ")" << std::endl;
        //     std::cout << "Total force moment on the hub from mapped load due to blade " << iBlade << std::endl;
        //     std::cout << "Force = (";
        //     for(size_t j=0; j < ndim; j++)
        //         std::cout << hubForceMomentMapped[j] << ", ";
        //     std::cout << ") Moment = (" ;
        //     for(size_t j=0; j < ndim; j++)
        //         std::cout << hubForceMomentMapped[3+j] << ", ";
        //     std::cout << ")" << std::endl;
        // }
        
        iStart += nPtsBlade;
    }

}

//! Split a force and moment into the surrounding 'left' and 'right' nodes in a variationally consistent manner using
void fsiTurbine::splitForceMoment(double *totForceMoment, double interpFac, double *leftForceMoment, double *rightForceMoment) {
    for(size_t i=0; i<6; i++) {
        leftForceMoment[i] += (1.0 - interpFac) * totForceMoment[i];
        rightForceMoment[i] += interpFac * totForceMoment[i];
    }
}

void fsiTurbine::computeHubForceMomentForPart(std::vector<double> & hubForceMoment, std::vector<double> & hubPos, stk::mesh::PartVector partVec) {

    VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
    VectorFieldType* meshDisp = meta_.get_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_displacement");
    VectorFieldType* pressureForce = meta_.get_field<VectorFieldType>(stk::topology::NODE_RANK, "pressure_force");
    VectorFieldType* tauWall = meta_.get_field<VectorFieldType>(stk::topology::NODE_RANK, "tau_wall");
    std::vector<double> l_hubForceMoment(6,0.0);
    std::array<double,3> tmpMeshPos {0.0,0.0,0.0}; //Vector to temporarily store mesh node location
    
    stk::mesh::Selector sel(meta_.locally_owned_part() & stk::mesh::selectUnion(partVec));
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);
    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* xyz = stk::mesh::field_data(*modelCoords, node);
            double* xyz_disp = stk::mesh::field_data(*meshDisp, node);
            double * pressureForceNode = stk::mesh::field_data(*pressureForce, node);
            double * viscForceNode = stk::mesh::field_data(*tauWall, node);
            std::array<double,3> fsiForceNode;
            for (int i=0; i < 3; i++) {
                fsiForceNode[i] = pressureForceNode[i] + viscForceNode[i];
                tmpMeshPos[i] = xyz[i]  + xyz_disp[i];
            }
            computeEffForceMoment(fsiForceNode.data(), tmpMeshPos.data(), l_hubForceMoment.data(), hubPos.data());
        }
    }

    stk::all_reduce_sum(bulk_.parallel(), l_hubForceMoment.data(), hubForceMoment.data(), 6);

}

//! Compute the effective force and moment at the OpenFAST mesh node for a given force at the CFD surface mesh node
void fsiTurbine::computeEffForceMoment(double *forceCFD, double *xyzCFD, double *forceMomOF, double *xyzOF) {

    const int ndim=3; //I don't see this ever being used in other situations
    for(size_t j=0; j < ndim; j++)
        forceMomOF[j] += forceCFD[j];
    forceMomOF[3] += (xyzCFD[1]-xyzOF[1])*forceCFD[2] - (xyzCFD[2]-xyzOF[2])*forceCFD[1] ;
    forceMomOF[4] += (xyzCFD[2]-xyzOF[2])*forceCFD[0] - (xyzCFD[0]-xyzOF[0])*forceCFD[2] ;
    forceMomOF[5] += (xyzCFD[0]-xyzOF[0])*forceCFD[1] - (xyzCFD[1]-xyzOF[1])*forceCFD[0] ;

}

//! Set displacement corresponding to rotation at a constant rpm on the OpenFAST mesh before mapping to the turbine blade surface mesh
void fsiTurbine::setRotationDisplacement(std::array<double,3> axis, double omega, double curTime) {

    const int iproc = bulk_.parallel_rank();
    double theta=omega*curTime;
    double twopi = 2.0 * M_PI;
    theta = std::fmod(theta, twopi);
    if ( iproc == turbineProc_) 
        std::cerr << "Setting rotation of " << theta * 180.0/M_PI << " degrees about ["  << axis[0] << "," << axis[1] << "," << axis[2] << "]"  << std::endl;

    //Rotate the hub first
    double hubRot = 4.0*tan(0.25*theta);
    std::vector<double> wmHubRot = {hubRot * axis[0], hubRot * axis[1], hubRot * axis[2]};
    for (size_t i=0; i<3; i++)
        brFSIdata_.hub_def[3+i] = -wmHubRot[i];

    //For each node on the openfast blade1 mesh - compute distance from the blade root node. Apply a rotation varying as the square of the distance between 0 - 45 degrees about the [0 1 0] axis. Apply a translation displacement that produces a tip displacement of 5m
    int iStart = 0;
    int nBlades = params_.numBlades;;
    for (size_t iBlade=0; iBlade < nBlades; iBlade++) {
        double bladeRot = 4.0*tan(0.25 * iBlade * 120.0 * M_PI / 180.0);
        std::vector<double> wmRotBlade_ref = {bladeRot * axis[0], bladeRot * axis[1], bladeRot * axis[2]};
        std::vector<double> wmRotBlade(3,0.0);
        composeWM(wmHubRot.data(), wmRotBlade_ref.data(), wmRotBlade.data());

        int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
        for (size_t i=0; i < nPtsBlade; i++) {
            
            //Transpose the whole thing
            for(size_t j=0; j < 3; j++)
                brFSIdata_.bld_def[(iStart+i)*6+3+j] = -wmRotBlade[j];

            //Set translational displacement
            std::vector<double> r(3,0.0);
            for(size_t j=0; j < 3; j++)
                r[j] = brFSIdata_.bld_ref_pos[(iStart+i)*6+j] - brFSIdata_.hub_ref_pos[j];

            std::vector<double> rRot(3,0.0);

            applyWMrotation(wmHubRot.data(), r.data(), rRot.data());
            brFSIdata_.bld_def[(iStart+i)*6+0] = rRot[0] - r[0];
            brFSIdata_.bld_def[(iStart+i)*6+1] = rRot[1] - r[1];
            brFSIdata_.bld_def[(iStart+i)*6+2] = rRot[2] - r[2];

        }
        iStart += nPtsBlade;
    }
}
    

//! Set sample displacement on the OpenFAST mesh before mapping to the turbine blade surface mesh
void fsiTurbine::setSampleDisplacement(double curTime) {



    //Turbine rotates at 12.1 rpm
    double omega=(12.1/60.0)*2.0*M_PI; //12.1 rpm
    double theta=omega*curTime;

    double sinOmegaT = std::sin(omega*curTime);

    //Rotate the hub first
    double hubRot = 4.0*tan(0.25*theta);
    std::vector<double> wmHubRot = {hubRot, 0.0, 0.0};
    for (size_t i=0; i<3; i++)
        brFSIdata_.hub_def[3+i] = -wmHubRot[i];

    //For each node on the openfast blade1 mesh - compute distance from the blade root node. Apply a rotation varying as the square of the distance between 0 - 45 degrees about the [0 1 0] axis. Apply a translation displacement that produces a tip displacement of 5m
    int iStart = 0;
    int nBlades = params_.numBlades;;
    for (size_t iBlade=0; iBlade < nBlades; iBlade++) {
        std::vector<double> wmRotBlade_ref = {4.0*tan(0.25 * iBlade * 120.0 * M_PI / 180.0), 0.0, 0.0};
        std::vector<double> wmRotBlade(3,0.0);
        composeWM(wmHubRot.data(), wmRotBlade_ref.data(), wmRotBlade.data());

        int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
        for (size_t i=0; i < nPtsBlade; i++) {

            double rDistSq = calcDistanceSquared(&(brFSIdata_.bld_ref_pos[(iStart+i)*6]), &(brFSIdata_.bld_ref_pos[(iStart)*6]) )/10000.0;
            double sinRdistSq = std::sin(rDistSq);
            double tanRdistSq = std::tan(rDistSq);
            //Set rotational displacement
            std::vector<double> wmRot1 = {1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};
            std::vector<double> wmRot(3,0.0);
            applyWMrotation(wmRotBlade.data(), wmRot1.data(), wmRot.data());
            double rot = 4.0*tan(0.25 * (45.0 * M_PI / 180.0) * sinRdistSq  * sinOmegaT ); // 4.0 * tan(phi/4.0) parameter for Wiener-Milenkovic
            for(size_t j= 0; j < 3; j++) {
                wmRot[j] *= rot;
            }

            std::vector<double> finalRot(3,0.0);
            composeWM(wmRot.data(), wmRotBlade.data(), finalRot.data()); //Compose with hub orientation to account for rotation of turbine

            std::vector<double> origZaxis = {0.0, 0.0, 1.0};
            std::vector<double> rotZaxis(3,0.0);
            applyWMrotation(finalRot.data(), origZaxis.data(), rotZaxis.data());

            //Finally transpose the whole thing
            for(size_t j=0; j < 3; j++)
                brFSIdata_.bld_def[(iStart+i)*6+3+j] = -finalRot[j];


            //Set translational displacement
            double xDisp = sinRdistSq * 15.0 * sinOmegaT;

            std::vector<double> r(3,0.0);
            for(size_t j=0; j < 3; j++)
                r[j] = brFSIdata_.bld_ref_pos[(iStart+i)*6+j] - brFSIdata_.hub_ref_pos[j];

            std::vector<double> rRot(3,0.0);

            std::vector<double> transDisp = {xDisp, xDisp, xDisp};
            std::vector<double> transDispRot(3,0.0);

            applyWMrotation(wmRotBlade.data(), transDisp.data(), transDispRot.data());

            applyWMrotation(wmHubRot.data(), r.data(), rRot.data());
            brFSIdata_.bld_def[(iStart+i)*6+0] = transDispRot[0] + rRot[0] - r[0];
            brFSIdata_.bld_def[(iStart+i)*6+1] = transDispRot[1] + rRot[1] - r[1];
            brFSIdata_.bld_def[(iStart+i)*6+2] = transDispRot[2] + rRot[2] - r[2];

            for (size_t j=0; j < 3; j++) {
                brFSIdata_.bld_vel[(iStart+i)*6+j] = tanRdistSq * 3.743; // Completely arbitrary values
                brFSIdata_.bld_vel[(iStart+i)*6+3+j] = sinRdistSq * 6.232; // Completely arbitrary values
            }

        }
        iStart += nPtsBlade;
    }
}


//! Set reference displacement on the turbine blade surface mesh, for comparison with Sample displacement set in setSampleDisplacement
void fsiTurbine::setRefDisplacement(double curTime) {

  //Turbine rotates at 12.1 rpm
  double omega=(12.1/60.0)*2.0*M_PI; //12.1 rpm
  double theta=omega*curTime;

  //Rotate the hub first
  std::vector<double> hubPos = {0, 0, 130.0};
  std::vector<double> wmHubRot = {4.0*tan(0.25*theta), 0.0, 0.0};

  double sinOmegaT = std::sin(omega*curTime);

  // extract the vector field type set by this function
  const int ndim = meta_.spatial_dimension();
  VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
  stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* refDisp = meta_.get_field<VectorFieldType>(
  stk::topology::NODE_RANK, "mesh_displacement_ref");
  VectorFieldType* refVel = meta_.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "mesh_velocity_ref");

  std::vector<double> zAxis0 = {0.0, 0.0, 1.0};

  for (size_t iBlade=0; iBlade < 3; iBlade++) {

      std::vector<double> wmRotBlade_ref = {4.0*tan(0.25 * iBlade * 120.0 * M_PI / 180.0), 0.0, 0.0};
      std::vector<double> wmRotBlade(3,0.0);
      composeWM(wmHubRot.data(), wmRotBlade_ref.data(), wmRotBlade.data());

      std::vector<double> nHatRef(3,0.0);
      std::vector<double> nHat(3,0.0);

      applyWMrotation(wmRotBlade_ref.data(), zAxis0.data(), nHatRef.data());
      applyWMrotation(wmRotBlade.data(), zAxis0.data(), nHat.data());

      stk::mesh::Selector sel(stk::mesh::selectUnion(bladeParts_[iBlade])); // extract blade
      const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel); // extract buckets for the blade

      for (auto b: bkts)  {// loop over number of buckets
          for (size_t in=0; in < b->size(); in++) { // loop over all nodes in the bucket
              auto node = (*b)[in];
              double* xyz = stk::mesh::field_data(*modelCoords, node);
              double* vecRefNode = stk::mesh::field_data(*refDisp, node);
              double* velRefNode = stk::mesh::field_data(*refVel, node);

              std::vector<double> xyzMhub(3,0.0);
              for(size_t j=0; j < 3; j++)
                  xyzMhub[j] = xyz[j] - hubPos[j];

              //Translational displacement due to turbine rotation
              std::vector<double> xyzMhubRot(3,0.0);
              applyWMrotation(wmHubRot.data(),xyzMhub.data(), xyzMhubRot.data());

              // Compute position of current node relative to blade root
              double rDistSq = (dot(xyzMhub.data(), nHatRef.data()) - 1.5)*(dot(xyzMhub.data(), nHatRef.data()) - 1.5)/10000.0;
              double sinRdistSq = std::sin(rDistSq);
              double tanRdistSq = std::tan(rDistSq);

              //Set translational displacement due to deflection
              double xDisp = sinRdistSq * 15.0 * sinOmegaT;
              std::vector<double> transDisp = {xDisp, xDisp, xDisp};
              std::vector<double> transDispRot(3,0.0);
              applyWMrotation(wmRotBlade.data(), transDisp.data(), transDispRot.data());


              //Translational displacement due to rotational deflection
              double xyzMhubDotNHatRef = dot(xyzMhub.data(), nHatRef.data());
              std::vector<double> pGlobRef(3,0.0);
              std::vector<double> pLoc(3,0.0);
              for(size_t j=0; j < 3; j++)
                  pGlobRef[j] = xyzMhub[j] - xyzMhubDotNHatRef * nHatRef[j];
              applyWMrotation(wmRotBlade_ref.data(),pGlobRef.data(),pLoc.data(),-1.0);
              std::vector<double> pGlob(3,0.0);
              applyWMrotation(wmRotBlade.data(),pLoc.data(),pGlob.data());

              double rot = 4.0*tan(0.25 * (45.0 * M_PI / 180.0) * sinRdistSq * sinOmegaT); // 4.0 * tan(phi/4.0) parameter for Wiener-Milenkovic
              std::vector<double> wmRot1 = {1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};
              std::vector<double> wmRot(3,0.0);
              applyWMrotation(wmRotBlade.data(), wmRot1.data(), wmRot.data());
              for(size_t j= 0; j < 3; j++)
                  wmRot[j] *= rot;

              std::vector<double> r_rot(3,0.0);
              applyWMrotation(wmRot.data(), pGlob.data(), r_rot.data());

              for(size_t j=0; j < ndim; j++ )
                  vecRefNode[j] = xyzMhubRot[j] - xyzMhub[j] + transDispRot[j] + r_rot[j] - pGlob[j];

              std::vector<double> omega = {sinRdistSq * 6.232, sinRdistSq * 6.232, sinRdistSq * 6.232};
              std::vector<double> omegaCrossRrot(3,0.0);
              cross(omega.data(), r_rot.data(), omegaCrossRrot.data());
              for(size_t j=0; j < ndim; j++ )
                  velRefNode[j] = tanRdistSq * 3.743 +  omegaCrossRrot[j];

          }
      }
  }
}

//! Calculate the distance between 3-dimensional vectors 'a' and 'b'
double fsiTurbine::calcDistanceSquared(double * a, double * b) {

    double dist = 0.0;
    for(size_t i=0; i < 3; i++)
        dist += (a[i]-b[i])*(a[i]-b[i]);
    return dist;

}

//! Map the deflections from the openfast nodes to the turbine surface CFD mesh. Will call 'computeDisplacement' for each node on the turbine surface CFD mesh.
void fsiTurbine::mapDisplacements() {

   //To implement this function - assume that 'dispMap_' field contains the lower node id of the corresponding element along the blade or the tower along with the 'bldDispMapInterp_' field that contains the non-dimensional location of the CFD surface mesh node on that element.

    // For e.g., for blade 'k' if the lower node id from 'dispMap_' is 'j' and the non-dimenional location from 'dispMapInterp_' is 'm', then the translational displacement for the CFD surface mesh is
    // (1-m) * bld_def[k][j*6+0] + m * bld_def[k][(j+1)*6+0]
    // (1-m) * bld_def[k][j*6+1] + m * bld_def[k][(j+1)*6+1]
    // (1-m) * bld_def[k][j*6+2] + m * bld_def[k][(j+1)*6+2]

    //TODO: When the turbine is rotating, displacement of the surface of the blades and hub (not the nacelle and tower) should be split into a rigid body motion due to the rotation of the turbine, yawing of the turbine and a deflection of the structure itself. The yaw rate and rotation rate will vary over time. OpenFAST always stores the displacements of all nodes with respect to the reference configuration. When using a sliding mesh interface (yaw = 0), the mesh blocks inside the rotating part of the sliding interface should be moved with rigid body motion corresponding to the rotation rate first and then a second mesh deformation procedure should be performed to apply the remaining structural deflection. Figure out how to do this.

    VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");
    VectorFieldType* displacement = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_displacement");
    VectorFieldType* refDisplacement = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_displacement_ref");
    VectorFieldType* meshVelocity = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_velocity");
    VectorFieldType* refVelocity = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_velocity_ref");

    std::vector<double> totDispNode(6,0.0); // Total displacement at any node in (transX, transY, transZ, rotX, rotY, rotZ)
    std::vector<double> totVelNode(6,0.0); // Total velocity at any node in (transX, transY, transZ, rotX, rotY, rotZ)
    std::vector<double> tmpNodePos(6,0.0); // Vector to temporarily store a position and orientation vector

    //Do the tower first
    stk::mesh::Selector sel(stk::mesh::selectUnion(twrParts_));
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* oldxyz = stk::mesh::field_data(*modelCoords, node);
            double *dx = stk::mesh::field_data(*displacement, node);
            int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
            double* dispMapInterpNode = stk::mesh::field_data(*dispMapInterp_, node);
            double *mVel = stk::mesh::field_data(*meshVelocity, node);

            //Find the interpolated reference position first
            linInterpTotDisplacement(&brFSIdata_.twr_ref_pos[(*dispMapNode)*6], &brFSIdata_.twr_ref_pos[(*dispMapNode + 1)*6], *dispMapInterpNode, tmpNodePos.data());

            //Now linearly interpolate the deflections to the intermediate location
            linInterpTotDisplacement(&brFSIdata_.twr_def[(*dispMapNode)*6], &brFSIdata_.twr_def[(*dispMapNode + 1)*6], *dispMapInterpNode, totDispNode.data());

            //Now transfer the interpolated displacement to the CFD mesh node
            computeDisplacement(totDispNode.data(), tmpNodePos.data(), dx, oldxyz);

            //Now linearly interpolate the velocity to the intermediate location
            linInterpTotVelocity(&brFSIdata_.twr_vel[(*dispMapNode)*6], &brFSIdata_.twr_vel[(*dispMapNode + 1)*6], *dispMapInterpNode, totVelNode.data());

            //Now transfer the interpolated translational and rotational velocity to an equivalent translational velocity on the CFD mesh node
            computeMeshVelocity(totVelNode.data(), totDispNode.data(), tmpNodePos.data(), mVel, oldxyz);

        }
    }


    // Now the blades
    int nBlades = params_.numBlades;
    int iStart = 0;
    for (int iBlade=0; iBlade < nBlades; iBlade++) {
        int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
        stk::mesh::Selector sel(stk::mesh::selectUnion(bladeParts_[iBlade]));
        const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

        for (auto b: bkts) {
            for (size_t in=0; in < b->size(); in++) {
                auto node = (*b)[in];
                double* oldxyz = stk::mesh::field_data(*modelCoords, node);
                double *dx = stk::mesh::field_data(*displacement, node);
                int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
                double* dispMapInterpNode = stk::mesh::field_data(*dispMapInterp_, node);
                double *mVel = stk::mesh::field_data(*meshVelocity, node);

                //Find the interpolated reference position first
                linInterpTotDisplacement(&brFSIdata_.bld_ref_pos[(*dispMapNode + iStart)*6], &brFSIdata_.bld_ref_pos[(*dispMapNode + iStart + 1)*6], *dispMapInterpNode, tmpNodePos.data());

                //Now linearly interpolate the deflections to the intermediate location
                linInterpTotDisplacement(&brFSIdata_.bld_def[(*dispMapNode + iStart)*6], &brFSIdata_.bld_def[(*dispMapNode + iStart + 1)*6], *dispMapInterpNode, totDispNode.data());

                //Now transfer the interpolated displacement to the CFD mesh node
                computeDisplacement(totDispNode.data(), tmpNodePos.data(), dx, oldxyz);

                //Now linearly interpolate the velocity to the intermediate location
                linInterpTotVelocity(&brFSIdata_.bld_vel[(*dispMapNode + iStart)*6], &brFSIdata_.bld_vel[(*dispMapNode + iStart + 1)*6], *dispMapInterpNode, totVelNode.data());

                //Now transfer the interpolated translational and rotational velocity to an equivalent translational velocity on the CFD mesh node
                computeMeshVelocity(totVelNode.data(), totDispNode.data(), tmpNodePos.data(), mVel, oldxyz);

            }
        }


        // std::vector<double> errorNorm(3,0.0);
        // compute_error_norm(displacement, refDisplacement, bladeParts_[iBlade], errorNorm);

        // if (!bulk_.parallel_rank()) {
        //     std::cout << "Error in displacement for blade " << iBlade << " = " << errorNorm[0] << " " << errorNorm[1] << " " << errorNorm[2] << std::endl ;
        // }

        // compute_error_norm(meshVelocity, refVelocity, bladeParts_[iBlade], errorNorm);

        // if (!bulk_.parallel_rank()) {
        //     std::cout << "Error in velocity for blade " << iBlade << " = " << errorNorm[0] << " " << errorNorm[1] << " " << errorNorm[2] << std::endl ;
        // }

        iStart += nPtsBlade;
    }


    //Now the hub
    stk::mesh::Selector hubsel(stk::mesh::selectUnion(hubParts_));
    const auto& hubbkts = bulk_.get_buckets(stk::topology::NODE_RANK, hubsel);
    for (auto b: hubbkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* oldxyz = stk::mesh::field_data(*modelCoords, node);
            double *dx = stk::mesh::field_data(*displacement, node);
            double *mVel = stk::mesh::field_data(*meshVelocity, node);

            //Now transfer the displacement to the CFD mesh node
            computeDisplacement(brFSIdata_.hub_def.data(), brFSIdata_.hub_ref_pos.data(), dx, oldxyz);

            //Now transfer the translational and rotational velocity to an equivalent translational velocity on the CFD mesh node
            computeMeshVelocity(brFSIdata_.hub_vel.data(), brFSIdata_.hub_def.data(), brFSIdata_.hub_ref_pos.data(), mVel, oldxyz);

        }
    }

    //Now the nacelle
    stk::mesh::Selector nacsel(stk::mesh::selectUnion(nacelleParts_));
    const auto& nacbkts = bulk_.get_buckets(stk::topology::NODE_RANK, nacsel);
    for (auto b: nacbkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* oldxyz = stk::mesh::field_data(*modelCoords, node);
            double *dx = stk::mesh::field_data(*displacement, node);
            double *mVel = stk::mesh::field_data(*meshVelocity, node);

            //Now transfer the displacement to the CFD mesh node
            computeDisplacement(brFSIdata_.nac_def.data(), brFSIdata_.nac_ref_pos.data(), dx, oldxyz);

            //Now transfer the translational and rotational velocity to an equivalent translational velocity on the CFD mesh node
            computeMeshVelocity(brFSIdata_.nac_vel.data(), brFSIdata_.nac_def.data(), brFSIdata_.nac_ref_pos.data(), mVel, oldxyz);

        }
    }

}

//! Linearly interpolate dispInterp = dispStart + interpFac * (dispEnd - dispStart). Special considerations for Wiener-Milenkovic parameters
void fsiTurbine::linInterpTotDisplacement(double *dispStart, double *dispEnd, double interpFac, double * dispInterp) {

    // Handle the translational displacement first
    linInterpVec(dispStart, dispEnd, interpFac, dispInterp);
    // Now deal with the rotational displacement
    linInterpRotation( &dispStart[3], &dispEnd[3], interpFac, &dispInterp[3]);

}

//! Linearly interpolate velInterp = velStart + interpFac * (velEnd - velStart).
void fsiTurbine::linInterpTotVelocity(double *velStart, double *velEnd, double interpFac, double * velInterp) {

    // Handle the translational velocity first
    linInterpVec(velStart, velEnd, interpFac, velInterp);
    // Now deal with the rotational velocity
    linInterpVec(&velStart[3], &velEnd[3], interpFac, &velInterp[3]);

}

//! Linearly interpolate between 3-dimensional vectors 'a' and 'b' with interpolating factor 'interpFac'
void fsiTurbine::linInterpVec(double * a, double * b, double interpFac, double * aInterpb) {

    for (size_t i=0; i < 3; i++)
        aInterpb[i] = a[i] + interpFac * (b[i] - a[i]);

}

/* Linearly interpolate the Wiener-Milenkovic parameters between 'qStart' and 'qEnd' into 'qInterp' with an interpolating factor 'interpFac'
    see O.A.Bauchau, 2011, Flexible Multibody Dynamics p. 649, section 17.2, Algorithm 1'
*/
void fsiTurbine::linInterpRotation(double * qStart, double * qEnd, double interpFac, double * qInterp) {

    std::vector<double> intermedQ(3,0.0);
    composeWM(qStart, qEnd, intermedQ.data(), -1.0); //Remove rigid body rotation of qStart
    for(size_t i=0; i < 3; i++)
        intermedQ[i] = interpFac * intermedQ[i]; // Do the interpolation
    composeWM(qStart, intermedQ.data(), qInterp); // Add rigid body rotation of qStart back

}

//! Compose Wiener-Milenkovic parameters 'p' and 'q' into 'pPlusq'. If a transpose of 'p' is required, set tranposeP to '-1', else leave blank or set to '+1'
void fsiTurbine::composeWM(double * p, double * q, double * pPlusq, double transposeP, double transposeQ) {

    double p0 = 2.0 - 0.125*dot(p,p);
    double q0 = 2.0 - 0.125*dot(q,q);
    std::vector<double> pCrossq(3,0.0);
    cross(p, q, pCrossq.data());

    double delta1 = (4.0-p0)*(4.0-q0);
    double delta2 = p0*q0 - transposeP*dot(p,q);
    double premultFac = 0.0;
    if (delta2 < 0)
        premultFac = -4.0/(delta1 - delta2);
    else
        premultFac = 4.0/(delta1 + delta2);

    for (size_t i=0; i < 3; i++)
        pPlusq[i] = premultFac * (transposeQ * p0 * q[i] + transposeP * q0 * p[i] + transposeP * transposeQ * pCrossq[i] );

}

double fsiTurbine::dot(double * a, double * b) {

    return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);

}

void fsiTurbine::cross(double * a, double * b, double * aCrossb) {

    aCrossb[0] = a[1]*b[2] - a[2]*b[1];
    aCrossb[1] = a[2]*b[0] - a[0]*b[2];
    aCrossb[2] = a[0]*b[1] - a[1]*b[0];

}

//! Convert one array of 6 deflections (transX, transY, transZ, wmX, wmY, wmZ) into one vector of translational displacement at a given node on the turbine surface CFD mesh.
void fsiTurbine::computeDisplacement(double *totDispNode, double * totPosOF,  double *transDispNode, double * xyzCFD) {

    //Get the relative distance between totPosOF and xyzCFD in the inertial frame
    std::vector<double> p(3,0.0);
    for (size_t i=0; i < 3; i++)
        p[i] = xyzCFD[i] - totPosOF[i];
    //Convert 'p' vector to the local frame of reference
    std::vector<double> pLoc(3,0.0);
    applyWMrotation(&(totPosOF[3]), p.data(), pLoc.data());

    std::vector<double> pRot(3,0.0);
    applyWMrotation(&(totDispNode[3]), pLoc.data(), pRot.data(),-1); // Apply the rotation corresponding to the final orientation to bring back to inertial frame

    for (size_t i=0; i < 3; i++)
        transDispNode[i] = totDispNode[i] + pRot[i] - p[i];

}

//! Convert one array of 6 velocities (transX, transY, transZ, wmX, wmY, wmZ) into one vector of translational velocity at a given node on the turbine surface CFD mesh.
void fsiTurbine::computeMeshVelocity(double *totVelNode, double * totDispNode, double * totPosOF,  double *transVelNode, double * xyzCFD) {

    //Get the relative distance between totPosOF and xyzCFD in the inertial frame
    std::vector<double> p(3,0.0);
    for (size_t i=0; i < 3; i++)
        p[i] = xyzCFD[i] - totPosOF[i];
    //Convert 'p' vector to the local frame of reference
    std::vector<double> pLoc(3,0.0);
    applyWMrotation(&(totPosOF[3]), p.data(), pLoc.data());

    std::vector<double> pRot(3,0.0);
    applyWMrotation(&(totDispNode[3]), pLoc.data(), pRot.data(),-1.0); // Apply the rotation corresponding to the final orientation to bring back to inertial frame

    std::vector<double> omegaCrosspRot(3,0.0);
    cross(&(totVelNode[3]), pRot.data(), omegaCrosspRot.data());

    for (size_t i=0; i < 3; i++)
        transVelNode[i] = totVelNode[i] + omegaCrosspRot[i];

}

void fsiTurbine::compute_error_norm(VectorFieldType * vec, VectorFieldType * vec_ref, stk::mesh::PartVector partVec, std::vector<double> & err) {

    const int ndim = meta_.spatial_dimension();
    std::vector<double> errorNorm(3,0);
    int nNodes = 0;

    stk::mesh::Selector sel( stk::mesh::selectUnion(partVec));
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* vecNode = stk::mesh::field_data(*vec, node);
            double* vecRefNode = stk::mesh::field_data(*vec_ref, node);
            for(size_t i=0; i < ndim; i++)
                errorNorm[i] += (vecNode[i] - vecRefNode[i])*(vecNode[i] - vecRefNode[i]);

            nNodes++;
        }
    }

    std::vector<double> g_errorNorm(3,0.0);
    stk::all_reduce_sum(bulk_.parallel(), errorNorm.data(), g_errorNorm.data(), 3);

    int g_nNodes = 0;
    stk::all_reduce_sum(bulk_.parallel(), &nNodes, &g_nNodes, 1);

    for (size_t i=0; i < ndim; i++)
        err[i] = sqrt(g_errorNorm[i]/g_nNodes);

}

//! Apply a Wiener-Milenkovic rotation 'wm' to a vector 'r' into 'rRot'. To optionally transpose the rotation, set 'tranpose=-1.0'.
void fsiTurbine::applyWMrotation(double * wm, double * r, double *rRot, double transpose) {

    double wm0 = 2.0-0.125*dot(wm, wm);
    double nu = 2.0/(4.0-wm0);
    double cosPhiO2 = 0.5*wm0*nu;
    std::vector<double> wmCrossR(3,0.0);
    cross(wm, r, wmCrossR.data());
    std::vector<double> wmCrosswmCrossR(3,0.0);
    cross(wm, wmCrossR.data(), wmCrosswmCrossR.data());

    for(size_t i=0; i < 3; i++)
        rRot[i] = r[i] + transpose * nu * cosPhiO2 * wmCrossR[i] + 0.5 * nu * nu * wmCrosswmCrossR[i];

}

//! Map each node on the turbine surface CFD mesh to the blade beam mesh
void fsiTurbine::computeMapping() {

    const int ndim = meta_.spatial_dimension();
    VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");

    // Do the tower first
    stk::mesh::Selector sel(stk::mesh::selectUnion(twrParts_));
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b: bkts) {
        for (size_t in=0; in < b->size(); in++) {
            auto node = (*b)[in];
            double* xyz = stk::mesh::field_data(*modelCoords, node);
            int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
            double* dispMapInterpNode = stk::mesh::field_data(*dispMapInterp_, node);
            std::vector<double> ptCoords(ndim, 0.0);
            for(int i=0; i < ndim; i++)
                ptCoords[i] = xyz[i];
            bool foundProj = false;
            double nDimCoord = -1.0;
            int nPtsTwr = params_.nBRfsiPtsTwr;
            if (nPtsTwr > 0) {
                for (int i=0; i < nPtsTwr-1; i++) {
                    std::vector<double> lStart = {brFSIdata_.twr_ref_pos[i*6], brFSIdata_.twr_ref_pos[i*6+1], brFSIdata_.twr_ref_pos[i*6+2]};
                    std::vector<double> lEnd = {brFSIdata_.twr_ref_pos[(i+1)*6], brFSIdata_.twr_ref_pos[(i+1)*6+1], brFSIdata_.twr_ref_pos[(i+1)*6+2]};
                    nDimCoord = projectPt2Line(ptCoords, lStart, lEnd);

                    if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {
                        *dispMapInterpNode = nDimCoord;
                        *dispMapNode = i;
//                        *loadMapNode = i + std::round(nDimCoord);
                        foundProj = true;
                        break;
                    }
                }

                //If no element in the OpenFAST mesh contains this node do some sanity check on the perpendicular distance between the surface mesh node and the line joining the ends of the tower
                if (!foundProj) {
                    std::vector<double> lStart = {brFSIdata_.twr_ref_pos[0], brFSIdata_.twr_ref_pos[1], brFSIdata_.twr_ref_pos[2]};
                    std::vector<double> lEnd = {brFSIdata_.twr_ref_pos[(nPtsTwr-1)*6], brFSIdata_.twr_ref_pos[(nPtsTwr-1)*6+1], brFSIdata_.twr_ref_pos[(nPtsTwr-1)*6+2]};
                    double perpDist = perpProjectDist_Pt2Line(ptCoords, lStart, lEnd);
                    if (perpDist > 1.0) {// Something's wrong if a node on the surface mesh of the tower is more than 20% of the tower length away from the tower axis.
                        throw std::runtime_error("Can't find a projection for point (" + std::to_string(ptCoords[0]) + "," + std::to_string(ptCoords[1]) + "," + std::to_string(ptCoords[2]) + ") on the tower on turbine " + std::to_string(params_.TurbID) + ". The tower extends from " + std::to_string(lStart[0]) + "," + std::to_string(lStart[1]) + "," + std::to_string(lStart[2]) + ") to " + std::to_string(lEnd[0]) + "," + std::to_string(lEnd[1]) + "," + std::to_string(lEnd[2]) + "). Are you sure the initial position and orientation of the mesh is consistent with the input file parameters and the OpenFAST model.");
                    }
                    if (nDimCoord < 0.0)  {
                        //Assign this node to the first point and element of the OpenFAST mesh
                        *dispMapInterpNode = 0.0;
                        *dispMapNode = 0;
//                        *loadMapNode = 0;
                    } else if (nDimCoord > 1.0) { //Assign this node to the last point and element of the OpenFAST mesh
                        *dispMapInterpNode = 1.0;
                        *dispMapNode = nPtsTwr-2;
//                        *loadMapNode = nPtsTwr-1;
                    }
                }
            }
        }
    }

    // Now the blades
    int nBlades = params_.numBlades;
    int iStart = 0;
    for (int iBlade=0; iBlade < nBlades; iBlade++) {
        int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
        stk::mesh::Selector sel(stk::mesh::selectUnion(bladeParts_[iBlade]));
        const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

        for (auto b: bkts) {
            for (size_t in=0; in < b->size(); in++) {
                auto node = (*b)[in];
                double* xyz = stk::mesh::field_data(*modelCoords, node);
                int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
                double* dispMapInterpNode = stk::mesh::field_data(*dispMapInterp_, node);
                std::vector<double> ptCoords(ndim, 0.0);
                for(int i=0; i < ndim; i++)
                    ptCoords[i] = xyz[i];
                bool foundProj = false;
                double nDimCoord = -1.0;
                for (int i=0; i < nPtsBlade-1; i++) {
                    std::vector<double> lStart = {brFSIdata_.bld_ref_pos[(iStart+i)*6], brFSIdata_.bld_ref_pos[(iStart+i)*6+1], brFSIdata_.bld_ref_pos[(iStart+i)*6+2]};
                    std::vector<double> lEnd = {brFSIdata_.bld_ref_pos[(iStart+i+1)*6], brFSIdata_.bld_ref_pos[(iStart+i+1)*6+1], brFSIdata_.bld_ref_pos[(iStart+i+1)*6+2]};
                    nDimCoord = projectPt2Line(ptCoords, lStart, lEnd);

                    if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {
                        foundProj = true;
                        *dispMapInterpNode = nDimCoord;
                        *dispMapNode = i;
//                        *loadMapNode = i + std::round(nDimCoord);
                        break;
                    }
                }

                //If no element in the OpenFAST mesh contains this node do some sanity check on the perpendicular distance between the surface mesh node and the line joining the ends of the blade
                if (!foundProj) {

                    std::vector<double> lStart = {brFSIdata_.bld_ref_pos[iStart*6], brFSIdata_.bld_ref_pos[iStart*6+1], brFSIdata_.bld_ref_pos[iStart*6+2]};
                    std::vector<double> lEnd = {brFSIdata_.bld_ref_pos[(iStart+nPtsBlade-1)*6], brFSIdata_.bld_ref_pos[(iStart+nPtsBlade-1)*6+1], brFSIdata_.bld_ref_pos[(iStart+nPtsBlade-1)*6+2]};

                    // std::cout << "Can't find a projection for point (" + std::to_string(ptCoords[0]) + "," + std::to_string(ptCoords[1]) + "," + std::to_string(ptCoords[2]) + ") on blade " + std::to_string(iBlade) + " on turbine " + std::to_string(params_.TurbID) + ". The blade extends from " + std::to_string(lStart[0]) + "," + std::to_string(lStart[1]) + "," + std::to_string(lStart[2]) + ") to " + std::to_string(lEnd[0]) + "," + std::to_string(lEnd[1]) + "," + std::to_string(lEnd[2]) + ")." << std::endl ;
                    double perpDist = perpProjectDist_Pt2Line(ptCoords, lStart, lEnd);
                    if (perpDist > 1.0) {// Something's wrong if a node on the surface mesh of the blade is more than 20% of the blade length away from the blade axis.
                        throw std::runtime_error("Can't find a projection for point (" + std::to_string(ptCoords[0]) + "," + std::to_string(ptCoords[1]) + "," + std::to_string(ptCoords[2]) + ") on blade " + std::to_string(iBlade) + " on turbine " + std::to_string(params_.TurbID) + ". The blade extends from " + std::to_string(lStart[0]) + "," + std::to_string(lStart[1]) + "," + std::to_string(lStart[2]) + ") to " + std::to_string(lEnd[0]) + "," + std::to_string(lEnd[1]) + "," + std::to_string(lEnd[2]) + "). Are you sure the initial position and orientation of the mesh is consistent with the input file parameters and the OpenFAST model.");
                    }

                    if (nDimCoord < 0.0)  {
                        //Assign this node to the first point and element of the OpenFAST mesh
                            *dispMapInterpNode = 0.0;
                            *dispMapNode = 0;
//                            *loadMapNode = 0;
                    } else if (nDimCoord > 1.0) { //Assign this node to the last point and element of the OpenFAST mesh
                        *dispMapInterpNode = 1.0;
                        *dispMapNode = nPtsBlade-2;
//                        *loadMapNode = nPtsBlade-1;
                    }
                }
            }
        }
        iStart += nPtsBlade;
    }

    //Write reference positions to netcdf file
    //write_nc_ref_pos();

}

//! Map each sub-control surface on the turbine surface CFD mesh to the blade beam mesh
void fsiTurbine::computeLoadMapping() {

    const int ndim = meta_.spatial_dimension();
    VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "coordinates");

    // nodal fields to gather
    std::vector<double> ws_coordinates;
    std::vector<double> coord_bip(3,0.0);
    std::vector<double> ws_face_shape_function;

    // Do the tower first
    stk::mesh::Selector sel(meta_.locally_owned_part() & stk::mesh::selectUnion(twrBndyParts_));
    const auto& bkts =  bulk_.get_buckets( meta_.side_rank(), sel );
    
    for (auto b: bkts) {
        // face master element
        MasterElement *meFC =
            MasterElementRepo::get_surface_master_element(b->topology());
        const int nodesPerFace = meFC->nodesPerElement_;
        const int numScsBip = meFC->num_integration_points();

        // mapping from ip to nodes for this ordinal;
        // face perspective (use with face_node_relations)
        const int *faceIpNodeMap = meFC->ipNodeMap();

        ws_face_shape_function.resize(numScsBip*nodesPerFace);
        meFC->shape_fcn(ws_face_shape_function.data());

        ws_coordinates.resize(ndim*nodesPerFace);
        
        for (size_t in=0; in < b->size(); in++) {

            // get face
            stk::mesh::Entity face = (*b)[in];
            // face node relations
            stk::mesh::Entity const * face_node_rels = bulk_.begin_nodes(face);
            // gather nodal data off of face
            for ( int ni = 0; ni < nodesPerFace; ++ni ) {
                stk::mesh::Entity node = face_node_rels[ni];
                // gather coordinates
                const double* xyz = stk::mesh::field_data(*modelCoords, node);
                for (auto i=0; i < ndim; i++)
                    ws_coordinates[ni*ndim+i] = xyz[i];
            }

            // Get reference to load map and loadMapInterp at all ips on this face
            int* loadMapFace = stk::mesh::field_data(*loadMap_, face);
            double* loadMapInterpFace = stk::mesh::field_data(*loadMapInterp_, face);
            const double* pforce = stk::mesh::field_data(*pressureForceSCS_, face);
            const double* vforce = stk::mesh::field_data(*tauWallSCS_, face);
             
            for ( int ip = 0; ip < numScsBip; ++ip ) {
                //Get coordinates of this ip
                for (auto i=0; i < ndim; i++)
                    coord_bip[i] = 0.0;
                for (int ni = 0; ni < nodesPerFace; ni++) {
                    for (int i=0; i < ndim; i++)
                        coord_bip[i] += ws_face_shape_function[ip*nodesPerFace + ni]
                            * ws_coordinates[ni*ndim+i];
                }

                //Create map at this ip
                bool foundProj = false;
                double nDimCoord = -1.0;
                int nPtsTwr = params_.nBRfsiPtsTwr;
                if (nPtsTwr > 0) {
                    for (int i=0; i < nPtsTwr-1; i++) {
                        std::vector<double> lStart = {brFSIdata_.twr_ref_pos[i*6],
                                                      brFSIdata_.twr_ref_pos[i*6+1],
                                                      brFSIdata_.twr_ref_pos[i*6+2]};
                        std::vector<double> lEnd = {brFSIdata_.twr_ref_pos[(i+1)*6],
                                                    brFSIdata_.twr_ref_pos[(i+1)*6+1],
                                                    brFSIdata_.twr_ref_pos[(i+1)*6+2]};
                        nDimCoord = projectPt2Line(coord_bip, lStart, lEnd);
                    
                        if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {
                            loadMapInterpFace[ip] = nDimCoord;
                            loadMapFace[ip] = i;
                            foundProj = true;
                            break;
                        }
                    }
                    
                    //If no element in the OpenFAST mesh contains this node do
                    //some sanity check on the perpendicular distance between
                    //the surface mesh node and the line joining the ends of the
                    //tower
                    if (!foundProj) {
                        std::vector<double> lStart =
                            {brFSIdata_.twr_ref_pos[0],
                             brFSIdata_.twr_ref_pos[1],
                             brFSIdata_.twr_ref_pos[2]};
                        std::vector<double> lEnd =
                            {brFSIdata_.twr_ref_pos[(nPtsTwr-1)*6],
                             brFSIdata_.twr_ref_pos[(nPtsTwr-1)*6+1],
                             brFSIdata_.twr_ref_pos[(nPtsTwr-1)*6+2]};
                        double perpDist =
                            perpProjectDist_Pt2Line(coord_bip, lStart, lEnd);
                        // Something's wrong if a node on the surface mesh of
                        // the tower is more than 20% of the tower length away
                        // from the tower axis.
                        if (perpDist > 1.0) {
                            throw std::runtime_error(
                                "Can't find a projection for point ("
                                + std::to_string(coord_bip[0]) + ","
                                + std::to_string(coord_bip[1]) + ","
                                + std::to_string(coord_bip[2])
                                + ") on the tower on turbine "
                                + std::to_string(params_.TurbID)
                                + ". The tower extends from "
                                + std::to_string(lStart[0])
                                + "," + std::to_string(lStart[1])
                                + "," + std::to_string(lStart[2])
                                + ") to " + std::to_string(lEnd[0]) + ","
                                + std::to_string(lEnd[1]) + ","
                                + std::to_string(lEnd[2])
                                + "). Are you sure the initial position and orientation of the mesh is consistent with the input file parameters and the OpenFAST model.");
                        }
                        if (nDimCoord < 0.0)  {
                            //Assign this node to the first point and
                            //element of the OpenFAST mesh
                            loadMapInterpFace[ip] = 0.0;
                            loadMapFace[ip] = 0;
                        } else if (nDimCoord > 1.0) {
                            //Assign this node to the last point and
                            //element of the OpenFAST mesh
                            loadMapInterpFace[ip] = 1.0;
                            loadMapFace[ip] = nPtsTwr-2;
                        }
                    }
                }
            }
        }
    }


    // int nBlades = params_.numBlades;
    // int iStart = 0;
    // for (int iBlade=0; iBlade < nBlades; iBlade++) {
    //     int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
    //     bld_rmm_[iStart][0] = brFSIdata_.bld_rloc[iStart];
    //     bld_rmm_[iStart][1] = 0.5*(brFSIdata_.bld_rloc[iStart] + brFSIdata_.bld_rloc[iStart+1]);
    //     bld_dr_[iStart] = (bld_rmm_[iStart][1]-bld_rmm_[iStart][0]);
    //     // if (!bulk_.parallel_rank())
    //     //     std::cout << "i = 0, " << bld_rmm_[iStart][0] << " - " << bld_rmm_[iStart][1] << ", dr = " << bld_dr_[iStart] << std::endl ;
    //     for (int i=1; i < nPtsBlade-1; i++) {
    //         bld_rmm_[iStart + i][0] = 0.5*(brFSIdata_.bld_rloc[iStart+i-1] + brFSIdata_.bld_rloc[iStart+i]);
    //         bld_rmm_[iStart + i][1] = 0.5*(brFSIdata_.bld_rloc[iStart+i] + brFSIdata_.bld_rloc[iStart+i+1]);
    //         bld_dr_[iStart + i] = (bld_rmm_[iStart + i][1]-bld_rmm_[iStart + i][0]);
    //         // if (!bulk_.parallel_rank())
    //         //     std::cout << "i = " << i << ", " << bld_rmm_[iStart+i][0] << " - " << bld_rmm_[iStart+i][1] << ", dr = " << bld_dr_[iStart+i] << std::endl ;
    //     }
    //     bld_rmm_[iStart+nPtsBlade-1][0] = 0.5*(brFSIdata_.bld_rloc[iStart+nPtsBlade-2]+brFSIdata_.bld_rloc[iStart+nPtsBlade-1]);
    //     bld_rmm_[iStart+nPtsBlade-1][1] = brFSIdata_.bld_rloc[iStart+nPtsBlade-1];
    //     bld_dr_[iStart+nPtsBlade-1] = (bld_rmm_[iStart+nPtsBlade-1][1]-bld_rmm_[iStart+nPtsBlade-1][0]);
    //     // if (!bulk_.parallel_rank())
    //     //     std::cout << "i = " << nPtsBlade-1 << ", " << bld_rmm_[iStart+nPtsBlade-1][0] << " - " << bld_rmm_[iStart+nPtsBlade-1][1] << ", dr = " << bld_dr_[iStart+nPtsBlade-1] << std::endl ;
        
    //     iStart += nPtsBlade;
    // }
     
    // Now the blades
    int nBlades = params_.numBlades;
    int iStart = 0;
    for (int iBlade=0; iBlade < nBlades; iBlade++) {
        std::vector<double> cfd_mesh_rloc;
        int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
        stk::mesh::Selector sel(meta_.locally_owned_part() & stk::mesh::selectUnion(bladeBndyParts_[iBlade]));
        const auto& bkts =  bulk_.get_buckets( meta_.side_rank(), sel );

        for (auto b: bkts) {
            // face master element
            MasterElement *meFC =
                MasterElementRepo::get_surface_master_element(b->topology());
            const int nodesPerFace = meFC->nodesPerElement_;
            const int numScsBip = meFC->num_integration_points();

            // mapping from ip to nodes for this ordinal;
            // face perspective (use with face_node_relations)
            const int *faceIpNodeMap = meFC->ipNodeMap();

            ws_face_shape_function.resize(numScsBip*nodesPerFace);
            meFC->shape_fcn(ws_face_shape_function.data());

            ws_coordinates.resize(ndim*nodesPerFace);
            
            for (size_t in=0; in < b->size(); in++) {

                // get face
                stk::mesh::Entity face = (*b)[in];
                // face node relations
                stk::mesh::Entity const * face_node_rels = bulk_.begin_nodes(face);
                // gather nodal data off of face
                for ( int ni = 0; ni < nodesPerFace; ++ni ) {
                    stk::mesh::Entity node = face_node_rels[ni];
                    // gather coordinates
                    const double* xyz = stk::mesh::field_data(*modelCoords, node);
                    for (auto i=0; i < ndim; i++)
                        ws_coordinates[ni*ndim+i] = xyz[i];
                }

                // Get reference to load map and loadMapInterp at all ips on this face
                int* loadMapFace = stk::mesh::field_data(*loadMap_, face);
                double* loadMapInterpFace = stk::mesh::field_data(*loadMapInterp_, face);

                for ( int ip = 0; ip < numScsBip; ++ip ) {
                    //Get coordinates of this ip
                    for (auto i=0; i < ndim; i++)
                        coord_bip[i] = 0.0;
                    for (int ni = 0; ni < nodesPerFace; ni++) {
                        for (int i=0; i < ndim; i++)
                            coord_bip[i] += ws_face_shape_function[ip*nodesPerFace + ni]
                                * ws_coordinates[ni*ndim+i];
                    }
                
                    bool foundProj = false;
                    double nDimCoord = -1.0;
                    for (int i=0; i < nPtsBlade-1; i++) {
                        std::vector<double> lStart = {
                            brFSIdata_.bld_ref_pos[(iStart+i)*6],
                            brFSIdata_.bld_ref_pos[(iStart+i)*6+1],
                            brFSIdata_.bld_ref_pos[(iStart+i)*6+2]};
                        std::vector<double> lEnd = {
                            brFSIdata_.bld_ref_pos[(iStart+i+1)*6],
                            brFSIdata_.bld_ref_pos[(iStart+i+1)*6+1],
                            brFSIdata_.bld_ref_pos[(iStart+i+1)*6+2]};
                        nDimCoord = projectPt2Line(coord_bip, lStart, lEnd);
                        if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {                            
                            foundProj = true;
                            loadMapInterpFace[ip] = nDimCoord; 
                            loadMapFace[ip] = i;
                            break;
                        }
                    }

                    //If no element in the OpenFAST mesh contains this
                    //node do some sanity check on the perpendicular
                    //distance between the surface mesh node and the line
                    //joining the ends of the blade
                    if (!foundProj) {

                        std::vector<double> lStart = {
                            brFSIdata_.bld_ref_pos[iStart*6],
                            brFSIdata_.bld_ref_pos[iStart*6+1],
                            brFSIdata_.bld_ref_pos[iStart*6+2]};
                        std::vector<double> lEnd = {
                            brFSIdata_.bld_ref_pos[(iStart+nPtsBlade-1)*6],
                            brFSIdata_.bld_ref_pos[(iStart+nPtsBlade-1)*6+1],
                            brFSIdata_.bld_ref_pos[(iStart+nPtsBlade-1)*6+2]};

                        // std::cout << "Can't find a projection for point (" + std::to_string(coord_bip[0]) + "," + std::to_string(coord_bip[1]) + "," + std::to_string(coord_bip[2]) + ") on blade " + std::to_string(iBlade) + " on turbine " + std::to_string(params_.TurbID) + ". The blade extends from " + std::to_string(lStart[0]) + "," + std::to_string(lStart[1]) + "," + std::to_string(lStart[2]) + ") to " + std::to_string(lEnd[0]) + "," + std::to_string(lEnd[1]) + "," + std::to_string(lEnd[2]) + ")." << std::endl ;
                        double perpDist =
                            perpProjectDist_Pt2Line(coord_bip, lStart, lEnd);
                        // Something's wrong if a node on the surface
                        // mesh of the blade is more than 20% of the
                        // blade length away from the blade axis.
                        if (perpDist > 1.0) {
                            throw std::runtime_error(
                                "Can't find a projection for point ("
                                + std::to_string(coord_bip[0]) + ","
                                + std::to_string(coord_bip[1]) + ","
                                + std::to_string(coord_bip[2])
                                + ") on blade "
                                + std::to_string(iBlade)
                                + " on turbine "
                                + std::to_string(params_.TurbID)
                                + ". The blade extends from "
                                + std::to_string(lStart[0]) + ","
                                + std::to_string(lStart[1]) + ","
                                + std::to_string(lStart[2]) + ") to "
                                + std::to_string(lEnd[0]) + ","
                                + std::to_string(lEnd[1]) + ","
                                + std::to_string(lEnd[2])
                                + "). Are you sure the initial position and orientation of the mesh is consistent with the input file parameters and the OpenFAST model.");
                        }

                        if (nDimCoord < 0.0)  {
                            //Assign this node to the first point and element of the OpenFAST mesh
                            loadMapInterpFace[ip] = 0.0;
                            loadMapFace[ip] = 0;
                        } else if (nDimCoord > 1.0) { //Assign this node to the last point and element of the OpenFAST mesh
                            loadMapInterpFace[ip] = 1.0;//brFSIdata_.bld_rloc[iStart+nPtsBlade-1];
                            loadMapFace[ip] = nPtsBlade-2;
                        }
                    }
                }
            }
        }

        iStart += nPtsBlade;
    }
}


/** Project a point 'pt' onto a line from 'lStart' to 'lEnd' and return the non-dimensional location of the projected point along the line in [0-1] coordinates
    \f[
    nonDimCoord = \frac{ (\vec{pt} - \vec{lStart}) \cdot ( \vec{lEnd} - \vec{lStart} ) }{ (\vec{lEnd} - \vec{lStart}) \cdot (\vec{lEnd} - \vec{lStart}) }
    \f]
*/
double fsiTurbine::projectPt2Line(std::vector<double> & pt, std::vector<double> & lStart, std::vector<double> & lEnd) {

    double nonDimCoord = 0.0;

    double num = 0.0;
    double denom = 0.0;

    for (int i=0; i < 3; i++) {
        num += (pt[i] - lStart[i]) * (lEnd[i] - lStart[i]) ;
        denom += (lEnd[i] - lStart[i]) * (lEnd[i] - lStart[i]) ;
    }

    nonDimCoord = num/denom;
    return nonDimCoord;
}

/** Project a point 'pt' onto a line from 'lStart' to 'lEnd' and return the non-dimensional distance of 'pt' from the line w.r.t the distance from 'lStart' to 'lEnd'
    \f[
    \vec{perp} &= (\vec{pt} - \vec{lStart}) - \frac{ (\vec{pt} - \vec{lStart}) \cdot ( \vec{lEnd} - \vec{lStart} ) }{ (\vec{lEnd} - \vec{lStart}) \cdot (\vec{lEnd} - \vec{lStart}) } ( \vec{lEnd} - \vec{lStart} ) \ \
    nonDimPerpDist = \frac{\lvert \vec{perp} \rvert}{ \lvert  (\vec{lEnd} - \vec{lStart}) \rvert }
    \f]
*/
double fsiTurbine::perpProjectDist_Pt2Line(std::vector<double> & pt, std::vector<double> & lStart, std::vector<double> & lEnd) {

    double nonDimCoord = 0.0;
    double num = 0.0;
    double denom = 0.0;
    for (int i=0; i < 3; i++) {
        num += (pt[i] - lStart[i]) * (lEnd[i] - lStart[i]) ;
        denom += (lEnd[i] - lStart[i]) * (lEnd[i] - lStart[i]) ;
    }
    nonDimCoord = num/denom;

    double nonDimPerpDist = 0.0;
    for(int i=0; i < 3; i++) {
        double tmp = (pt[i] - lStart[i]) - nonDimCoord * (lEnd[i] - lStart[i]) ;
        nonDimPerpDist += tmp * tmp ;
    }
    nonDimPerpDist = sqrt(nonDimPerpDist/denom) ;

    return nonDimPerpDist;
}

void fsiTurbine::compute_div_mesh_velocity() {

    VectorFieldType* meshVelocity = meta_.get_field<VectorFieldType>
        (stk::topology::NODE_RANK, "mesh_velocity");

    ScalarFieldType* divMeshVel = meta_.get_field<ScalarFieldType>
        (stk::topology::NODE_RANK, "div_mesh_velocity");

    compute_vector_divergence(bulk_, partVec_, bndyPartVec_, meshVelocity, divMeshVel);

}

} // nalu


} // sierra
