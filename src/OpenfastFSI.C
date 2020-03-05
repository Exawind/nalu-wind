#include "OpenfastFSI.h"
#include "FSIturbine.h"
#include <NaluParsing.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

namespace sierra {

namespace nalu {

OpenfastFSI::OpenfastFSI(
    stk::mesh::MetaData& meta,
    stk::mesh::BulkData& bulk,
    const YAML::Node& node,
    SurfaceFMPostProcessing* sfm_pp
) : meta_(meta),
    bulk_(bulk),
    mesh_motion_(false),
    enable_calc_loads_(true),
    sfm_pp_(sfm_pp)
{
    load(node);
}

OpenfastFSI::~OpenfastFSI() {

    FAST.end() ;

}

void OpenfastFSI::read_turbine_data(
    int iTurb,
    fast::fastInputs & fi,
    YAML::Node turbNode)
{

    //Read turbine data for a given turbine using the YAML node
    if (turbNode["turb_id"]) {
        fi.globTurbineData[iTurb].TurbID = turbNode["turb_id"].as<int>();
    } else {
        turbNode["turb_id"] = iTurb;
    }
    if (turbNode["sim_type"]) {
        if (turbNode["sim_type"].as<std::string>() == "ext-loads") {
            fi.globTurbineData[iTurb].sType = fast::EXTLOADS;
            fsiTurbineData_[iTurb] = new fsiTurbine(iTurb, turbNode, meta_, bulk_);
            mesh_motion_ = true;
        } else {
            fi.globTurbineData[iTurb].sType = fast::EXTINFLOW;
        }
    } else {
        fi.globTurbineData[iTurb].sType = fast::EXTINFLOW;
    }
    if (turbNode["FAST_input_filename"]) {
        fi.globTurbineData[iTurb].FASTInputFileName = turbNode["FAST_input_filename"].as<std::string>() ;
    } else {
        fi.globTurbineData[iTurb].FASTInputFileName = "";
    }
    if (turbNode["restart_filename"]) {
        fi.globTurbineData[iTurb].FASTRestartFileName = turbNode["restart_filename"].as<std::string>() ;
    } else {
        fi.globTurbineData[iTurb].FASTRestartFileName = "";
    }
    if ( (fi.globTurbineData[iTurb].FASTRestartFileName == "") && (fi.globTurbineData[iTurb].FASTInputFileName == "") )
        throw std::runtime_error("Both FAST_input_filename and restart_filename are empty or not specified for Turbine " + std::to_string(iTurb));
    if (turbNode["turbine_base_pos"].IsSequence() ) {
        fi.globTurbineData[iTurb].TurbineBasePos = turbNode["turbine_base_pos"].as<std::vector<float> >() ;
    } else {
        fi.globTurbineData[iTurb].TurbineBasePos = std::vector<float>(3,0.0);
    }
    if (turbNode["turbine_hub_pos"].IsSequence() ) {
        fi.globTurbineData[iTurb].TurbineHubPos = turbNode["turbine_hub_pos"].as<std::vector<double> >() ;
    } else {
        fi.globTurbineData[iTurb].TurbineHubPos =  std::vector<double>(3,0.0);
    }
    if (turbNode["num_force_pts_blade"]) {
        fi.globTurbineData[iTurb].numForcePtsBlade = turbNode["num_force_pts_blade"].as<int>();
    } else {
        fi.globTurbineData[iTurb].numForcePtsBlade = 0;
    }
    if (turbNode["num_force_pts_tower"]) {
        fi.globTurbineData[iTurb].numForcePtsTwr = turbNode["num_force_pts_tower"].as<int>();
    } else {
        fi.globTurbineData[iTurb].numForcePtsTwr = 0;
    }
    if (turbNode["nacelle_cd"]) {
        fi.globTurbineData[iTurb].nacelle_cd = turbNode["nacelle_cd"].as<float>();
    } else {
        fi.globTurbineData[iTurb].nacelle_cd = 0.0;
    }
    if (turbNode["nacelle_area"]) {
        fi.globTurbineData[iTurb].nacelle_area = turbNode["nacelle_area"].as<float>();
    } else {
        fi.globTurbineData[iTurb].nacelle_area = 0.0;
    }
    if (turbNode["air_density"]) {
        fi.globTurbineData[iTurb].air_density = turbNode["air_density"].as<float>();
    } else {
        fi.globTurbineData[iTurb].air_density = 0.0;
    }
}

void OpenfastFSI::load(const YAML::Node& node)
{

    fi.comm = MPI_COMM_WORLD;

    get_required(node, "n_turbines_glob", fi.nTurbinesGlob);

    if (fi.nTurbinesGlob > 0) {

        get_if_present(node, "dry_run", fi.dryRun, false);
        get_if_present(node, "debug", fi.debug, false);
        std::string simStartType = "na";
        get_required(node, "sim_start", simStartType);
        if (simStartType == "init") {
            fi.simStart = fast::INIT;
        } else if (simStartType == "trueRestart") {
            fi.simStart = fast::TRUERESTART;
        } else if (simStartType == "restartDriverInitFAST") {
            fi.simStart = fast::RESTARTDRIVERINITFAST;
        }
        get_required(node, "t_max", fi.tMax); // tMax is the total
        // duration to which you want to run FAST.  This should be the
        // same or greater than the max time given in the FAST fst
        // file. Choose this carefully as FAST writes the output file
        // only at this point if you choose the binary file output.
        
        if (node["super_controller"]) {
            get_required(node, "super_controller", fi.scStatus);
            get_required(node, "sc_libFile", fi.scLibFile);
            get_required(node, "num_sc_inputs", fi.numScInputs);
            get_required(node, "num_sc_outputs", fi.numScOutputs);
        }

        fsiTurbineData_.resize(fi.nTurbinesGlob);
        fi.globTurbineData.resize(fi.nTurbinesGlob);
        for (int iTurb=0; iTurb < fi.nTurbinesGlob; iTurb++) {
            if (node["Turbine" + std::to_string(iTurb)]) {
                read_turbine_data(iTurb, fi, node["Turbine" + std::to_string(iTurb)] );
            } else {
                throw std::runtime_error("Node for Turbine" + std::to_string(iTurb) + " not present in input file or I cannot read it");
            }
        }

    } else {
        throw std::runtime_error("Number of turbines <= 0 ");
    }

    FAST.setInputs(fi);

}

void OpenfastFSI::setup() {

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != NULL) // This may not be a turbine intended for blade-resolved simulation
            fsiTurbineData_[i]->setup(sfm_pp_);

    }

}

void OpenfastFSI::initialize(double dtNalu, double restartFreqNalu, double curTime)
{

    FAST.allocateTurbinesToProcsSimple();
    FAST.setDriverTimeStep(dtNalu);
    FAST.init();
    FAST.setDriverCheckpoint(restartFreqNalu);
    //TODO: Check here on the processor containing the turbine that the number of blades on the turbine is the same as the number of blade parts specified in the input file.

    //TODO: In the documentation, mention that the CFD mesh must always be created for the turbine in the reference position defined in OpenFAST, i.e. with blade1 pointing up and the other blades following it in order as the turbine rotates clockwise facing downwind. If the mesh is not created this way, the mapping won't work. Any non-zero initial azimuth and/or initial yaw must be only specified in the OpenFAST input file and the mesh will automatically be deformed after calling solution0. Requiring the initial CFD mesh to be in the reference configuration may not always work if the mesh domain and initial yaw setting does not align with the reference configuration. May be this is isn't an issue because unlike AeroDyn, ExtLoads does not create the initial mesh independent of the ElastoDyn/BeamDyn. May be ExtLoads already has the correct yaw and azimuth setting from OpenFAST after the init call. In this case, the CFD mesh must start in the correct azimuth and yaw configuration. In which case, the initial yaw and azimuth must be obtained from OpenFAST and the mesh around the turbine must be deformed through rigid body motion first before starting any mapping.

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        // This may not be a turbine intended for blade-resolved simulation
        if(fsiTurbineData_[i] != NULL) {
            int turbProc = FAST.getProc(i);
            fsiTurbineData_[i]->setProc(turbProc);
            if (bulk_.parallel_rank() == turbProc) {
                FAST.get_turbineParams(i, fsiTurbineData_[i]->params_);
            }
            bcast_turbine_params(i);
            fsiTurbineData_[i]->initialize();
         }
    }

    // for (int i=0; i < nTurbinesGlob; i++) {
    //     // This may not be a turbine intended for blade-resolved simulation
    //     if(fsiTurbineData_[i] != NULL) {
    //         std::vector<double> hub_center(3,0.0);
    //         FAST.getHubRefPosition(hub_center, i);
    //         sfm_pp_.set_centroid_coords(i, hub_center.data());
    //     }
    // }

        compute_mapping();
        if (FAST.isTimeZero()) {
            send_loads(0.0);
            FAST.solution0();
        }
        get_displacements(curTime);
        
}


void OpenfastFSI::bcast_turbine_params(int iTurb) {

    std::vector<int> tIntParams(7,0); //Assumes a max number of blades of 3
    std::vector<double> tDoubleParams(6,0.0);
    int turbProc = fsiTurbineData_[iTurb]->getProc();
    if (bulk_.parallel_rank() == turbProc) {
        tIntParams[0] = fsiTurbineData_[iTurb]->params_.TurbID;
        tIntParams[1] = fsiTurbineData_[iTurb]->params_.numBlades;
        tIntParams[2] = fsiTurbineData_[iTurb]->params_.nTotBRfsiPtsBlade;
        tIntParams[3] = fsiTurbineData_[iTurb]->params_.nBRfsiPtsTwr;
        for (int i=0; i < fsiTurbineData_[iTurb]->params_.numBlades; i++) {
            tIntParams[4+i] = fsiTurbineData_[iTurb]->params_.nBRfsiPtsBlade[i];
        }

        for(int i=0; i < 3; i++) {
            tDoubleParams[i] = fsiTurbineData_[iTurb]->params_.TurbineBasePos[i];
            tDoubleParams[3+i] = fsiTurbineData_[iTurb]->params_.TurbineHubPos[i];
        }

    }
    int iError = MPI_Bcast(tIntParams.data(), 7, MPI_INT, turbProc, bulk_.parallel());
    iError = MPI_Bcast(tDoubleParams.data(), 6, MPI_DOUBLE, turbProc, bulk_.parallel());

    if (bulk_.parallel_rank() != turbProc) {
        fsiTurbineData_[iTurb]->params_.TurbID = tIntParams[0];
        fsiTurbineData_[iTurb]->params_.numBlades = tIntParams[1] ;
        fsiTurbineData_[iTurb]->params_.nTotBRfsiPtsBlade = tIntParams[2] ;
        fsiTurbineData_[iTurb]->params_.nBRfsiPtsTwr = tIntParams[3] ;
        fsiTurbineData_[iTurb]->params_.nBRfsiPtsBlade.resize(fsiTurbineData_[iTurb]->params_.numBlades);
        for (int i=0; i < fsiTurbineData_[iTurb]->params_.numBlades; i++) {
            fsiTurbineData_[iTurb]->params_.nBRfsiPtsBlade[i] = tIntParams[4+i] ;
        }

        fsiTurbineData_[iTurb]->params_.TurbineBasePos.resize(3);
        fsiTurbineData_[iTurb]->params_.TurbineHubPos.resize(3);
        for(int i=0; i < 3; i++) {
            fsiTurbineData_[iTurb]->params_.TurbineBasePos[i] = tDoubleParams[i];
            fsiTurbineData_[iTurb]->params_.TurbineHubPos[i] = tDoubleParams[3+i];
        }

    }

}

void OpenfastFSI::compute_mapping() {

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != NULL) {// This may not be a turbine intended for blade-resolved simulation
            int turbProc = fsiTurbineData_[i]->getProc();
            if (bulk_.parallel_rank() == turbProc) {
                FAST.getTowerRefPositions(fsiTurbineData_[i]->brFSIdata_.twr_ref_pos, i);
                FAST.getBladeRefPositions(fsiTurbineData_[i]->brFSIdata_.bld_ref_pos, i);
                FAST.getHubRefPosition(fsiTurbineData_[i]->brFSIdata_.hub_ref_pos, i);
                FAST.getNacelleRefPosition(fsiTurbineData_[i]->brFSIdata_.nac_ref_pos, i);
                FAST.getBladeRloc(fsiTurbineData_[i]->brFSIdata_.bld_rloc, i);
                FAST.getBladeChord(fsiTurbineData_[i]->brFSIdata_.bld_chord, i);
            }

            int iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.twr_ref_pos.data(), (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6, MPI_DOUBLE, turbProc, bulk_.parallel());
            int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.bld_ref_pos.data(), nTotBldNodes*6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.hub_ref_pos.data(), 6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.nac_ref_pos.data(), 6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.bld_rloc.data(), nTotBldNodes, MPI_DOUBLE, turbProc, bulk_.parallel());
            // No need to bcast chord
            if (! bulk_.parallel_rank())
                std::cerr << "Computing mapping " << std::endl ;
            fsiTurbineData_[i]->computeMapping();
            fsiTurbineData_[i]->computeLoadMapping();
        }
    }

}

void OpenfastFSI::predict_struct_states()
{
    FAST.predict_states();
}

void OpenfastFSI::predict_struct_timestep(const double curTime)
{
    send_loads(curTime);
    FAST.update_states_driver_time_step();
}

void OpenfastFSI::advance_struct_timestep(const double curTime)
{
    
    FAST.advance_to_next_driver_time_step();

    tStep_ += 1;
    
    // int nTurbinesGlob = FAST.get_nTurbinesGlob();
    // for (int i=0; i < nTurbinesGlob; i++) {
    //     if(fsiTurbineData_[i] != nullptr)
    //         fsiTurbineData_[i]->write_nc_def_loads(tStep_, curTime);
    // }
    
}


void OpenfastFSI::send_loads(const double curTime) {

    if (sfm_pp_ != nullptr)
        sfm_pp_->execute();

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != NULL) {// This may not be a turbine intended for blade-resolved simulation
            int turbProc = fsiTurbineData_[i]->getProc();
            fsiTurbineData_[i]->mapLoads();

            int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
            if (bulk_.parallel_rank() == turbProc) {
                int iError = MPI_Reduce(MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.twr_ld.data(), (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
                for(int k=0; k < (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6; k++)
                    fsiTurbineData_[i]->brFSIdata_.twr_ld[k] = fsiTurbineData_[i]->brFSIdata_.twr_ld[k];
                FAST.setTowerForces(fsiTurbineData_[i]->brFSIdata_.twr_ld, i);

                iError = MPI_Reduce(MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.bld_ld.data(), nTotBldNodes*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
                for(int k=0; k < nTotBldNodes*6; k++)
                    fsiTurbineData_[i]->brFSIdata_.bld_ld[k] = fsiTurbineData_[i]->brFSIdata_.bld_ld[k];
                FAST.setBladeForces(fsiTurbineData_[i]->brFSIdata_.bld_ld, i);

            } else {
                int iError = MPI_Reduce(fsiTurbineData_[i]->brFSIdata_.twr_ld.data(), NULL, (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
                iError = MPI_Reduce(fsiTurbineData_[i]->brFSIdata_.bld_ld.data(), NULL, (nTotBldNodes)*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
            }
        }
    }

}

void OpenfastFSI::get_displacements(double current_time) {

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != NULL) {// This may not be a turbine intended for blade-resolved simulation
            int turbProc = fsiTurbineData_[i]->getProc();
            if (bulk_.parallel_rank() == turbProc) {
                FAST.getTowerDisplacements(fsiTurbineData_[i]->brFSIdata_.twr_def, fsiTurbineData_[i]->brFSIdata_.twr_vel, i);
                FAST.getBladeDisplacements(fsiTurbineData_[i]->brFSIdata_.bld_def, fsiTurbineData_[i]->brFSIdata_.bld_vel, i);
                FAST.getHubDisplacement(fsiTurbineData_[i]->brFSIdata_.hub_def, fsiTurbineData_[i]->brFSIdata_.hub_vel, i);
                FAST.getNacelleDisplacement(fsiTurbineData_[i]->brFSIdata_.nac_def, fsiTurbineData_[i]->brFSIdata_.nac_vel, i);
            }

            int iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.twr_def.data(), (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6, MPI_DOUBLE, turbProc, bulk_.parallel());
            int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.bld_def.data(), nTotBldNodes*6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.hub_def.data(), 6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.nac_def.data(), 6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.twr_vel.data(), (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.bld_vel.data(), nTotBldNodes*6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.hub_vel.data(), 6, MPI_DOUBLE, turbProc, bulk_.parallel());
            iError = MPI_Bcast(fsiTurbineData_[i]->brFSIdata_.nac_vel.data(), 6, MPI_DOUBLE, turbProc, bulk_.parallel());

            // if (bulk_.parallel_rank() == turbProc) {
            //     std::ofstream bld_bm_mesh;
            //     bld_bm_mesh.open("blade_beam_mesh.csv", std::ios_base::out) ;
            //     for(int k=0; k < nTotBldNodes; k++) {
            //         bld_bm_mesh << fsiTurbineData_[i]->brFSIdata_.bld_ref_pos[k*6] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_ref_pos[k*6+1] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_ref_pos[k*6+2] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_def[k*6] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_def[k*6+1] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_def[k*6+2] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_def[k*6+3] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_def[k*6+4] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_def[k*6+5] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_vel[k*6] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_vel[k*6+1] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_vel[k*6+2] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_vel[k*6+3] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_vel[k*6+4] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.bld_vel[k*6+5] << std::endl;

            //     }

            //     bld_bm_mesh << fsiTurbineData_[i]->brFSIdata_.nac_ref_pos[0] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_ref_pos[1] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_ref_pos[2] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_def[0] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_def[1] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_def[2] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_def[3] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_def[4] << ","
            //                 << fsiTurbineData_[i]->brFSIdata_.nac_def[5] << std::endl;


            //     for (int k=0; k < fsiTurbineData_[i]->params_.nBRfsiPtsTwr; k++) {

            //         bld_bm_mesh << fsiTurbineData_[i]->brFSIdata_.twr_ref_pos[k*6+0] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.twr_ref_pos[k*6+1] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.twr_ref_pos[k*6+2] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.twr_def[k*6+0] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.twr_def[k*6+1] << ","
            //                     << fsiTurbineData_[i]->brFSIdata_.twr_def[k*6+2] << std::endl;
            //     }
            //     bld_bm_mesh.close();
            // }

//            fsiTurbineData_[i]->setSampleDisplacement(current_time);

            //For testing purposes
//            fsiTurbineData_[i]->setRefDisplacement(current_time);
        }
    }
}

void OpenfastFSI::compute_div_mesh_velocity() {

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != NULL) // This may not be a turbine intended for blade-resolved simulation
            fsiTurbineData_[i]->compute_div_mesh_velocity();
    }

}

void OpenfastFSI::set_rotational_displacement(std::array<double,3> axis, double omega, double curTime) {

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != NULL) // This may not be a turbine intended for blade-resolved simulation
            fsiTurbineData_[i]->setRotationDisplacement(axis, omega, curTime);
    }

    
}
void OpenfastFSI::map_displacements(double current_time)
{

    get_displacements(current_time); // Get displacements from the OpenFAST - C++ API

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != NULL) {// This may not be a turbine intended for blade-resolved simulation {
//            fsiTurbineData_[i]->setSampleDisplacement(current_time);
//            fsiTurbineData_[i]->setRefDisplacement(current_time);
            fsiTurbineData_[i]->mapDisplacements();
        }
    }

}

void OpenfastFSI::map_loads(const int tStep, const double curTime)
{

    int nTurbinesGlob = FAST.get_nTurbinesGlob();
    for (int i=0; i < nTurbinesGlob; i++) {
        if(fsiTurbineData_[i] != nullptr) {// This may not be a turbine intended for blade-resolved simulation
            int turbProc = fsiTurbineData_[i]->getProc();
            fsiTurbineData_[i]->mapLoads();
            int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
            if (bulk_.parallel_rank() == turbProc) {
                std::cerr << "nTotBldNodes = " << nTotBldNodes << std::endl ;
                int iError = MPI_Reduce(MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.twr_ld.data(), (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
                iError = MPI_Reduce(MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.bld_ld.data(), nTotBldNodes*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
            } else {
                int iError = MPI_Reduce(fsiTurbineData_[i]->brFSIdata_.twr_ld.data(), NULL, (fsiTurbineData_[i]->params_.nBRfsiPtsTwr)*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
                iError = MPI_Reduce(fsiTurbineData_[i]->brFSIdata_.bld_ld.data(), NULL, (nTotBldNodes)*6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_.parallel());
            }
            if (bulk_.parallel_rank() == turbProc) {
                for (size_t j=0 ; j < nTotBldNodes; j++) {
                    std::cerr
                        << fsiTurbineData_[i]->brFSIdata_.bld_ld[j*6+0] << " "
                        << fsiTurbineData_[i]->brFSIdata_.bld_ld[j*6+1] << " "
                        << fsiTurbineData_[i]->brFSIdata_.bld_ld[j*6+2] << " "
                        << fsiTurbineData_[i]->brFSIdata_.bld_ld[j*6+3] << " "
                        << fsiTurbineData_[i]->brFSIdata_.bld_ld[j*6+4] << " "
                        << fsiTurbineData_[i]->brFSIdata_.bld_ld[j*6+5] << std::endl;
                }
            }
            
            fsiTurbineData_[i]->write_nc_def_loads(tStep, curTime);
        }
    }

}

} // nalu

} // sierra
