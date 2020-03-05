/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SURFACEFMPOSTPROCESSING_H
#define SURFACEFMPOSTPROCESSING_H

#include <FieldTypeDef.h>

#include <stk_mesh/base/Part.hpp>

#include <yaml-cpp/yaml.h>

// basic c++
#include <string>
#include <memory>

namespace sierra {
    namespace nalu {

        class Realm;

        struct SurfaceFMData {
            stk::mesh::PartVector partVector_;
            std::vector<std::string> partNames_;
            std::string outputFileName_;
            std::array<double,3> centroidCoords_ {{0.0,0.0,0.0}};
            int frequency_ {1};
            int iSurface_;
            bool wallFunction_ {false};

            SurfaceFMData() {}

            SurfaceFMData(std::vector<std::string> partNames,
                          std::string outputFileName,
                          int iSurface):
                partNames_(partNames),
                outputFileName_(outputFileName),
                iSurface_(iSurface)
                {}
        };

/** Post-processing to compute force and moment on various surfaces

 *  This class implements computing the force and moment on various surfaces

 *  Currently supported:
 *    - Compute pressure (pressureForce) and viscous force (tau_wall)
 *    - Use of wall function to compute viscous force in under-resolved
 *      turbulent flow simulations
 *    - Computing yPlus based on distance of first node from the wall and
 total tangential stress
*/

        class SurfaceFMPostProcessing
        {
        public:
            SurfaceFMPostProcessing(Realm&);

            ~SurfaceFMPostProcessing() = default;

            void load(const YAML::Node & y_node);

            void register_surface_pp(const SurfaceFMData&);

            void setup();

            void set_centroid_coords(int iSurface,
                                     double * centroid);

            void execute();

        private:
            Realm& realm_;

            std::vector<SurfaceFMData> surfaceFMData_;

            stk::mesh::PartVector allPartVector_;

            const double yplusCrit_;
            const double elog_;
            const double kappa_;

            VectorFieldType *coordinates_;
            ScalarFieldType *pressure_;
            VectorFieldType *pressureForce_;
            VectorFieldType *pressureForceSCS_;            
            ScalarFieldType *density_;
            ScalarFieldType *viscosity_;
            GenericFieldType *dudx_;
            VectorFieldType *viscousForce_;
            VectorFieldType *viscousForceSCS_;            
            ScalarFieldType *yplus_;
            GenericFieldType *exposedAreaVec_;
            ScalarFieldType *assembledArea_;
            VectorFieldType *velocity_;
            VectorFieldType *bcVelocity_;
            GenericFieldType *wallFrictionVelocityBip_;
            GenericFieldType *wallNormalDistanceBip_;

            void calc_assembled_area(stk::mesh::PartVector & partvec);

            void calc_surface_force(SurfaceFMData &);

            void calc_surface_force_wallfn(SurfaceFMData &);

            void create_file(std::string fileName);

            void cross_product(double *, double *, double *);

            void parallel_assemble_fields();

            void zero_fields();

        };

    }  // nalu
}  // sierra


#endif /* SURFACEFMPOSTPROCESSING_H */
