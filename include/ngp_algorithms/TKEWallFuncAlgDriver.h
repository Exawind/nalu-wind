/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TKEWALLFUNCALGDRIVER_H
#define TKEWALLFUNCALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

/** Compute nodal TKE values when using ABL wall function
 *
 *  Use a driver/algorithm style design to allow the actual computation to be
 *  templated on the face element type. Computation of the "wall_model_tke_bc"
 *  field is performed in TKEWallFuncAlg class. This field is synchronized in
 *  parallel as well as host/device in TKEWallFuncAlgDriver::post_work and
 *  "turbulent_ke" and "tke_bc" fields are updated for the wall boundaries where
 *  wall function is being used.
 *
 *  \sa TKEWallFuncAlg
 */
class TKEWallFuncAlgDriver : public NgpAlgDriver
{
public:
  TKEWallFuncAlgDriver(Realm&);

  virtual ~TKEWallFuncAlgDriver() = default;

  //! Reset fields before calling algorithms
  virtual void pre_work() override;

  //! Synchronize fields after algorithms have done their work
  virtual void post_work() override;

private:
  unsigned tke_ {stk::mesh::InvalidOrdinal};
  unsigned bctke_ {stk::mesh::InvalidOrdinal};
  unsigned bcNodalTke_ {stk::mesh::InvalidOrdinal};
  unsigned wallArea_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* TKEWALLFUNCALGDRIVER_H */
