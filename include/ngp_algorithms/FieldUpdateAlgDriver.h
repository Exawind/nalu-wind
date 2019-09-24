/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef FIELDUPDATEALGDRIVER_H
#define FIELDUPDATEALGDRIVER_H

#include "ngp_algorithms/NgpAlgDriver.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class FieldUpdateAlgDriver : public NgpAlgDriver
{

public:
  FieldUpdateAlgDriver(Realm&, const std::string&);

  virtual ~FieldUpdateAlgDriver() = default;

  //! Reset fields before calling algorithms
  virtual void pre_work() override;

  //! Synchronize fields after algorithms have done their work
  virtual void post_work() override;

private:
  //! Field that is synchronized pre/post updates
  const std::string fieldName_;
};

} // namespace nalu
} // namespace sierra

#endif /* FIELDUPDATEALGDRIVER_H */
