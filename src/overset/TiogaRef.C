// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef NALU_USES_TIOGA

#include "overset/TiogaRef.h"

#include "tioga.h"

#include <iostream>

namespace tioga_nalu {

TiogaRef& TiogaRef::self(TIOGA::tioga* tg)
{
  static bool initialized{false};
  static std::unique_ptr<TiogaRef> tgref;

  if (initialized) {
    if (tg != nullptr)
      throw std::runtime_error("Multiple registration of TIOGA object encountered");
  } else {
    if (tg == nullptr) {
      tgref.reset(new TiogaRef());
    } else {
      tgref.reset(new TiogaRef(tg));
    }
    initialized = true;
  }

  return *tgref;
}

TiogaRef::TiogaRef()
  : tg_(new TIOGA::tioga())
  , owned_(true)
{}

TiogaRef::TiogaRef(TIOGA::tioga* tg)
  : tg_(tg)
  , owned_(false)
{}

TiogaRef::~TiogaRef()
{
  if (owned_ && (tg_ != nullptr)) {
    delete tg_;
    tg_ = nullptr;
  }
}

}

#endif
