// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TIOGAREF_H
#define TIOGAREF_H

#include <memory>

namespace TIOGA {
class tioga;
}

namespace tioga_nalu {

/** Manager for the TIOGA handle
 *
 *  This class manages access to the global TIOGA instance across multiple
 *  realms as well as across multi-solver interfaces (e.g., amr-wind/nalu-wind
 *  coupling). This class allows both use-cases: 1. nalu-wind only run where
 *  overset might be coupling within just one realm or multiple realms but owns
 *  the TIOGA reference, 2. hybrid solver run where an external code interface
 *  owns the TIOGA reference.
 *
 */
class TiogaRef
{
public:
  /** Access the reference object
   *
   *  The first invocation can pass a pointer to a valid TIOGA instance which
   *  makes this a non-owning reference holder. Otherwise it will create a new
   *  instance and take ownership of that reference.
   *
   *  \param tg Pointer to an existing TIOGA instance
   */
  static TiogaRef& self(TIOGA::tioga* tg = nullptr);

  ~TiogaRef();

  TiogaRef(const TiogaRef&) = delete;
  TiogaRef& operator=(const TiogaRef&) = delete;

  inline operator TIOGA::tioga&()
  {
    return *tg_;
  }

  //! Access the underlying TIOGA reference
  inline TIOGA::tioga& get()
  {
    return *tg_;
  }

private:
  TiogaRef();

  TiogaRef(TIOGA::tioga* tg);

  TIOGA::tioga* tg_{nullptr};

  bool owned_{false};
};

}

#endif /* TIOGAREF_H */
