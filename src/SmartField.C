// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include<SmartField.h>

namespace sierra::nalu{
using namespace tags;

#define TYPE_INSTANTIATOR_NGP(T) \
  template class SmartField<stk::mesh::NgpField<T>, DEVICE, READ>; \
  template class SmartField<stk::mesh::NgpField<T>, DEVICE, WRITE>; \
  template class SmartField<stk::mesh::NgpField<T>, DEVICE, READ_WRITE>; \
  template class SmartField<stk::mesh::HostField<T>, HOST, READ>; \
  template class SmartField<stk::mesh::HostField<T>, HOST, WRITE>; \
  template class SmartField<stk::mesh::HostField<T>, HOST, READ_WRITE>

TYPE_INSTANTIATOR_NGP(double);
TYPE_INSTANTIATOR_NGP(int);
TYPE_INSTANTIATOR_NGP(unsigned);
}
