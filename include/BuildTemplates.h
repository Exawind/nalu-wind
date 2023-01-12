// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef BuildTemplates_h
#define BuildTemplates_h

#include "AlgTraits.h"
#include "master_element/Hex8CVFEM.h"
#include "master_element/Tet4CVFEM.h"
#include "master_element/Pyr5CVFEM.h"
#include "master_element/Wed6CVFEM.h"
#include "master_element/Quad43DCVFEM.h"
#include "master_element/Quad42DCVFEM.h"
#include "master_element/Tri32DCVFEM.h"
#include "master_element/Edge22DCVFEM.h"
#include "master_element/Tri33DCVFEM.h"

namespace sierra {
namespace nalu {

#define INSTANTIATE_KERNEL_3D(ClassName)                                       \
  template class ClassName<AlgTraitsHex8>;                                     \
  template class ClassName<AlgTraitsTet4>;                                     \
  template class ClassName<AlgTraitsPyr5>;                                     \
  template class ClassName<AlgTraitsWed6>;

#define INSTANTIATE_KERNEL_FACE_3D(ClassName)                                  \
  template class ClassName<AlgTraitsTri3>;                                     \
  template class ClassName<AlgTraitsQuad4>;

#define INSTANTIATE_KERNEL_2D(ClassName)                                       \
  template class ClassName<AlgTraitsQuad4_2D>;                                 \
  template class ClassName<AlgTraitsTri3_2D>;

#define INSTANTIATE_KERNEL_FACE_2D(ClassName)                                  \
  template class ClassName<AlgTraitsEdge_2D>;

#define INSTANTIATE_KERNEL_FACE_ELEMENT_3D(ClassName)                          \
  template class ClassName<AlgTraitsTri3Tet4>;                                 \
  template class ClassName<AlgTraitsTri3Pyr5>;                                 \
  template class ClassName<AlgTraitsTri3Wed6>;                                 \
  template class ClassName<AlgTraitsQuad4Pyr5>;                                \
  template class ClassName<AlgTraitsQuad4Wed6>;                                \
  template class ClassName<AlgTraitsQuad4Hex8>;

#define INSTANTIATE_KERNEL_FACE_ELEMENT_2D(ClassName)                          \
  template class ClassName<AlgTraitsEdge2DTri32D>;                             \
  template class ClassName<AlgTraitsEdge2DQuad42D>;
// HO templates: generates 4 instantiations per kernel type
// 2,3,4 and one that can be set at compile time

#ifndef USER_POLY_ORDER
#define USER_POLY_ORDER 1
#endif

#define INSTANTIATE_POLY_TEMPLATE(ClassName)                                   \
  template class ClassName<2>;                                                 \
  template class ClassName<3>;                                                 \
  template class ClassName<4>;                                                 \
  template class ClassName<BaseTraitsName<USER_POLY_ORDER>>;

// Instantiate the actual kernels

#define INSTANTIATE_KERNEL(ClassName)                                          \
  INSTANTIATE_KERNEL_3D(ClassName)                                             \
  INSTANTIATE_KERNEL_2D(ClassName)

#define INSTANTIATE_KERNEL_FACE(ClassName)                                     \
  INSTANTIATE_KERNEL_FACE_3D(ClassName)                                        \
  INSTANTIATE_KERNEL_FACE_2D(ClassName)

#define INSTANTIATE_KERNEL_FACE_ELEMENT(ClassName)                             \
  INSTANTIATE_KERNEL_FACE_ELEMENT_3D(ClassName)                                \
  INSTANTIATE_KERNEL_FACE_ELEMENT_2D(ClassName)

} // namespace nalu
} // namespace sierra

#endif
