/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef BuildTemplates_h
#define BuildTemplates_h

#include "AlgTraits.h"
#include "master_element/Hex8CVFEM.h"
#include "master_element/Hex27CVFEM.h"
#include "master_element/Tet4CVFEM.h"
#include "master_element/Pyr5CVFEM.h"
#include "master_element/Wed6CVFEM.h"
#include "master_element/Quad43DCVFEM.h"
#include "master_element/Quad42DCVFEM.h"
#include "master_element/Quad92DCVFEM.h"
#include "master_element/Quad93DCVFEM.h"
#include "master_element/Tri32DCVFEM.h"
#include "master_element/Edge32DCVFEM.h"
#include "master_element/Edge22DCVFEM.h"
#include "master_element/Tri33DCVFEM.h"

namespace sierra{
namespace nalu{

#define INSTANTIATE_KERNEL_3D(ClassName)                          \
template class ClassName<AlgTraitsHex8>;                          \
template class ClassName<AlgTraitsHex27>;                         \
template class ClassName<AlgTraitsTet4>;                          \
template class ClassName<AlgTraitsPyr5>;                          \
template class ClassName<AlgTraitsWed6>;                          \

#define INSTANTIATE_KERNEL_FACE_3D(ClassName)                     \
template class ClassName<AlgTraitsTri3>;                          \
template class ClassName<AlgTraitsQuad4>;                         \
template class ClassName<AlgTraitsQuad9>;                         \

#define INSTANTIATE_KERNEL_2D(ClassName)                          \
template class ClassName<AlgTraitsQuad4_2D>;                      \
template class ClassName<AlgTraitsQuad9_2D>;                      \
template class ClassName<AlgTraitsTri3_2D>;                       \

#define INSTANTIATE_KERNEL_FACE_2D(ClassName)                     \
template class ClassName<AlgTraitsEdge_2D>;                       \
template class ClassName<AlgTraitsEdge3_2D>;                      \

#define INSTANTIATE_KERNEL_FACE_ELEMENT_3D(ClassName)             \
template class ClassName<AlgTraitsTri3Tet4>;                      \
template class ClassName<AlgTraitsTri3Pyr5>;                      \
template class ClassName<AlgTraitsTri3Wed6>;                      \
template class ClassName<AlgTraitsQuad4Pyr5>;                     \
template class ClassName<AlgTraitsQuad4Wed6>;                     \
template class ClassName<AlgTraitsQuad4Hex8>;                     \
template class ClassName<AlgTraitsQuad9Hex27>;                    \

#define INSTANTIATE_KERNEL_FACE_ELEMENT_2D(ClassName)             \
template class ClassName<AlgTraitsEdge2DTri32D>;                  \
template class ClassName<AlgTraitsEdge2DQuad42D>;                 \
template class ClassName<AlgTraitsEdge32DQuad92D>;                \

// HO templates: generates 4 instantiations per kernel type
// 2,3,4 and one that can be set at compile time

#ifndef USER_POLY_ORDER
#define USER_POLY_ORDER 5
#endif

#define INSTANTIATE_POLY_TEMPLATE(ClassName,BaseTraitsName)       \
template class ClassName<BaseTraitsName<2>>;                      \
template class ClassName<BaseTraitsName<3>>;                      \
template class ClassName<BaseTraitsName<4>>;                      \
template class ClassName<BaseTraitsName<USER_POLY_ORDER>>;        \

#define INSTANTIATE_KERNEL_3D_HO(ClassName)                       \
INSTANTIATE_POLY_TEMPLATE(ClassName,AlgTraitsHexGL)               \

#define INSTANTIATE_KERNEL_2D_HO(ClassName)                       \
INSTANTIATE_POLY_TEMPLATE(ClassName,AlgTraitsQuadGL_2D)           \

#define INSTANTIATE_KERNEL_FACE_2D_HO(ClassName)                  \
INSTANTIATE_POLY_TEMPLATE(ClassName,AlgTraitsEdgeGL)              \

#define INSTANTIATE_KERNEL_FACE_3D_HO(ClassName)                  \
INSTANTIATE_POLY_TEMPLATE(ClassName,AlgTraitsQuadGL)              \

#define INSTANTIATE_KERNEL_FACE_ELEMENT_3D_HO(ClassName)          \
INSTANTIATE_POLY_TEMPLATE(ClassName,AlgTraitsQuadPHexPGL)         \

#define INSTANTIATE_KERNEL_FACE_ELEMENT_2D_HO(ClassName)          \
INSTANTIATE_POLY_TEMPLATE(ClassName,AlgTraitsEdgePQuadPGL)        \

#define INSTANTIATE_KERNEL_3D_HOSGL(ClassName)                    \
template class ClassName<AlgTraitsHex<2>>;                        \
template class ClassName<AlgTraitsHex<3>>;                        \
template class ClassName<AlgTraitsHex<4>>;                        \
template class ClassName<AlgTraitsHex<USER_POLY_ORDER>>;          \

// Instantiate the actual kernels

#define INSTANTIATE_KERNEL(ClassName)                             \
  INSTANTIATE_KERNEL_3D(ClassName)                                \
  INSTANTIATE_KERNEL_2D(ClassName)                                \
  INSTANTIATE_KERNEL_3D_HO(ClassName)                             \
  INSTANTIATE_KERNEL_2D_HO(ClassName)                             \

#define INSTANTIATE_KERNEL_FACE(ClassName)                        \
  INSTANTIATE_KERNEL_FACE_3D(ClassName)                           \
  INSTANTIATE_KERNEL_FACE_2D(ClassName)                           \
  INSTANTIATE_KERNEL_FACE_3D_HO(ClassName)                        \
  INSTANTIATE_KERNEL_FACE_2D_HO(ClassName)                        \

#define INSTANTIATE_KERNEL_FACE_ELEMENT(ClassName)                \
  INSTANTIATE_KERNEL_FACE_ELEMENT_3D(ClassName)                   \
  INSTANTIATE_KERNEL_FACE_ELEMENT_2D(ClassName)                   \
  INSTANTIATE_KERNEL_FACE_ELEMENT_3D_HO(ClassName)                \
  INSTANTIATE_KERNEL_FACE_ELEMENT_2D_HO(ClassName)                \

#define INSTANTIATE_KERNEL_HOSGL(ClassName)                       \
  INSTANTIATE_KERNEL_3D_HOSGL(ClassName)                          \



} // namespace nalu
} // namespace Sierra


#endif
