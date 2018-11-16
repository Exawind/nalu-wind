/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ASSEMBLEWALLDISTNONCONFORMALALGORITHM_H
#define ASSEMBLEWALLDISTNONCONFORMALALGORITHM_H

#include "SolverAlgorithm.h"
#include "FieldTypeDef.h"

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra {
namespace nalu {

class Realm;

/** Contributions to the WallDistance linear system from the non-conformal
 * interfaces
 *
 */
class AssembleWallDistNonConformalAlgorithm : public SolverAlgorithm
{
public:
  AssembleWallDistNonConformalAlgorithm(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*);

  virtual ~AssembleWallDistNonConformalAlgorithm() = default;

  virtual void initialize_connectivity();

  virtual void execute();

private:
  AssembleWallDistNonConformalAlgorithm() = delete;
  AssembleWallDistNonConformalAlgorithm(
    const AssembleWallDistNonConformalAlgorithm&) = delete;

  //! Reference to the coordinates field
  VectorFieldType* coordinates_;

  //! Reference to the exposed area vector on the non-conformal interface
  GenericFieldType* exposedAreaVec_;

  //! Fields that must be ghosted for use with this algorithm
  std::vector<const stk::mesh::FieldBase*> ghostFieldVec_;

  //! Flag indicating we should use current normal for computing inverse length
  const bool useCurrentNormal_;
};

}  // nalu
}  // sierra



#endif /* ASSEMBLEWALLDISTNONCONFORMALALGORITHM_H */
