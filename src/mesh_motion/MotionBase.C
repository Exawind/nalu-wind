
#include "mesh_motion/MotionBase.h"

namespace sierra{
namespace nalu{

const MotionBase::transMatType MotionBase::identityMat_
  = {{{1,0,0,0},
      {0,1,0,0},
      {0,0,1,0},
      {0,0,0,1}}};

MotionBase::transMatType MotionBase::add_motion(
    const transMatType& motionL,
    const transMatType& motionR)
{
  transMatType comp_trans_mat_ = {};

  for (int r = 0; r < transMatSize; r++) {
    for (int c = 0; c < transMatSize; c++) {
      for (int k = 0; k < transMatSize; k++) {
        comp_trans_mat_[r][c] += motionL[r][k] * motionR[k][c];
      } // end for loop - k index
    } // end for loop - column index
  } // end for loop - row index

  return comp_trans_mat_;
}

} // nalu
} // sierra
