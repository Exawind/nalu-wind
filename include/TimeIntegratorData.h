#ifndef TIMEINTEGRATORDATA_H
#define TIMEINTEGRATORDATA_H

namespace sierra {
namespace nalu {

struct TimeIntegratorData
{
  double timeStepN_;
  double timeStepNm1_;
  double gamma1_;
  double gamma2_;
  double gamma3_;
};

} // namespace nalu
} // namespace sierra

#endif /* TIMEINTEGRATORDATA_H */
