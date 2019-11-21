// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ABLProfileFunction_h
#define ABLProfileFunction_h

namespace sierra{
namespace nalu{

class ABLProfileFunction
{
 public:

  ABLProfileFunction();
  virtual ~ABLProfileFunction();
  virtual double velocity(const double znorm) const = 0;
  virtual double temperature(const double znorm) const = 0;
};

class StableABLProfileFunction : public ABLProfileFunction
{
 public:
  StableABLProfileFunction(double gamma_m, double gamma_h);
  virtual ~StableABLProfileFunction();
  double velocity(const double znorm) const;
  double temperature(const double znorm) const;

 private:
  double gamma_m_;
  double gamma_h_;
};

class UnstableABLProfileFunction : public ABLProfileFunction
{
 public:
  UnstableABLProfileFunction(double beta_m, double beta_h);
  virtual ~UnstableABLProfileFunction();
  double velocity(const double znorm) const;
  double temperature(const double znorm) const;

 private:
  double beta_m_;
  double beta_h_;
  double pi_;
};

class NeutralABLProfileFunction : public ABLProfileFunction
{
 public:
  NeutralABLProfileFunction();
  virtual ~NeutralABLProfileFunction();
  double velocity(const double znorm) const;
  double temperature(const double znorm) const;
};

} // namespace nalu
} // namespace Sierra

#endif
