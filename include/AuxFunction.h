/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AuxFunction_h
#define AuxFunction_h

namespace sierra{
namespace nalu{

class AuxFunction
{
public:
  AuxFunction(const unsigned beginPos, const unsigned endPos)
    : beginPos_(beginPos),
      endPos_(endPos) {}
  virtual ~AuxFunction() {}

  // coords:
  //    coordinates at each point, (x,y) for 2d, (x,y,z) for 3d
  // time:
  //    time value
  // spatial_dimension:
  //    1 for 1d, 2 for 2d, 3 for 3d
  // num_points:
  //    number of points to evaluate (e.g., length of coords should be 3*num_points for 3d)
  // field:
  //    where to write the values
  // field_dimension:
  //    number of field values at each point (scalar=1, vector=spatial_dimension)
  // begin_pos:
  // end_pos:
  //    to only write a subset of the field values, specify these values
  //
  //    For example, the set the 2nd and 3rd components of a 3d vector, use:
  //        spatial_dimension=3
  //        field_dimension=3
  //        begin_pos=1
  //        end_pos=3

  void evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize) const
  {
    if(beginPos_ == 0 && endPos_ == fieldSize)
      do_evaluate(coords, time, spatialDimension, numPoints, fieldPtr, fieldSize);
    else
      do_evaluate(coords, time, spatialDimension, numPoints, fieldPtr, fieldSize, beginPos_, endPos_);
  }
  virtual void setup(const double /* time */) {}

protected:

  // Derived classes must at_least implement this method
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const = 0;

  // Derived classes may override this form for efficiency (all field values being assigned)
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize) const
  {
    do_evaluate(coords, time, spatialDimension, numPoints, fieldPtr, fieldSize, 0, fieldSize);
  }

  const unsigned beginPos_;
  const unsigned endPos_;
};

} // namespace nalu
} // namespace Sierra

#endif
