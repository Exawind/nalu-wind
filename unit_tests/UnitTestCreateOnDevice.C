#include <gtest/gtest.h>

#include "UnitTestUtils.h"

#include <string>
#include <iostream>
#include <vector>

#include <KokkosInterface.h>
#include "utils/CreateDeviceExpression.h"

class Shape
{
public:
  KOKKOS_FORCEINLINE_FUNCTION Shape() {}
  KOKKOS_DEFAULTED_FUNCTION virtual ~Shape() = default;
  KOKKOS_FUNCTION
  virtual double area() const = 0;
};

class Rectangle : public Shape
{
  const double length_, width_;

public:
  Rectangle(const double l, const double w) : Shape(), length_(l), width_(w) {}
  KOKKOS_FORCEINLINE_FUNCTION Rectangle(const Rectangle& r)
    : Shape(), length_(r.length_), width_(r.width_)
  {
  }
  KOKKOS_DEFAULTED_FUNCTION virtual ~Rectangle() = default;
  KOKKOS_FUNCTION
  virtual double area() const final { return length_ * width_; }
};

class Circle : public Shape
{
  const double radius_;

public:
  Circle(const double radius) : Shape(), radius_(radius) {}
  KOKKOS_FORCEINLINE_FUNCTION Circle(const Circle& c)
    : Shape(), radius_(c.radius_)
  {
  }
  KOKKOS_DEFAULTED_FUNCTION virtual ~Circle() = default;
  KOKKOS_FUNCTION
  virtual double area() const final { return 3.14159265 * radius_ * radius_; }
};

template <typename T>
double
do_shape_test(Shape* s)
{
  Shape* s_dev =
    sierra::nalu::create_device_expression<T>(*dynamic_cast<T*>(s));
  double area = 0.0;
  Kokkos::parallel_reduce(
    sierra::nalu::DeviceRangePolicy(0, 1),
    KOKKOS_LAMBDA(int, double& a) { a = s_dev->area(); }, area);

  sierra::nalu::kokkos_free_on_device(s_dev);
  return area;
}

TEST(CreateDeviceExpression, Shapes)
{
  // Create a couple of virtual classes on host and device.
  Shape* r = new Rectangle(4, 5);
  Shape* c = new Circle(2);

  double r_area = do_shape_test<Rectangle>(r);
  double c_area = do_shape_test<Circle>(c);

  EXPECT_EQ(r_area, r->area()) << "Area of a 4x5 Rectangle on device and host";
  EXPECT_EQ(c_area, c->area())
    << "Area of a radius 2 Circle on device and host";

  delete r;
  delete c;
}
