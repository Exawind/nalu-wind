#include <gtest/gtest.h>

#include "UnitTestUtils.h"


#include <string>
#include <iostream>
#include <vector>

#include <KokkosInterface.h>
#include "utils/CreateDeviceExpression.h"

class Deviceable {
protected :
  Deviceable *deviceCopy_;
public :
  KOKKOS_FORCEINLINE_FUNCTION Deviceable() : deviceCopy_(nullptr) {} 
  virtual ~Deviceable() {
    if (deviceCopy_) delete_device_copy();
    deviceCopy_ = nullptr;
  }
  template <class T> void copy_to_device(const T &t) {
    deviceCopy_ = sierra::nalu::create_device_expression(t);
  }
  void delete_device_copy() {
    sierra::nalu::kokkos_free_on_device(deviceCopy_);
  }
};

class Shape : public Deviceable {
public :
  KOKKOS_FORCEINLINE_FUNCTION Shape() {} 
  virtual ~Shape() {}
  KOKKOS_FUNCTION
  virtual double area() const = 0;
};

class Rectangle : public Shape {
  const double length_,width_;
public :
  Rectangle(const double l,const double w):Shape(),length_(l),width_(w) {
    copy_to_device(*this);
  }
  KOKKOS_FORCEINLINE_FUNCTION Rectangle(const Rectangle &r):Shape(),length_(r.length_),width_(r.width_) {} 
  virtual ~Rectangle(){}
  KOKKOS_FUNCTION
  virtual double area() const final {
    return length_ * width_;
  }
};

class Circle : public Shape {
  const double radius_;
public :
  Circle(const double radius):Shape(),radius_(radius) {
    copy_to_device(*this);
  }
  KOKKOS_FORCEINLINE_FUNCTION Circle(const Circle &c):Shape(),radius_(c.radius_) {} 
  virtual ~Circle(){}
  KOKKOS_FUNCTION
  virtual double area() const final {
    return 3.14159265 * radius_ * radius_;
  }
};

template<typename T>
double do_shape_test(Shape* s)
{
  Shape* s_dev = sierra::nalu::create_device_expression<T>(
      *dynamic_cast<T*>(s));
  double area = 0.0;
  Kokkos::parallel_reduce(1, KOKKOS_LAMBDA(int, double &a) {
      a = s_dev->area();
      }, area);

  delete s_dev;

  return area;
}


TEST(CreateDeviceExpression, Shapes) 
{
  // Create a couple of virtual classes on host and device.
  Shape *r = new Rectangle(4,5);
  Shape *c = new Circle(2);

  double r_area = do_shape_test<Rectangle>(r);
  double c_area = do_shape_test<Circle>(c);

  EXPECT_EQ(r_area, r->area()) << "Area of a 4x5 Rectangle on device and host";
  EXPECT_EQ(c_area, c->area()) << "Area of a radius 2 Circle on device and host";

  delete r;
  delete c;
}

