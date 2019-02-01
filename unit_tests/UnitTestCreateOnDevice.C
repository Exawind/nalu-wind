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
  template<class T> T* device_copy() const {return dynamic_cast<T*>(deviceCopy_);}
};

class Shape : public Deviceable {
public :
  KOKKOS_FORCEINLINE_FUNCTION Shape() {} 
  virtual ~Shape() {}
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
  virtual double area() const final {
    return 3.14159265 * radius_ * radius_;
  }
};


TEST(CreateDeviceExpression, Shapes) 
{
  // Create a couple of virtual classes on host and device.
  Shape *r = new Rectangle(4,5);
  Shape *c = new Circle(2);
  Shape *r_dev = r->device_copy<Shape>();
  Shape *c_dev = c->device_copy<Shape>();

  double r_area;
  auto r_on_device = [&] (int  /* i */, double &a) {
    a = r_dev->area();
  };
  sierra::nalu::kokkos_parallel_reduce(1, r_on_device, r_area, "Call Rectangle on Device.");

  double c_area;
  auto c_on_device = [&] (int  /* i */, double &a) {
    a = c_dev->area();
  };
  sierra::nalu::kokkos_parallel_reduce(1, c_on_device, c_area, "Call Circle on Device.");

  EXPECT_EQ(r_area, r->area()) << "Area of a 4x5 Rectangle on device and host";
  EXPECT_EQ(c_area, c->area()) << "Area of a radius 2 Circle on device and host";

  delete r;
  delete c;
}

