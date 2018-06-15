#include <gtest/gtest.h>

#include "UnitTestUtils.h"


#include <string>
#include <iostream>
#include <vector>

#include <KokkosInterface.h>
#include "utils/CreateDeviceExpression.h"

class deviceable {
protected :
  deviceable *DeviceCopy;
public :
  KOKKOS_FORCEINLINE_FUNCTION deviceable() : DeviceCopy(nullptr) {} 
  virtual ~deviceable() {
    if (DeviceCopy) delete_device_copy();
    DeviceCopy = nullptr;
  }
  template <class T> void copy_to_device(const T &t) {
    DeviceCopy = sierra::nalu::create_device_expression(t);
  }
  void delete_device_copy() {
    sierra::nalu::kokkos_free_on_device(DeviceCopy);
  }
  template<class T> T* device_copy() const {return dynamic_cast<T*>(DeviceCopy);}
};

class shape : public deviceable {
public :
  KOKKOS_FORCEINLINE_FUNCTION shape() {} 
  virtual ~shape() {}
  virtual double area() const = 0;
};

class rectangle : public shape {
  const double L,W;
public :
  KOKKOS_FORCEINLINE_FUNCTION rectangle(const double l,const double w):shape(),L(l),W(w) {
    copy_to_device(*this);
  }
  KOKKOS_FORCEINLINE_FUNCTION rectangle(const rectangle &r):shape(),L(r.L),W(r.W) {} 
  virtual ~rectangle(){}
  virtual double area() const final {
    return L*W;
  }
};

class circle : public shape {
  const double R;
public :
  KOKKOS_FORCEINLINE_FUNCTION circle(const double r):shape(),R(r) {
    copy_to_device(*this);
  }
  KOKKOS_FORCEINLINE_FUNCTION circle(const circle &c):shape(),R(c.R) {} 
  virtual ~circle(){}
  virtual double area() const final {
    return 2*3.14159265*R;
  }
};


TEST(CreateDeviceExpression, shapes) 
{
// Create a couple of virtual classes on host and device.
shape *r = new rectangle(4,5);
shape *c = new circle(2);
shape *r_dev = r->device_copy<shape>();
shape *c_dev = c->device_copy<shape>();

double r_area;
auto r_on_device = [&] (int i, double &a) {
  a = r_dev->area();
}; 
sierra::nalu::kokkos_parallel_reduce(1, r_on_device, r_area, "Call Rectangle on Device.");

double c_area;
auto c_on_device = [&] (int i, double &a) {
  a = c_dev->area();
}; 
sierra::nalu::kokkos_parallel_reduce(1, c_on_device, c_area, "Call Circle on Device.");

EXPECT_EQ(r_area, r->area()) << "Area of a 4x5 rectangle on device and host";
EXPECT_EQ(c_area, c->area()) << "Area of a radius 2 circle on device and host";

delete r;
delete c;
}

