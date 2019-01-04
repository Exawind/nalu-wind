/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
/*
 * UnitTestActuatorDiskFAST.C
 *
 *  Created on: Jan 11, 2019
 *      Author: psakiev
 */

#include <gtest/gtest.h>
#include <ActuatorDiskFAST.h>
#include <algorithm>
#include <functional>
#include <random>
#include <sstream>

namespace sierra{
namespace nalu{

TEST(ActuatorDiskFAST, SweptPointLocation){
  const double PI {std::acos(-1.0)};
  std::vector<double> hub={1,1,1};
  Point point={1,1,2};
  std::vector<double> axis={0,1,0};
  const int nPoints = 3;
  for (int nBlades=1; nBlades<4; nBlades++){
    const double dTheta = 2.0*PI/nBlades/(nPoints+1);

    for (int j=0; j<nPoints; j++){

      double expectTheta = dTheta*(j+1);
      if(expectTheta>PI){
        expectTheta = 2.0*PI-expectTheta;
      }

      Point result=SweptPointLocation(nBlades,3,j,point,hub,axis);

      double radius{0};
      double dot{0};
      for (int i=0; i<3; i++){
        radius += (result[i]-hub[i])*(result[i]-hub[i]);
        dot+=(result[i]-hub[i])*(point[i]-hub[i]);
      }

      // Ensure that angle is incremented by dTheta and radius stays constant
      // during mapping process.
      double resultTheta = std::acos(dot/radius);

      EXPECT_DOUBLE_EQ(expectTheta, resultTheta) << "Blades: " << nBlades << " Point Number: " << j;
      EXPECT_DOUBLE_EQ(1.0, std::sqrt(radius));
    }
  }
}
template<class T>
double F_distance(T& p1, T& p2){
  double d = 0;
  for (int i=0; i<3; i++){
    d+= std::pow(p1[i]-p2[i],2.0);
  }
  return std::sqrt(d);
}

Point F_center(Point& A, Point& B, Point& C){
  Point center = {
      (A[0]+B[0]+C[0])/3.0,
      (A[1]+B[1]+C[1])/3.0,
      (A[2]+B[2]+C[2])/3.0
  };
  return center;
}

std::vector<double> F_RotateAboutAxis(int axis, double angle, const std::vector<double>& p, const std::vector<double>& h){
  std::vector<double> pPrime(3);
  const double dCos{std::cos(angle)}, dSin{std::sin(angle)};
  int j = (axis+1)%3;
  int k = (j+1)%3;
  pPrime[axis] = p[axis];
  pPrime[j] = dCos*(p[j]-h[j])-dSin*(p[k]-h[k])+h[j];
  pPrime[k] = dCos*(p[k]-h[k])+dSin*(p[j]-h[j])+h[k];
  return pPrime;
}

TEST(ActuatorDiskFAST, SweptPointLocatorBasis){
  SweptPointLocator alpha;
  EXPECT_DOUBLE_EQ(2.0/3.0, alpha.periodic_basis(0.0));
  EXPECT_DOUBLE_EQ(2.0/3.0*std::pow(std::cos(0.5),2.0), alpha.periodic_basis(1.0));
}

TEST(ActuatorDiskFAST, PointsOnACircle){
  const double PI = std::acos(-1.0);
  const std::vector<double> origin(3,0.0);
  std::vector<double> hub(3,0.0);
  std::mt19937::result_type seed = std::time(0);
  auto fn_np_real_rand = std::bind(std::uniform_real_distribution<double>(-1.0,1.0),std::mt19937(seed));
  auto fn_int_rand = std::bind(std::uniform_int_distribution<int>(0,2),std::mt19937(seed));

  // check rotation function
  {
    std::vector<double> a(3),b(3);
    a={1,0,0}; b={4,1,0};
    auto c = F_RotateAboutAxis(2,0.5*PI,a,origin);
    ASSERT_NEAR(0.0,c[0],1e-12);
    ASSERT_NEAR(1.0,c[1],1e-12);
    ASSERT_NEAR(0.0,c[2],1e-12);

    a[1]=1.0;
    c = F_RotateAboutAxis(2,0.5*PI,b,a);
    ASSERT_NEAR(1.0,c[0],1e-12);
    ASSERT_NEAR(4.0,c[1],1e-12);
    ASSERT_NEAR(0.0,c[2],1e-12);
  }

  int i=0;
  while (i < 20)
  {
    std::ostringstream message;
    for(int i =0; i<3; i++){
      hub[i] = fn_np_real_rand();
    }

    const int index = fn_int_rand();

    // three random points with unique directions
    // with respect to the hub
    std::vector<double> p1(3),p2(3),p3(3);
    p1={fn_np_real_rand(),fn_np_real_rand(),fn_np_real_rand()};
    p2=F_RotateAboutAxis(index,2.0*PI/3.0, p1,hub);
    p3=F_RotateAboutAxis(index,4.0*PI/3.0, p1,hub);


    // All the points are equidistance from the hub
    Point A{p1[0], p1[1], p1[2]},
          B{p2[0], p2[1], p2[2]},
          C{p3[0], p3[1], p3[2]};

    SweptPointLocator locator;
    locator.update_point_location(0,A);
    locator.update_point_location(1,B);
    locator.update_point_location(2,C);

    Point center = F_center(A,B,C);
    const double radius = F_distance<Point>(A,center);

    auto contPnts = locator.get_control_points();

    // just in case it fails...
    message << "Failure for points: " << std::endl
        <<"A: "<<A[0] <<", " <<A[1] << ", " << A[2] <<std::endl
        <<"B: "<<B[0] <<", " <<B[1] << ", " << B[2] <<std::endl
        <<"C: "<<C[0] <<", " <<C[1] << ", " << C[2] <<std::endl
        <<"Hub: "<<hub[0] <<", " <<hub[1] << ", " << hub[2] <<std::endl
        <<"Center: "<<center[0] <<", " <<center[1] << ", " << center[2] <<std::endl
        <<"Control Point 0: "<<contPnts[0][0] <<", " <<contPnts[0][1] << ", " << contPnts[0][2] <<std::endl
        <<"Control Point 1: "<<contPnts[1][0] <<", " <<contPnts[1][1] << ", " << contPnts[1][2] <<std::endl
        <<"Control Point 2: "<<contPnts[2][0] <<", " <<contPnts[2][1] << ", " << contPnts[2][2] <<std::endl
        <<"Rotation Axis: " << index <<std::endl;

    int match1{(index+1)%3}, match2{(index+2)%3};
    ASSERT_NEAR(hub[match1],center[match1],1e-12) << message.str();
    ASSERT_NEAR(hub[match2],center[match2],1e-12) << message.str();

    double t = fn_np_real_rand()*PI;
    Point Ap = locator(t);

    // is point on circle that intersected all three points
    EXPECT_NEAR(0.0,std::fabs(F_distance<Point>(Ap,center)-radius),1e-12) << message.str()
        << "Failure at t=: "<< t << " was calculated as " <<Ap[0] << ", "<<Ap[1] <<", " <<Ap[2]<<std::endl;
    i++;
  }


}

TEST(ActuatorDiskFAST, FindClosestIndex){
  std::mt19937::result_type seed = std::time(0);
  auto fn_real_rand = std::bind(std::uniform_real_distribution<double>(0,1),std::mt19937(seed));

  for (int i=0; i<20; i++){
    // create a vec of random values between 0 and 1
    std::vector<double> listOfRadius(20);
    for(auto&& value : listOfRadius){
      value = fn_real_rand();
    }

    // sort and remove duplicates
    std::sort(listOfRadius.begin(), listOfRadius.end());
    auto it = std::unique(listOfRadius.begin(), listOfRadius.end());

    if(it!=listOfRadius.end()){
      listOfRadius.resize(std::distance(listOfRadius.begin(),it));
    }

    // generate a random radial value
    double radius = fn_real_rand();

    // find closest
    int closest = FindClosestIndex(radius, listOfRadius);

    // check
    bool isClosest = true;
    std::ostringstream fail_test_message;
    fail_test_message << "Failure of iteration " << i+1 << " of 20" << std::endl;
    fail_test_message << "Vector: " << std::endl;
    for (auto&& v : listOfRadius){
      fail_test_message << std::to_string(v) << ", ";
    }
    fail_test_message << std::endl;
    if(closest<listOfRadius.size()-1){
      if(std::fabs(listOfRadius[closest]-radius) > std::fabs(listOfRadius[closest+1]-radius) ){
        isClosest = false;
        fail_test_message
          << " Radius value: "
          << std::to_string(radius)
          << std::endl
          << " Function Match (delta): " << listOfRadius[closest]
          << " (" <<std::fabs(listOfRadius[closest]-radius) << ")"
          << std::endl
          << " Closer Value (delta): " << listOfRadius[closest+1]
          << " (" <<std::fabs(listOfRadius[closest+1]-radius) << ")"
          << std::endl
          << " Function and Closer Indices: " << closest << " " << closest+1
          << std::endl;
      }
    }
    if(closest>0){
      if(std::fabs(listOfRadius[closest]-radius) > std::fabs(listOfRadius[closest-1]-radius)){
        isClosest = false;
        fail_test_message
          << " Radius value: "
          << std::to_string(radius)
          << std::endl
          << " Function Match (delta): " << listOfRadius[closest]
          << " (" <<std::fabs(listOfRadius[closest]-radius) << ")"
          << std::endl
          << " Closer Value (delta): " << listOfRadius[closest-1]
          << " (" <<std::fabs(listOfRadius[closest-1]-radius) << ")"
          << std::endl
          << " Function and Closer Indices: " << closest << " " << closest-1
          << std::endl;
      }
    }
    EXPECT_TRUE(isClosest) << fail_test_message.str();
  }

}

TEST(ActuatorDiskFAST, NormalizedDirection){
  std::vector<double> p1(3),p2(3),n(3);
  p1[0]=0; p1[1]=0; p1[2]=0;
  p2[0]=1; p2[1]=1; p2[2]=1;
  double s3 = 1.0/std::sqrt(3.0);

  n = NormalizedDirection(p2,p1);
  EXPECT_DOUBLE_EQ(s3,n[0]);
  EXPECT_DOUBLE_EQ(s3,n[1]);
  EXPECT_DOUBLE_EQ(s3,n[2]);

  p2[1]*=-1.0;
  n = NormalizedDirection(p2,p1);
  EXPECT_DOUBLE_EQ(s3,n[0]);
  EXPECT_DOUBLE_EQ(-s3,n[1]);
  EXPECT_DOUBLE_EQ(s3,n[2]);

  p2[0]*=2.0; p2[1]*=-2.0; p2[2]*=2.0;
  n = NormalizedDirection(p2,p1);
  EXPECT_DOUBLE_EQ(s3,n[0]);
  EXPECT_DOUBLE_EQ(s3,n[1]);
  EXPECT_DOUBLE_EQ(s3,n[2]);

  p2[0]+=2.0; p2[1]+=2.0; p2[2]+=2.0;
  p1[0]+=2.0; p1[1]+=2.0; p1[2]+=2.0;
  n = NormalizedDirection(p2,p1);
  EXPECT_DOUBLE_EQ(s3,n[0]);
  EXPECT_DOUBLE_EQ(s3,n[1]);
  EXPECT_DOUBLE_EQ(s3,n[2]);


}

}
}
