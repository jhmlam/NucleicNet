// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.

// $Id: Point3D.cc 914 2011-05-10 17:56:41Z mikewong899 $

#include "Point3D.h"
#define sqr(x) ((x) * (x))

/**
 * @brief default constructor set x,y,z to 0
 **/
Point3D::Point3D() : x(0), y(0), z(0) {
}

/**
 * @brief constructor with parameters, set x,y,z to the corresponding values
 **/
Point3D::Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {
}

/**
 * @brief constructor passing a Point3D, set x,y,z to the corresponding values
 **/
Point3D::Point3D(Point3D const &source) : x(source.x), y(source.y), z(source.z) {
}

/**
 * @brief method to get the distance between this object and another Point3D
 * @return  the distance between this element and the SPoint 
 **/
double Point3D::getDistance(Point3D const *coord) const {
    return (sqrt(getDistanceSqr(*coord)));
}

/**
 * @brief method to get the distance between this object and another Point3D
 * @return  the distance between this element and the SPoint 
 **/
double Point3D::getDistance(Point3D const &coord) const {
    return (sqrt(getDistanceSqr(coord)));
}

/**
 * @brief method to get the distance between this object and another Point3D
 * @return  the distance between this element and the SPoint 
 **/
double Point3D::getDistanceSqr(Point3D const &coord) const {
    double xdiff = coord.x - x;
    double ydiff = coord.y - y;
    double zdiff = coord.z - z;
    double sum = sqr(xdiff) + sqr(ydiff) + sqr(zdiff);
    return (sum);
}

/**
 * @brief operator = overloded to use with Point3D, set x,y,z to the xyz of the element passed
 **/
Point3D &Point3D::operator=(Point3D const &rhs) {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    return (*this);
}

/**
 * @brief operator += overloded to use with Point3D, updates the values of the calling object
 **/
Point3D &Point3D::operator+=(Point3D const &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return (*this);
}

/**
 * @brief operator -= overloded to use with Point3D, updates the values of the calling object
 **/
Point3D &Point3D::operator-=(Point3D const &rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return (*this);
}

/**
 * @brief operator *= overloded to use with Point3D, updates the values of the calling object
 **/
Point3D &Point3D::operator*=(double rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    return (*this);
}

/**
 * @brief operator /= overloded to use with Point3D, updates the values of the calling object
 **/
Point3D &Point3D::operator/=(double rhs) {
    x /= rhs;
    y /= rhs;
    z /= rhs;
    return (*this);
}

/**
 * @brief operator + overloded to use with Point3D, updates the values of the calling object
 **/
Point3D Point3D::operator+(Point3D const &rhs) const {
    return Point3D(x + rhs.x, y + rhs.y, z + rhs.z);
}

/**
 * @brief operator - overloded to use with Point3D, updates the values of the calling object
 **/
Point3D Point3D::operator-(Point3D const &rhs) const {
    return Point3D(x - rhs.x, y - rhs.y, z - rhs.z);
}

/**
 * @brief operator * overloded to use with Point3D, updates the values of the calling object
 **/
Point3D Point3D::operator*(double rhs) const {
    return Point3D(x*rhs, y*rhs, z * rhs);
}

/**
 * @brief operator / overloded to use with Point3D, updates the values of the calling object
 **/
Point3D Point3D::operator/(double rhs) const {
    return Point3D(x / rhs, y / rhs, z / rhs);
}

/**
 * @brief accessor to the values x,y,z
 **/
void Point3D::getValues(double *x, double *y, double *z) const {
    *x = this->x;
    *y = this->y;
    *z = this->z;
}

/**
 * @brief accessor to  x
 **/
double Point3D::getx()
{
	return x;
}

/**
 * @brief accessor to y
 **/
double Point3D::gety()
{
	return y;
}

/**
 * @brief accessor to z
 **/
double Point3D::getz()
{
	return z;
}

/**
 * @brief mutator to  x
 **/
void Point3D::setx(double x)
{
	 this->x = x;
}

/**
 * @brief mutator for y
 **/
void Point3D::sety(double y)
{
	this->y = y;
}

/**
 * @brief mutator for z
 **/
void Point3D::setz(double z)
{
	this->z = z;
}
