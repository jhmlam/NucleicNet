// FEATURE
// See COPYING and AUTHORS for copyright and license agreement terms.
// $Id: Point3D.h 881 2011-04-06 19:54:53Z mikewong899 $

#ifndef POINT3D_H
#define POINT3D_H

#include <cmath>

/**
 * @brief 3D Cartesian coordinates
 * @ingroup euclidean_module
 **/
class Point3D {
	public:
		Point3D();
		Point3D(double _x, double _y, double _z);
		Point3D(Point3D const &source);

		double getx();
		double gety();
		double getz();
		void   setx(double x);
		void   sety(double y);
		void   setz(double z);

		Point3D & operator=(Point3D const &rhs);
		Point3D & operator+=(Point3D const &rhs);
		Point3D & operator-=(Point3D const &rhs);
		Point3D & operator*=(double rhs);
		Point3D & operator/=(double rhs);
		Point3D   operator+(Point3D const &rhs) const;
		Point3D   operator-(Point3D const &rhs) const;
		Point3D   operator*(double rhs) const;
		Point3D   operator/(double rhs) const;

		double getDistance(Point3D const *coord) const;
		double getDistance(Point3D const &coord) const;
		double getDistanceSqr(Point3D const &coord) const;

		void getValues(double *x, double *y, double *z) const;

		double x, y, z;
};



#endif
