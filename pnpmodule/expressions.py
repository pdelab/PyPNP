#! /usr/bin/python2.7
"""
Linear Expression Class.
    You have two choices: the python version or the C++
    Python:
        Exp = Linear_Function(0, -Lx/2.0, Lx/2.0, 0.0, -2.0, degree=2)
    C++:
        Exp = Expression(LinearFunction_cpp, degree=2)
        CationExpression .update(0, -Lx/2.0, Lx/2.0, 0.0, -2.0)
"""
from dolfin import *
import numpy as np


class Linear_Function(Expression):
    def __init__(self, coordinate, mesh_min, mesh_max,
                 lower_value, upper_value, **kwargs):
        self._coordinate = coordinate
        self._mesh_min = mesh_min
        self._mesh_max = mesh_max
        self._lower_value = lower_value
        self._upper_value = upper_value
        self._distance = 1.0 / (mesh_max - mesh_min)

    def eval(self, value, x):
        value = self._lower_value * (self._mesh_max - x[self._coordinate])\
            * self._distance
        value += self._upper_value * (x[self._coordinate] - self._mesh_min)\
            * self._distance

    def value_shape(self):
        return (1,)


LinearFunction_cpp = '''
namespace dolfin {
    class Linear_Function : public Expression
    {
    public:
      Linear_Function() : Expression() {};
      void eval(Array<double>& values, const Array<double>& x) const
      {
          values[0] = _lower_value * (_mesh_max - x[_coordinate]) * _distance;
          values[0] += _upper_value * (x[_coordinate] - _mesh_min) * _distance;
      }
      void update (
             std::size_t coordinate,
             double mesh_min,
             double mesh_max,
             double lower_value,
             double upper_value) {

                   _coordinate = coordinate;
                   _mesh_min = mesh_min;
                   _mesh_max = mesh_max;
                   _lower_value = lower_value;
                   _upper_value = upper_value;
                   _distance = 1.0 / (_mesh_max - _mesh_min);

           }

        std::size_t _coordinate;
        double _mesh_min, _mesh_max;
        double _lower_value, _upper_value;
        double _distance;
    };
}
'''
