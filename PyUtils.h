//
// Created by changan on 9/19/18.
//

#ifndef DYNAV_LOOMO_PYUTILS_H
#define DYNAV_LOOMO_PYUTILS_H

#include <vector>
using namespace std;

PyObject* vectorToList_Float(const vector<float> &data);

// ======
// TUPLES
// ======

PyObject* vectorToTuple_Float(const vector<float> &data);

PyObject* vectorVectorToTuple_Float(const vector< vector< float > > &data);

// PyObject -> Vector
vector<float> listTupleToVector_Float(PyObject* incoming);

// PyObject -> Vector
vector<int> listTupleToVector_Int(PyObject* incoming);

#endif //DYNAV_LOOMO_PYUTILS_H
