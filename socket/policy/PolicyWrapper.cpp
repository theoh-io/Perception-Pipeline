#include <Python.h>
#include <iostream>
#include <ctime>
#include <string>
#include "PyUtils.h"

int policy_wrapper()
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;

    const char *module_name = "sarl";
    const char *func_name = "predict";

    Py_Initialize();
    pName = PyUnicode_DecodeFSDefault(module_name);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, func_name);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            vector<float> input_state = {1, 2, 3};
            pArgs = vectorToTuple_Float(input_state);
            clock_t begin = clock();
            pValue = PyObject_CallObject(pFunc, pArgs);
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            cout << "Elapsed time(s): " << elapsed_secs << endl;
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                vector<float> action = listTupleToVector_Float(pValue);
                cout << action[0] << '\t' << action[1] << endl;
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", func_name);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", module_name);
        return 1;
    }
    Py_Finalize();
    return 0;
}