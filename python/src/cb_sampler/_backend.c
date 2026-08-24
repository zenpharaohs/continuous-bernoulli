/* SPDX-License-Identifier: MIT */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

#include "../../../src/c/cb_core.c"

typedef struct {
    PyObject_HEAD
    cb_stream_t *stream;
    uint64_t total_draws;
} BackendStream;

static int
parse_uint64(PyObject *obj, const char *name, uint64_t *out)
{
    unsigned long long value = PyLong_AsUnsignedLongLong(obj);
    if (value == (unsigned long long)-1 && PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "%s must be an unsigned 64-bit integer", name);
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static int
validate_stats(double chi, double nu)
{
    if (!isfinite(chi) || !isfinite(nu) || nu < 0.0 || chi < 0.0 || chi > nu) {
        PyErr_SetString(PyExc_ValueError, "Require finite sufficient statistics 0 <= chi <= nu");
        return 0;
    }
    return 1;
}

static int
validate_observation(double x)
{
    if (!isfinite(x) || x < 0.0 || x > 1.0) {
        PyErr_SetString(PyExc_ValueError, "Observations must be finite values in [0, 1]");
        return 0;
    }
    return 1;
}

static int
require_open(BackendStream *self)
{
    if (self->stream == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "CB stream is closed");
        return 0;
    }
    return 1;
}

static int
BackendStream_init(BackendStream *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"chi", "nu", "seed", "stream_idx", "buf_size", NULL};
    double chi;
    double nu;
    PyObject *seed_obj;
    PyObject *stream_idx_obj;
    int buf_size = 256;
    uint64_t seed;
    uint64_t stream_idx;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddOO|i", kwlist,
                                     &chi, &nu, &seed_obj, &stream_idx_obj, &buf_size)) {
        return -1;
    }
    if (!validate_stats(chi, nu)) {
        return -1;
    }
    if (!parse_uint64(seed_obj, "seed", &seed) ||
        !parse_uint64(stream_idx_obj, "stream_idx", &stream_idx)) {
        return -1;
    }
    if (buf_size < 0) {
        PyErr_SetString(PyExc_ValueError, "buf_size must be nonnegative");
        return -1;
    }

    self->stream = cb_stream_create(chi, nu, seed, stream_idx, buf_size);
    if (self->stream == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->total_draws = 0;
    return 0;
}

static void
BackendStream_dealloc(BackendStream *self)
{
    if (self->stream != NULL) {
        cb_stream_destroy(self->stream);
        self->stream = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
BackendStream_close(BackendStream *self, PyObject *Py_UNUSED(ignored))
{
    if (self->stream != NULL) {
        cb_stream_destroy(self->stream);
        self->stream = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
BackendStream_draw(BackendStream *self, PyObject *args)
{
    int n;
    PyObject *list;
    double *buf;

    if (!PyArg_ParseTuple(args, "i", &n)) {
        return NULL;
    }
    if (!require_open(self)) {
        return NULL;
    }
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "draw count must be a positive integer");
        return NULL;
    }

    list = PyList_New((Py_ssize_t)n);
    if (list == NULL) {
        return NULL;
    }
    buf = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (buf == NULL) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }

    if (cb_stream_draw(self->stream, n, buf) != n) {
        PyMem_Free(buf);
        Py_DECREF(list);
        PyErr_SetString(PyExc_RuntimeError, "backend failed to draw requested sample count");
        return NULL;
    }
    self->total_draws += (uint64_t)n;
    for (int i = 0; i < n; i++) {
        PyObject *value = PyFloat_FromDouble(buf[i]);
        if (value == NULL) {
            PyMem_Free(buf);
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, (Py_ssize_t)i, value);
    }
    PyMem_Free(buf);
    return list;
}

static PyObject *
BackendStream_update(BackendStream *self, PyObject *args)
{
    double x;
    if (!PyArg_ParseTuple(args, "d", &x)) {
        return NULL;
    }
    if (!require_open(self) || !validate_observation(x)) {
        return NULL;
    }
    if (!cb_stream_update(self->stream, x)) {
        PyErr_SetString(PyExc_ValueError, "observation would produce invalid sufficient statistics");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
BackendStream_update_batch(BackendStream *self, PyObject *args)
{
    PyObject *seq_obj;
    PyObject *seq;
    Py_ssize_t n;
    double *xs;

    if (!PyArg_ParseTuple(args, "O", &seq_obj)) {
        return NULL;
    }
    if (!require_open(self)) {
        return NULL;
    }
    seq = PySequence_Fast(seq_obj, "observations must be a sequence");
    if (seq == NULL) {
        return NULL;
    }
    n = PySequence_Fast_GET_SIZE(seq);
    if (n > INT_MAX) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "too many observations for one update_batch call");
        return NULL;
    }
    xs = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (xs == NULL) {
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        xs[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred() || !validate_observation(xs[i])) {
            PyMem_Free(xs);
            Py_DECREF(seq);
            return NULL;
        }
    }
    if (!cb_stream_update_batch(self->stream, xs, (int)n)) {
        PyMem_Free(xs);
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "observations would produce invalid sufficient statistics");
        return NULL;
    }
    PyMem_Free(xs);
    Py_DECREF(seq);
    Py_RETURN_NONE;
}

static PyObject *
BackendStream_set_stats(BackendStream *self, PyObject *args)
{
    double chi;
    double nu;
    if (!PyArg_ParseTuple(args, "dd", &chi, &nu)) {
        return NULL;
    }
    if (!require_open(self) || !validate_stats(chi, nu)) {
        return NULL;
    }
    if (!cb_stream_set_stats(self->stream, chi, nu)) {
        PyErr_SetString(PyExc_ValueError, "invalid sufficient statistics");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
BackendStream_peek(BackendStream *self, PyObject *Py_UNUSED(ignored))
{
    double chi;
    double nu;
    double sigma;
    int regime;

    if (!require_open(self)) {
        return NULL;
    }
    cb_stream_peek(self->stream, &chi, &nu, &regime, &sigma);
    return Py_BuildValue("{s:d,s:d,s:i,s:d}",
                         "chi", chi,
                         "nu", nu,
                         "regime", regime,
                         "sigma", sigma);
}

static PyObject *
BackendStream_diagnostics(BackendStream *self, PyObject *Py_UNUSED(ignored))
{
    if (!require_open(self)) {
        return NULL;
    }
    return Py_BuildValue(
        "{s:O,s:K,s:K}",
        "enabled", Py_False,
        "total_draws", (unsigned long long)self->total_draws,
        "hull_rebuilds", (unsigned long long)cb_stream_hull_rebuilds(self->stream));
}

static PyMethodDef BackendStream_methods[] = {
    {"close", (PyCFunction)BackendStream_close, METH_NOARGS, "Close the backend stream."},
    {"draw", (PyCFunction)BackendStream_draw, METH_VARARGS, "Draw samples from the backend stream."},
    {"update", (PyCFunction)BackendStream_update, METH_VARARGS, "Update with one observation."},
    {"update_batch", (PyCFunction)BackendStream_update_batch, METH_VARARGS, "Update with a sequence of observations."},
    {"set_stats", (PyCFunction)BackendStream_set_stats, METH_VARARGS, "Replace the sufficient statistics."},
    {"peek", (PyCFunction)BackendStream_peek, METH_NOARGS, "Return sufficient statistics and route metadata."},
    {"diagnostics", (PyCFunction)BackendStream_diagnostics, METH_NOARGS, "Return backend diagnostics."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject BackendStreamType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cb_sampler._backend.BackendStream",
    .tp_basicsize = sizeof(BackendStream),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)BackendStream_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Owned C continuous-binomial sampler stream.",
    .tp_methods = BackendStream_methods,
    .tp_init = (initproc)BackendStream_init,
    .tp_new = PyType_GenericNew,
};

static PyObject *
sample_c(PyObject *Py_UNUSED(module), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"chi", "nu", "seed", "n", NULL};
    double chi;
    double nu;
    PyObject *seed_obj;
    int n;
    uint64_t seed;
    double *buf;
    PyObject *list;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddOi", kwlist,
                                     &chi, &nu, &seed_obj, &n)) {
        return NULL;
    }
    if (!validate_stats(chi, nu) || !parse_uint64(seed_obj, "seed", &seed)) {
        return NULL;
    }
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "draw count must be a positive integer");
        return NULL;
    }
    list = PyList_New((Py_ssize_t)n);
    if (list == NULL) {
        return NULL;
    }
    buf = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (buf == NULL) {
        Py_DECREF(list);
        return PyErr_NoMemory();
    }
    if (cb_sample_c(chi, nu, seed, n, buf) != n) {
        PyMem_Free(buf);
        Py_DECREF(list);
        PyErr_SetString(PyExc_RuntimeError, "backend failed to draw requested sample count");
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        PyObject *value = PyFloat_FromDouble(buf[i]);
        if (value == NULL) {
            PyMem_Free(buf);
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, (Py_ssize_t)i, value);
    }
    PyMem_Free(buf);
    return list;
}

static PyObject *
draw_streams_c(PyObject *Py_UNUSED(module), PyObject *args)
{
    PyObject *sequence_obj;
    PyObject *sequence;
    PyObject *list;
    Py_ssize_t count;

    if (!PyArg_ParseTuple(args, "O", &sequence_obj)) {
        return NULL;
    }
    sequence = PySequence_Fast(sequence_obj, "streams must be a sequence");
    if (sequence == NULL) {
        return NULL;
    }
    count = PySequence_Fast_GET_SIZE(sequence);
    if (count <= 0) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "streams must be nonempty");
        return NULL;
    }
    list = PyList_New(count);
    if (list == NULL) {
        Py_DECREF(sequence);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(sequence, i);
        BackendStream *stream;
        PyObject *value;
        double draw;

        if (!PyObject_TypeCheck(item, &BackendStreamType)) {
            Py_DECREF(list);
            Py_DECREF(sequence);
            PyErr_SetString(PyExc_TypeError, "all items must be backend streams");
            return NULL;
        }
        stream = (BackendStream *)item;
        if (!require_open(stream)) {
            Py_DECREF(list);
            Py_DECREF(sequence);
            return NULL;
        }
        if (cb_stream_draw(stream->stream, 1, &draw) != 1) {
            Py_DECREF(list);
            Py_DECREF(sequence);
            PyErr_SetString(PyExc_RuntimeError, "backend failed to draw from stream");
            return NULL;
        }
        stream->total_draws += 1;
        value = PyFloat_FromDouble(draw);
        if (value == NULL) {
            Py_DECREF(list);
            Py_DECREF(sequence);
            return NULL;
        }
        PyList_SET_ITEM(list, i, value);
    }
    Py_DECREF(sequence);
    return list;
}

static PyMethodDef module_methods[] = {
    {"sample_c", (PyCFunction)sample_c, METH_VARARGS | METH_KEYWORDS, "Draw samples through the one-shot C helper."},
    {"draw_streams_c", (PyCFunction)draw_streams_c, METH_VARARGS, "Draw once from each backend stream."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef backend_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "cb_sampler._backend",
    .m_doc = "CPython binding for the continuous-binomial C backend.",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC
PyInit__backend(void)
{
    PyObject *module;

    if (PyType_Ready(&BackendStreamType) < 0) {
        return NULL;
    }
    module = PyModule_Create(&backend_module);
    if (module == NULL) {
        return NULL;
    }
    Py_INCREF(&BackendStreamType);
    if (PyModule_AddObject(module, "BackendStream", (PyObject *)&BackendStreamType) < 0) {
        Py_DECREF(&BackendStreamType);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
