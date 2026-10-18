// Streaming MATSim XML reader. Expat decodes XML; zlib accepts gzip or plain XML.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <expat.h>
#include <zlib.h>
#include <cstring>
#include <string>
#include <new>

struct Reader {
    XML_Parser parser = nullptr;
    PyObject *nodes = PyList_New(0), *links = PyList_New(0);
    PyObject *node_attrs = PyList_New(0), *link_attrs = PyList_New(0);
    PyObject *metadata = PyDict_New(), *attributes = PyList_New(0);
    PyObject *id = nullptr, *link_dict = nullptr;
    bool link = false, in_attribute = false, skip = false, has_name = false;
    std::string name, type, text;
    ~Reader() {
        if (parser) XML_ParserFree(parser);
        Py_XDECREF(nodes); Py_XDECREF(links); Py_XDECREF(node_attrs);
        Py_XDECREF(link_attrs); Py_XDECREF(metadata); Py_XDECREF(attributes);
        Py_XDECREF(id); Py_XDECREF(link_dict);
    }
    void stop() { if (PyErr_Occurred()) XML_StopParser(parser, XML_FALSE); }
};

static const char* local(const char* name) {
    const char* sep = strrchr(name, '}');
    return sep ? sep + 1 : name;
}
static const char* get(const char** attrs, const char* name, const char* fallback = nullptr) {
    for (int i = 0; attrs[i]; i += 2)
        if (!strcmp(attrs[i], name)) return attrs[i + 1];
    return fallback;
}
static PyObject* string(const char* value) {
    if (!value) { PyErr_SetString(PyExc_ValueError, "Missing required network XML attribute"); return nullptr; }
    return PyUnicode_FromString(value);
}
static PyObject* number(const char* value, bool integer = false) {
    PyObject* text = string(value);
    if (!text) return nullptr;
    PyObject* result = integer ? PyLong_FromUnicodeObject(text, 10) : PyFloat_FromString(text);
    Py_DECREF(text);
    return result;
}
static void append(PyObject* list, PyObject* item) {
    if (item) { PyList_Append(list, item); Py_DECREF(item); }
}

static void start_impl(Reader& r, const char* raw, const char** a) {
    const char* tag = local(raw);
    if (!strcmp(tag, "node") || !strcmp(tag, "link")) {
        Py_CLEAR(r.id); Py_CLEAR(r.link_dict);
        r.id = string(get(a, "id"));
        if (!r.id) return;
        r.link = !strcmp(tag, "link");
        if (r.link && !r.skip) r.link_dict = PyDict_New();
        const char* node_names[] = {"x", "y", "z"};
        const char* link_names[] = {"from", "to", "length", "freespeed", "capacity", "permlanes", "modes", "oneway"};
        int count = r.link ? 9 : 4;
        PyObject* row = PyTuple_New(count);
        if (!row) return;
        PyTuple_SET_ITEM(row, 0, Py_NewRef(r.id));
        for (int i = 1; i < count; ++i) {
            PyObject* value;
            if (!r.link) {
                const char* text = get(a, node_names[i-1]);
                value = i == 3 && !text ? Py_NewRef(Py_None) : number(text);
            } else {
                const char* text = get(a, link_names[i-1], i == 7 ? "" : i == 8 ? "1" : nullptr);
                value = i <= 2 || i == 7 ? string(text) : number(text, i == 8);
            }
            if (!value) { Py_DECREF(row); return; }
            PyTuple_SET_ITEM(row, i, value);
        }
        append(r.link ? r.links : r.nodes, row);
    } else if (!strcmp(tag, "attribute")) {
        const char* name = get(a, "name");
        r.has_name = name != nullptr;
        r.name = name ? name : "";
        r.type = get(a, "class", "");
        r.text.clear();
        r.in_attribute = true;
    }
}
static void end_impl(Reader& r, const char* raw) {
    const char* tag = local(raw);
    if (!strcmp(tag, "attribute")) {
        r.in_attribute = false;
        if (r.name == "coordinateReferenceSystem") {
            PyObject* value = r.text.empty() ? Py_NewRef(Py_None) : string(r.text.c_str());
            if (value) { PyDict_SetItemString(r.metadata, r.name.c_str(), value); Py_DECREF(value); }
        } else if (!r.skip && r.id) {
            PyObject* value;
            if (r.text.empty()) value = Py_NewRef(Py_None);
            else if (r.type == "java.lang.Long" || r.type == "java.lang.Integer") value = number(r.text.c_str(), true);
            else if (r.type == "java.lang.Double") value = number(r.text.c_str());
            else value = string(r.text.c_str());
            if (!value) return;
            PyObject* key = r.has_name ? PyUnicode_InternFromString(r.name.c_str()) : Py_NewRef(Py_None);
            if (key) {
                append(r.link ? r.link_attrs : r.node_attrs, PyTuple_Pack(3, r.id, key, value));
                if (r.link && r.link_dict) PyDict_SetItem(r.link_dict, key, value);
                Py_DECREF(key);
            }
            Py_DECREF(value);
        }
    } else if (!strcmp(tag, "node") || !strcmp(tag, "link")) {
        if (r.link) PyList_Append(r.attributes,
            r.link_dict && PyDict_Size(r.link_dict) ? r.link_dict : Py_None);
        Py_CLEAR(r.id); Py_CLEAR(r.link_dict);
    }
}
static void start(void* data, const char* tag, const char** attrs) {
    Reader& r = *static_cast<Reader*>(data);
    try { start_impl(r, tag, attrs); } catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    r.stop();
}
static void end(void* data, const char* tag) {
    Reader& r = *static_cast<Reader*>(data);
    try { end_impl(r, tag); } catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    r.stop();
}
static void characters(void* data, const char* value, int size) {
    Reader& r = *static_cast<Reader*>(data);
    try { if (r.in_attribute && (!r.skip || r.name == "coordinateReferenceSystem")) r.text.append(value, size); }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    r.stop();
}
static PyObject* read(PyObject*, PyObject* args) {
    const char* path; int skip = 0;
    if (!PyArg_ParseTuple(args, "s|p", &path, &skip)) return nullptr;
    gzFile file = gzopen(path, "rb");
    if (!file) return PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
    gzbuffer(file, 1024 * 1024);
    Reader r;
    r.skip = skip;
    r.parser = XML_ParserCreateNS(nullptr, '}');
    if (!r.parser || !r.nodes || !r.links || !r.node_attrs || !r.link_attrs || !r.metadata || !r.attributes) {
        gzclose(file); return PyErr_NoMemory();
    }
    XML_SetUserData(r.parser, &r);
    XML_SetElementHandler(r.parser, start, end);
    XML_SetCharacterDataHandler(r.parser, characters);
    char* buffer = new (std::nothrow) char[1024 * 1024];
    if (!buffer) { gzclose(file); return PyErr_NoMemory(); }
    int size;
    while ((size = gzread(file, buffer, 1024 * 1024)) > 0) {
        if (XML_Parse(r.parser, buffer, size, XML_FALSE) == XML_STATUS_ERROR) break;
        if (PyErr_CheckSignals() < 0) break;
    }
    if (!PyErr_Occurred()) {
        if (size < 0) {
            int code; const char* message = gzerror(file, &code);
            PyErr_SetString(PyExc_OSError, message);
        } else if (XML_GetErrorCode(r.parser) != XML_ERROR_NONE ||
                   XML_Parse(r.parser, "", 0, XML_TRUE) == XML_STATUS_ERROR) {
            PyErr_Format(PyExc_ValueError, "Invalid network XML at line %lu: %s",
                static_cast<unsigned long>(XML_GetCurrentLineNumber(r.parser)), XML_ErrorString(XML_GetErrorCode(r.parser)));
        }
    }
    delete[] buffer;
    int status = gzclose(file);
    if (status != Z_OK && !PyErr_Occurred()) PyErr_SetString(PyExc_OSError, "Invalid gzip network stream");
    if (PyErr_Occurred()) return nullptr;
    return PyTuple_Pack(6, r.nodes, r.links, r.node_attrs, r.link_attrs, r.metadata, r.attributes);
}
static PyMethodDef methods[] = {{"read", read, METH_VARARGS, "Read XML/gzip into network records."}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_network_native", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__network_native() { return PyModule_Create(&module); }
