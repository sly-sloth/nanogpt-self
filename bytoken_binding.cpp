#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bytoken.h"

namespace py = pybind11;

PYBIND11_MODULE(bytoken, m) {
    m.doc() = "Byte Pair Encoding Tokenizer";

    py::class_<ByToken>(m, "ByToken")
        .def(py::init<>(), "Initialize a new ByToken tokenizer")

        .def("train", &ByToken::train,
             py::arg("text_corpus"),
             py::arg("vocab_size"),
             py::arg("verbose") = false,
             "Train the tokenizer on a text corpus with a target vocabulary size")

        .def("encode", &ByToken::encode,
             py::arg("text"),
             "Encode a string into token indices")

        .def("decode", &ByToken::decode,
             py::arg("idx"),
             "Decode token indices back into a string");
}