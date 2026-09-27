# Writes a SPIR-V binary as a C++ byte array: -DIN=<file.spv> -DOUT=<file.cpp> -DSYMBOL=<name>
# Defines `const unsigned char <SYMBOL>[]` (4-byte aligned) and `const size_t <SYMBOL>_size` (bytes).
file(READ ${IN} _hex HEX)
string(LENGTH "${_hex}" _len)
math(EXPR _bytes "${_len} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _body "${_hex}")
string(REGEX REPLACE "((0x..,){32})" "\\1\n" _body "${_body}")
file(WRITE ${OUT} "// Generated from ${IN}; do not edit.\n#include <cstddef>\n"
  "alignas(4) extern const unsigned char ${SYMBOL}[] = {\n${_body}};\n"
  "extern const size_t ${SYMBOL}_size = ${_bytes};\n")
