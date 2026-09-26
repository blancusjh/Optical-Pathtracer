// Parser for the .owe scene language:
//
//   units = mm
//   body CrownSinglet {
//       type = lens
//       medium = N-BK7
//       front = sphere(R = 48 mm)
//       back = asphere(R = -120, k = -1.2, A4 = 2e-6)
//       thickness = 6.2 mm
//       diameter = 25 mm
//       position = (0, 0, 100) mm
//   }
//
// Values: numbers with optional units, strings, identifiers, tuples (...),
// lists [...], calls name(args, key = value) and nested blocks.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace owe {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
    enum class Kind { Number, String, Ident, Tuple, List, Call, Block } kind = Kind::Number;
    enum class Unit { None, Length, Angle, Temperature } unit = Unit::None;
    double num = 0;           // SI value when unit != None
    std::string str;          // String/Ident text, Call name, Block keyword
    std::string name;         // Block name
    std::vector<ValuePtr> items;                            // tuple/list/call positional args/child blocks
    std::vector<std::pair<std::string, ValuePtr>> named;    // call keyword args / block assignments
    int line = 0;

    ValuePtr get(const std::string& key) const;              // last assignment of key, or nullptr
    bool has(const std::string& key) const { return get(key) != nullptr; }
    std::string where() const { return "line " + std::to_string(line); }
};

// Parses a document; returns a Block with keyword "document".
ValuePtr parseSceneText(const std::string& text, const std::string& origin);

}  // namespace owe
