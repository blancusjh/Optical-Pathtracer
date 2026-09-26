#include "scene_parser.hpp"

#include <cctype>
#include <cmath>
#include <map>
#include <stdexcept>

namespace owe {

ValuePtr Value::get(const std::string& key) const {
    for (auto it = named.rbegin(); it != named.rend(); ++it)
        if (it->first == key) return it->second;
    return nullptr;
}

namespace {

struct Token {
    enum class T { Ident, Number, String, Sym, End } t = T::End;
    std::string s;
    double num = 0;
    int line = 0;
};

struct UnitDef {
    Value::Unit unit;
    double scale;
};
const std::map<std::string, UnitDef>& units() {
    static const std::map<std::string, UnitDef> u = {
        {"m", {Value::Unit::Length, 1}},        {"cm", {Value::Unit::Length, 1e-2}},
        {"mm", {Value::Unit::Length, 1e-3}},    {"um", {Value::Unit::Length, 1e-6}},
        {"nm", {Value::Unit::Length, 1e-9}},    {"km", {Value::Unit::Length, 1e3}},
        {"in", {Value::Unit::Length, 0.0254}},  {"deg", {Value::Unit::Angle, 3.14159265358979323846 / 180}},
        {"rad", {Value::Unit::Angle, 1}},       {"K", {Value::Unit::Temperature, 1}},
    };
    return u;
}

class Lexer {
public:
    Lexer(const std::string& s, std::string origin) : s_(s), origin_(std::move(origin)) {}
    std::vector<Token> run() {
        std::vector<Token> out;
        while (true) {
            skip();
            Token t;
            t.line = line_;
            if (i_ >= s_.size()) { t.t = Token::T::End; out.push_back(t); break; }
            char c = s_[i_];
            if (std::isalpha((unsigned char)c) || c == '_' || (unsigned char)c >= 0x80) {
                size_t j = i_;
                while (j < s_.size() && (std::isalnum((unsigned char)s_[j]) || s_[j] == '_' || s_[j] == '-' ||
                                         s_[j] == '.' || (unsigned char)s_[j] >= 0x80))
                    ++j;
                // Do not swallow a trailing '-' or '.' (e.g. "x-" before a number is not valid anyway).
                while (j > i_ + 1 && (s_[j - 1] == '-' || s_[j - 1] == '.')) --j;
                t.t = Token::T::Ident;
                t.s = s_.substr(i_, j - i_);
                i_ = j;
            } else if (std::isdigit((unsigned char)c) || ((c == '-' || c == '+' || c == '.') && i_ + 1 < s_.size() &&
                                                           (std::isdigit((unsigned char)s_[i_ + 1]) || s_[i_ + 1] == '.'))) {
                size_t j = i_;
                if (s_[j] == '-' || s_[j] == '+') ++j;
                while (j < s_.size() && (std::isdigit((unsigned char)s_[j]) || s_[j] == '.')) ++j;
                if (j < s_.size() && (s_[j] == 'e' || s_[j] == 'E')) {
                    size_t k = j + 1;
                    if (k < s_.size() && (s_[k] == '-' || s_[k] == '+')) ++k;
                    if (k < s_.size() && std::isdigit((unsigned char)s_[k])) {
                        j = k;
                        while (j < s_.size() && std::isdigit((unsigned char)s_[j])) ++j;
                    }
                }
                t.t = Token::T::Number;
                t.s = s_.substr(i_, j - i_);
                try { t.num = std::stod(t.s); } catch (...) { fail("bad number '" + t.s + "'"); }
                i_ = j;
            } else if (c == '"') {
                size_t j = i_ + 1;
                std::string v;
                while (j < s_.size() && s_[j] != '"') {
                    if (s_[j] == '\n') fail("unterminated string");
                    v += s_[j++];
                }
                if (j >= s_.size()) fail("unterminated string");
                t.t = Token::T::String;
                t.s = v;
                i_ = j + 1;
            } else if (std::string("{}()[]=,;").find(c) != std::string::npos) {
                t.t = Token::T::Sym;
                t.s = std::string(1, c);
                ++i_;
            } else {
                fail(std::string("unexpected character '") + c + "'");
            }
            out.push_back(t);
        }
        return out;
    }

private:
    void skip() {
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == '\n') { ++line_; ++i_; }
            else if (std::isspace((unsigned char)c)) ++i_;
            else if (c == '#' || (c == '/' && i_ + 1 < s_.size() && s_[i_ + 1] == '/')) {
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
            } else break;
        }
    }
    [[noreturn]] void fail(const std::string& m) {
        throw std::runtime_error(origin_ + ":" + std::to_string(line_) + ": " + m);
    }
    const std::string& s_;
    std::string origin_;
    size_t i_ = 0;
    int line_ = 1;
};

class Parser {
public:
    Parser(std::vector<Token> t, std::string origin) : t_(std::move(t)), origin_(std::move(origin)) {}

    ValuePtr document() {
        auto doc = std::make_shared<Value>();
        doc->kind = Value::Kind::Block;
        doc->str = "document";
        body(*doc, true);
        return doc;
    }

private:
    const Token& peek(size_t k = 0) const { return t_[std::min(p_ + k, t_.size() - 1)]; }
    bool isSym(const Token& t, const char* s) const { return t.t == Token::T::Sym && t.s == s; }
    Token next() { return t_[std::min(p_++, t_.size() - 1)]; }
    [[noreturn]] void fail(const std::string& m, int line) {
        throw std::runtime_error(origin_ + ":" + std::to_string(line) + ": " + m);
    }
    void expect(const char* s) {
        Token t = next();
        if (!isSym(t, s)) fail(std::string("expected '") + s + "' but found '" + t.s + "'", t.line);
    }

    void body(Value& blk, bool topLevel) {
        while (true) {
            const Token& t = peek();
            if (t.t == Token::T::End) {
                if (!topLevel) fail("unexpected end of file inside block '" + blk.str + "'", t.line);
                return;
            }
            if (isSym(t, "}")) {
                if (topLevel) fail("unmatched '}'", t.line);
                return;
            }
            if (isSym(t, ";")) { next(); continue; }
            if (t.t != Token::T::Ident) fail("expected a keyword, found '" + t.s + "'", t.line);
            if (isSym(peek(1), "=")) {
                Token key = next();
                next();
                blk.named.push_back({key.s, value()});
                continue;
            }
            // Block: keyword [name] { ... }
            Token kw = next();
            auto child = std::make_shared<Value>();
            child->kind = Value::Kind::Block;
            child->str = kw.s;
            child->line = kw.line;
            if (peek().t == Token::T::Ident || peek().t == Token::T::String) child->name = next().s;
            expect("{");
            body(*child, false);
            expect("}");
            blk.items.push_back(child);
        }
    }

    ValuePtr value() {
        ValuePtr v = primary();
        // Optional unit suffix applies to numbers and tuples of numbers.
        if (peek().t == Token::T::Ident) {
            auto it = units().find(peek().s);
            if (it != units().end() && (v->kind == Value::Kind::Number || v->kind == Value::Kind::Tuple ||
                                        v->kind == Value::Kind::List)) {
                next();
                applyUnit(*v, it->second);
            }
        }
        return v;
    }

    void applyUnit(Value& v, const UnitDef& u) {
        if (v.kind == Value::Kind::Number) {
            if (v.unit != Value::Unit::None) fail("value already has a unit", v.line);
            v.unit = u.unit;
            v.num *= u.scale;
        } else if (v.kind == Value::Kind::Tuple || v.kind == Value::Kind::List) {
            for (auto& x : v.items) applyUnit(*x, u);
        } else {
            fail("units apply to numbers only", v.line);
        }
    }

    ValuePtr primary() {
        Token t = next();
        auto v = std::make_shared<Value>();
        v->line = t.line;
        if (t.t == Token::T::Number) {
            v->kind = Value::Kind::Number;
            v->num = t.num;
            return v;
        }
        if (t.t == Token::T::String) {
            v->kind = Value::Kind::String;
            v->str = t.s;
            return v;
        }
        if (t.t == Token::T::Ident) {
            if (t.s == "inf" || t.s == "infinity") {
                v->kind = Value::Kind::Number;
                v->num = INFINITY;
                return v;
            }
            if (isSym(peek(), "(")) {
                next();
                v->kind = Value::Kind::Call;
                v->str = t.s;
                if (!isSym(peek(), ")")) {
                    while (true) {
                        if (peek().t == Token::T::Ident && isSym(peek(1), "=")) {
                            Token k = next();
                            next();
                            v->named.push_back({k.s, value()});
                        } else {
                            v->items.push_back(value());
                        }
                        if (isSym(peek(), ",")) { next(); continue; }
                        break;
                    }
                }
                expect(")");
                return v;
            }
            v->kind = Value::Kind::Ident;
            v->str = t.s;
            return v;
        }
        if (isSym(t, "(")) {
            v->kind = Value::Kind::Tuple;
            while (true) {
                v->items.push_back(value());
                if (isSym(peek(), ",")) { next(); continue; }
                break;
            }
            expect(")");
            if (v->items.size() == 1) return v->items[0];
            return v;
        }
        if (isSym(t, "[")) {
            v->kind = Value::Kind::List;
            while (!isSym(peek(), "]")) {
                v->items.push_back(value());
                if (isSym(peek(), ",")) next();
                if (peek().t == Token::T::End) fail("unterminated list", t.line);
            }
            next();
            return v;
        }
        fail("unexpected '" + t.s + "'", t.line);
    }

    std::vector<Token> t_;
    std::string origin_;
    size_t p_ = 0;
};

}  // namespace

ValuePtr parseSceneText(const std::string& text, const std::string& origin) {
    Lexer lx(text, origin);
    Parser ps(lx.run(), origin);
    return ps.document();
}

}  // namespace owe
