#include "owe/core/medium.hpp"

#include <cstdio>
#include <map>
#include <sstream>
#include <stdexcept>

namespace owe {

IndexModel IndexModel::constant(double n) {
    IndexModel m;
    m.kind_ = Kind::Constant;
    m.a_ = {n};
    return m;
}

IndexModel IndexModel::cauchy(double A, double B, double C) {
    IndexModel m;
    m.kind_ = Kind::Cauchy;
    m.a_ = {A, B, C};
    return m;
}

IndexModel IndexModel::sellmeier(std::vector<double> B, std::vector<double> C) {
    if (B.size() != C.size() || B.empty()) throw std::runtime_error("sellmeier needs matching non-empty B and C");
    IndexModel m;
    m.kind_ = Kind::Sellmeier;
    m.a_ = std::move(B);
    m.b_ = std::move(C);
    return m;
}

IndexModel IndexModel::tabulated(std::vector<double> lambdasNm, std::vector<double> n) {
    IndexModel m;
    m.kind_ = Kind::Tabulated;
    m.table_ = Spectrum::tabulated(std::move(lambdasNm), std::move(n));
    return m;
}

IndexModel IndexModel::ciddorAir() {
    IndexModel m;
    m.kind_ = Kind::Ciddor;
    return m;
}

double IndexModel::n(double lambdaNm) const {
    double v = nRelative(lambdaNm);
    if (relAir_) {
        double s2 = 1.0 / sqr(lambdaNm * 1e-3);
        v *= 1.0 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2);
    }
    return v;
}

double IndexModel::nRelative(double lambdaNm) const {
    double l = lambdaNm * 1e-3;  // μm
    double l2 = l * l;
    switch (kind_) {
        case Kind::Constant: return a_[0];
        case Kind::Cauchy: return a_[0] + a_[1] / l2 + a_[2] / (l2 * l2);
        case Kind::Sellmeier: {
            double s = 1;
            for (size_t i = 0; i < a_.size(); ++i) s += a_[i] * l2 / (l2 - b_[i]);
            return std::sqrt(s);
        }
        case Kind::Tabulated: return table_.eval(lambdaNm);
        case Kind::Ciddor: {
            double s2 = 1.0 / l2;
            return 1.0 + 0.05792105 / (238.0185 - s2) + 0.00167917 / (57.362 - s2);
        }
    }
    return 1;
}

std::string IndexModel::describe() const {
    std::ostringstream os;
    switch (kind_) {
        case Kind::Constant: os << "constant(" << a_[0] << ")"; break;
        case Kind::Cauchy: os << "cauchy(A=" << a_[0] << ", B=" << a_[1] << ", C=" << a_[2] << ")"; break;
        case Kind::Sellmeier:
            os << "sellmeier(";
            for (size_t i = 0; i < a_.size(); ++i) os << (i ? ", " : "") << "B" << i + 1 << "=" << a_[i] << " C" << i + 1 << "=" << b_[i];
            os << ")";
            break;
        case Kind::Tabulated: os << "tabulated"; break;
        case Kind::Ciddor: os << "ciddor-air"; break;
    }
    if (relAir_) os << " rel. air";
    return os.str();
}

double abbeNumber(const IndexModel& m) {
    return (m.nRelative(LambdaD) - 1) / (m.nRelative(LambdaF) - m.nRelative(LambdaC));
}

namespace {

Medium makeSellmeier(const std::string& name, std::vector<double> B, std::vector<double> C, bool relAir = false) {
    Medium m;
    m.name = name;
    m.index = IndexModel::sellmeier(std::move(B), std::move(C));
    if (relAir) m.index = m.index.relativeToAir();
    return m;
}

std::vector<double> squares(std::initializer_list<double> v) {
    std::vector<double> r;
    for (double x : v) r.push_back(x * x);
    return r;
}

const std::map<std::string, Medium>& catalog() {
    static const std::map<std::string, Medium> cat = [] {
        std::map<std::string, Medium> c;
        auto add = [&](Medium m) { c[m.name] = m; };
        {
            Medium m; m.name = "vacuum"; add(m);
        }
        {
            Medium m; m.name = "air"; m.index = IndexModel::ciddorAir(); add(m);
        }
        {
            Medium m; m.name = "opaque"; m.opaque = true; add(m);
        }
        // Schott catalog Sellmeier coefficients (C in μm², indices relative to air).
        add(makeSellmeier("N-BK7", {1.03961212, 0.231792344, 1.01046945}, {0.00600069867, 0.0200179144, 103.560653}, true));
        add(makeSellmeier("N-SK16", {1.34317774, 0.241144399, 0.994317969}, {0.00704687339, 0.0229005, 92.7508526}, true));
        add(makeSellmeier("N-BAF10", {1.5851495, 0.143559385, 1.08521269}, {0.00926681282, 0.0424489805, 105.613573}, true));
        add(makeSellmeier("F2", {1.34533359, 0.209073176, 0.937357162}, {0.00997743871, 0.0470450767, 111.886764}, true));
        add(makeSellmeier("N-F2", {1.39757037, 0.159201403, 1.2686543}, {0.00995906143, 0.0546931752, 119.248346}, true));
        add(makeSellmeier("N-SF6", {1.77931763, 0.338149866, 2.08734474}, {0.0133714182, 0.0617533621, 174.01759}, true));
        add(makeSellmeier("N-SF11", {1.73759695, 0.313747346, 1.89878101}, {0.013188707, 0.0623068142, 155.23629}, true));
        // Malitson (1965) fused silica.
        add(makeSellmeier("fused-silica", {0.6961663, 0.4079426, 0.8974794}, squares({0.0684043, 0.1162414, 9.896161})));
        // Malitson (1963) calcium fluoride.
        add(makeSellmeier("CaF2", {0.5675888, 0.4710914, 3.8484723}, squares({0.050263605, 0.1003909, 34.649040})));
        // Malitson & Dodge (1972) sapphire, ordinary ray.
        add(makeSellmeier("sapphire", {1.4313493, 0.65054713, 5.3414021}, squares({0.0726631, 0.1193242, 18.028251})));
        // Peter (1923) diamond.
        add(makeSellmeier("diamond", {0.3306, 4.3356}, squares({0.1750, 0.1060})));
        // Sultanova et al. (2009) PMMA.
        add(makeSellmeier("PMMA", {1.1819}, {0.011313}));
        {
            // Daimon & Masumura (2007), 20 °C; absorption after Pope & Fry (1997).
            Medium m = makeSellmeier("water", {5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1},
                                     {5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1});
            m.absorption = Spectrum::tabulated({380, 400, 450, 500, 550, 600, 650, 700, 750, 800},
                                               {0.0114, 0.00663, 0.0092, 0.0204, 0.0565, 0.2224, 0.34, 0.65, 2.47, 2.07});
            add(m);
        }
        {
            Medium m; m.name = "crown"; m.index = IndexModel::cauchy(1.5046, 0.00420); add(m);
        }
        {
            Medium m; m.name = "flint"; m.index = IndexModel::cauchy(1.6200, 0.01020); add(m);
        }
        return c;
    }();
    return cat;
}

const std::map<std::string, ConductorSpectrum>& conductors() {
    static const std::map<std::string, ConductorSpectrum> cat = [] {
        // Approximate room-temperature optical constants (after Rakić 1995 and
        // Johnson & Christy 1972), coarsely tabulated.
        std::map<std::string, ConductorSpectrum> c;
        auto add = [&](const std::string& name, std::vector<double> l, std::vector<double> n, std::vector<double> k) {
            c[name] = ConductorSpectrum{name, Spectrum::tabulated(l, n), Spectrum::tabulated(l, k)};
        };
        add("aluminium", {400, 450, 500, 550, 600, 650, 700, 750, 800},
            {0.49, 0.62, 0.77, 0.96, 1.20, 1.47, 1.83, 2.20, 2.80}, {4.86, 5.47, 6.08, 6.69, 7.26, 7.79, 8.31, 8.60, 8.45});
        add("silver", {400, 450, 500, 550, 600, 650, 700, 800},
            {0.05, 0.04, 0.05, 0.06, 0.06, 0.05, 0.04, 0.04}, {2.10, 2.65, 3.09, 3.59, 4.01, 4.45, 4.84, 5.60});
        add("gold", {400, 450, 500, 550, 600, 650, 700, 800},
            {1.47, 1.40, 0.97, 0.43, 0.25, 0.17, 0.16, 0.15}, {1.95, 1.88, 1.87, 2.45, 3.00, 3.55, 3.95, 4.90});
        add("copper", {400, 450, 500, 550, 600, 650, 700, 800},
            {1.18, 1.17, 1.12, 1.00, 0.27, 0.21, 0.21, 0.23}, {2.21, 2.40, 2.60, 2.58, 3.40, 3.70, 4.20, 4.90});
        c["aluminum"] = c["aluminium"];
        return c;
    }();
    return cat;
}

}  // namespace

bool catalogMedium(const std::string& name, Medium& out) {
    auto it = catalog().find(name);
    if (it == catalog().end()) return false;
    out = it->second;
    return true;
}

std::vector<std::string> catalogMediumNames() {
    std::vector<std::string> r;
    for (auto& [k, v] : catalog()) r.push_back(k);
    return r;
}

bool catalogConductor(const std::string& name, ConductorSpectrum& out) {
    auto it = conductors().find(name);
    if (it == conductors().end()) return false;
    out = it->second;
    return true;
}

std::vector<std::string> catalogConductorNames() {
    std::vector<std::string> r;
    for (auto& [k, v] : conductors()) r.push_back(k);
    return r;
}

}  // namespace owe
