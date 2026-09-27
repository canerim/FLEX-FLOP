// Strict state-dictionary depth inference for the pinned DCVC-UF decoder.
// This helper is independent of CUDA and does not validate tensor shapes.
#pragma once
#include <set>
#include <stdexcept>
#include <string>

namespace flex_depth {
template <class StateDict>
int infer_synthesis_depth(const StateDict& state)
{
    const std::string prefix = "dec_1.";
    std::set<int> blocks;
    for (const auto& item : state) {
        const auto& key = item.first;
        if (key.compare(0, prefix.size(), prefix) != 0) continue;
        const auto end = key.find('.', prefix.size());
        if (end == std::string::npos || end + 1 == key.size())
            throw std::invalid_argument("Malformed decoder parameter key: " + key);
        const auto token = key.substr(prefix.size(), end - prefix.size());
        if (token.empty() || token.size() > 2 || (token.size() > 1 && token[0] == '0'))
            throw std::invalid_argument("Noncanonical decoder block index: " + key);
        int index = 0;
        for (const auto c : token) {
            if (c < '0' || c > '9') throw std::invalid_argument("Invalid decoder block index: " + key);
            index = index * 10 + c - '0';
        }
        if (index > 12) throw std::invalid_argument("Decoder depth exceeds supported maximum");
        blocks.insert(index);
    }
    if (blocks.empty() || *blocks.begin() != 0)
        throw std::invalid_argument("Missing decoder upsampler at dec_1.0");
    const int depth = *blocks.rbegin();
    if (depth < 2 || depth > 12 || depth % 2 != 0)
        throw std::invalid_argument("Supported synthesis depths are 2/4/6/8/10/12");
    for (int index = 0; index <= depth; ++index)
        if (blocks.count(index) != 1) throw std::invalid_argument("Noncontiguous synthesis blocks");
    return depth;
}
}  // namespace flex_depth
