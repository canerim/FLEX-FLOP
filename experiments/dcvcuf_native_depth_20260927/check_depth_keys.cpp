#include "depth_keys.h"
#include <cassert>
#include <iostream>
#include <map>

using State = std::map<std::string, int>;
State make_state(int depth)
{
    State state;
    for (int i = 0; i <= depth; ++i) {
        state["dec_1." + std::to_string(i) + ".weight"] = i;
        state["dec_1." + std::to_string(i) + ".bias"] = i;
    }
    state["dec_2.weight"] = 0;
    return state;
}
void rejects(const State& state)
{
    try { (void)flex_depth::infer_synthesis_depth(state); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Malformed state was accepted");
}
int main()
{
    int accepted = 0, rejected = 0;
    for (int depth : {2, 4, 6, 8, 10, 12}) {
        assert(flex_depth::infer_synthesis_depth(make_state(depth)) == depth);
        ++accepted;
        for (int missing = 0; missing < depth; ++missing) {
            auto state = make_state(depth);
            state.erase("dec_1." + std::to_string(missing) + ".weight");
            state.erase("dec_1." + std::to_string(missing) + ".bias");
            rejects(state); ++rejected;
        }
    }
    for (int depth : {0, 1, 3, 5, 7, 9, 11, 13}) { rejects(make_state(depth)); ++rejected; }
    for (const char* key : {"dec_1.02.weight", "dec_1.-1.weight", "dec_1.2x.weight",
                                  "dec_1..weight", "dec_1.2.", "dec_1.2", "dec_1.999999.weight"}) {
        auto state = make_state(2); state[key] = 0; rejects(state); ++rejected;
    }
    rejects(State{}); ++rejected;
    std::cout << "{\"valid_depths\":" << accepted << ",\"rejected_states\":" << rejected
              << ",\"cuda_used\":false}" << std::endl;
}
