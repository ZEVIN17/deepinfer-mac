// Test-only host boundary for the unchanged Apache-2.0 DeepInfer header.
#include <iostream>
#include <iomanip>
#include "vendor/mtp_varlen.hpp"
int main() {
    std::cout << std::setprecision(17);
    SamplerRuntimeEstimator sampler;
    ForwardRuntimeEstimator forward;
    MTPVarlen scheduler;
    char command;
    while (std::cin >> command) {
        if (command == 'S') {
            double value; std::cin >> value; sampler.update(value);
            auto result = sampler.get();
            if (result) std::cout << *result << '\n'; else std::cout << "null\n";
        } else if (command == 'F') {
            int graph; double value; std::cin >> graph >> value;
            forward.update(graph, value); std::cout << forward.get() << '\n';
        } else if (command == 'C') {
            int count; std::cin >> count;
            std::vector<std::pair<int, float>> curve;
            for (int i = 0; i < count; ++i) { int graph; float ms; std::cin >> graph >> ms; curve.emplace_back(graph, ms); }
            scheduler.set_decode_perf_curve(curve);
        } else if (command == 'R') {
            int n, graph; double ms, sample; std::cin >> n >> graph >> ms >> sample;
            scheduler.update_runtime(n, graph, ms, sample);
        } else if (command == 'Q') {
            int count; std::cin >> count;
            std::vector<std::shared_ptr<Sequence>> states;
            for (int i = 0; i < count; ++i) {
                auto s = std::make_shared<Sequence>(); int nd, na, np;
                std::cin >> nd >> s->mtp.has_draft_info >> s->mtp.process_cnt >> s->mtp.schedule_cnt;
                s->mtp.draft_tokens.resize(nd);
                std::cin >> na; s->mtp.accept_rates.resize(na);
                for (auto &rate : s->mtp.accept_rates) std::cin >> rate;
                std::cin >> np; s->mtp.prev_accept_rates.resize(np);
                for (auto &rate : s->mtp.prev_accept_rates) std::cin >> rate;
                states.push_back(s);
            }
            std::vector<int> lengths; scheduler.schedule(states, lengths);
            for (int i = 0; i < count; ++i) std::cout << (i ? " " : "") << lengths[i];
            std::cout << '\n';
        }
    }
}
