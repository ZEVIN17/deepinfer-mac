#pragma once
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("TORCH_CHECK"); } while (0)
struct Sequence {
    struct {
        std::vector<int> draft_tokens;
        std::vector<float> accept_rates, prev_accept_rates;
        bool has_draft_info = false;
        int process_cnt = 0, schedule_cnt = 0;
    } mtp;
};
