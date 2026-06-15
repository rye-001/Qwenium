#include "token_sequence_section.h"

#include <stdexcept>
#include <string>

#include "snapshot_io.h"

namespace qinf::state {

void TokenSequenceSection::write(qinf::session::SnapshotWriter& w) const {
    if (gen_start_ > tokens_.size()) {
        throw std::runtime_error(
            std::string("token_sequence: field \"gen_start\" expected <= ") +
            std::to_string(tokens_.size()) + " (token count), got " +
            std::to_string(gen_start_));
    }
    w.put_u32(gen_start_);
    w.put_u32(static_cast<uint32_t>(tokens_.size()));
    for (int32_t t : tokens_) {
        w.put_u32(static_cast<uint32_t>(t));
    }
}

void TokenSequenceSection::read(qinf::session::SnapshotReader& r) {
    gen_start_ = r.get_u32();
    uint32_t n = r.get_u32();
    tokens_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        tokens_[i] = static_cast<int32_t>(r.get_u32());
    }
    if (gen_start_ > n) {
        throw std::runtime_error(
            std::string("token_sequence: field \"gen_start\" expected <= ") +
            std::to_string(n) + " (restored token count), got " +
            std::to_string(gen_start_));
    }
}

}  // namespace qinf::state
