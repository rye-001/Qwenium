#include "sampling_snapshot.h"

#include <stdexcept>
#include <string>

#include "snapshot_io.h"

namespace qwenium {

void SamplerStateSection::write(qinf::session::SnapshotWriter& w) const {
    // Record whether RNG state follows so restore can fail loud if the
    // receiver's sampler kind disagrees (e.g. greedy blob into a temperature
    // sampler, which would otherwise silently start from a fresh seed).
    const uint32_t has = sampler_.has_rng_state() ? 1u : 0u;
    w.put_u32(has);
    if (has) sampler_.write_rng_state(w);
}

void SamplerStateSection::read(qinf::session::SnapshotReader& r) {
    const uint32_t has = r.get_u32();
    const uint32_t expected = sampler_.has_rng_state() ? 1u : 0u;
    if (has != expected) {
        throw std::runtime_error(
            std::string("sampler_state: field \"has_rng_state\" expected ") +
            std::to_string(expected) + " (receiver sampler), got " +
            std::to_string(has) + " (snapshot)");
    }
    if (has) sampler_.read_rng_state(r);
}

}  // namespace qwenium
