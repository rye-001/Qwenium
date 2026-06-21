#pragma once
// section_ids.h — central allocation of stable SnapshotSection tags.
//
// Each SnapshotSection's id() must be globally unique so the manifest can
// validate the on-stream id against the registered section (session_manifest).
// Allocating them in one place is part of the snapshot *contract* — it prevents
// two modules in different directories from silently colliding. The constants
// are 4-char little-endian tags purely for greppability in a hex dump.
//
// Adding a section type = add one line here. Never reuse a retired id.

#include "snapshot_section.h"

namespace qinf::session {

// Control lane.
constexpr SectionId kTokenSequenceSectionId = 0x4B4F5453u;  // "STOK"
constexpr SectionId kSamplerStateSectionId  = 0x474E5253u;  // "SRNG"
constexpr SectionId kGrammarCursorSectionId = 0x4D524753u;  // "SGRM"

// AppendKV lane (L2).
constexpr SectionId kKvCacheSectionId       = 0x5643564Bu;  // "KVCV"
// OverwriteRecurrent lane (L2). Authoritative — L2 skips prefill, so recurrent
// state is stored, not feed_tokens-rebuilt.
constexpr SectionId kDeltaNetStateSectionId = 0x4E544C44u;  // "DLTN"

// Image-embedding artifact (vision V1). Input data, not append/overwrite state,
// so lane Control. The context-free encoder output cached to skip the ViT pass.
constexpr SectionId kImageEmbeddingSectionId = 0x474D4D49u;  // "IMMG"

}  // namespace qinf::session
