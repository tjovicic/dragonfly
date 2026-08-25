// Copyright 2022, Roman Gershman.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "src/core/detail/bitpacking.h"

#include <absl/base/internal/endian.h>

#if defined(__x86_64__) && (defined(__AVX2__) || defined(__BMI2__))
#include <immintrin.h>
#endif

#include "base/logging.h"
#include "core/sse_port.h"

using namespace std;

namespace dfly {

namespace detail {

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("Ofast")
#endif

static inline uint64_t Compress8x7bit(uint64_t x) {
  x = ((x & 0x7F007F007F007F00) >> 1) | (x & 0x007F007F007F007F);
  x = ((x & 0x3FFF00003FFF0000) >> 2) | (x & 0x00003FFF00003FFF);
  x = ((x & 0x0FFFFFFF00000000) >> 4) | (x & 0x000000000FFFFFFF);

  return x;
}

#if defined(__SSE3__) || defined(__aarch64__)
static inline pair<const char*, uint8_t*> simd_variant1_pack(const char* ascii, const char* end,
                                                             uint8_t* bin) {
  __m128i val, rpart, lpart;

  // Skips 8th byte (indexc 7) in the lower 8-byte part.
  const __m128i control = _mm_set_epi8(-1, -1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1, 0);

  // Based on the question I asked here: https://stackoverflow.com/q/74831843/2280111
  while (ascii <= end) {
    val = mm_loadu_si128(reinterpret_cast<const __m128i*>(ascii));

    /*
    x = ((x & 0x7F007F007F007F00) >> 1) | (x & 0x007F007F007F007F);
    x = ((x & 0x3FFF00003FFF0000) >> 2) | (x & 0x00003FFF00003FFF);
    x = ((x & 0x0FFFFFFF00000000) >> 4) | (x & 0x000000000FFFFFFF);
    */

    rpart = _mm_and_si128(val, _mm_set1_epi64x(0x007F007F007F007F));
    lpart = _mm_and_si128(val, _mm_set1_epi64x(0x7F007F007F007F00));
    val = _mm_or_si128(_mm_srli_epi64(lpart, 1), rpart);

    rpart = _mm_and_si128(val, _mm_set1_epi64x(0x00003FFF00003FFF));
    lpart = _mm_and_si128(val, _mm_set1_epi64x(0x3FFF00003FFF0000));
    val = _mm_or_si128(_mm_srli_epi64(lpart, 2), rpart);

    rpart = _mm_and_si128(val, _mm_set1_epi64x(0x000000000FFFFFFF));
    lpart = _mm_and_si128(val, _mm_set1_epi64x(0x0FFFFFFF00000000));
    val = _mm_or_si128(_mm_srli_epi64(lpart, 4), rpart);

    val = _mm_shuffle_epi8(val, control);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bin), val);
    bin += 14;
    ascii += 16;
  }

  return make_pair(ascii, bin);
}

static inline pair<const char*, uint8_t*> simd_variant2_pack(const char* ascii, const char* end,
                                                             uint8_t* bin) {
  // Skips 8th byte (indexc 7) in the lower 8-byte part.
  const __m128i control = _mm_set_epi8(-1, -1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1, 0);

  __m128i val, rpart, lpart;

  // Based on the question I asked here: https://stackoverflow.com/q/74831843/2280111
  while (ascii <= end) {
    val = mm_loadu_si128(reinterpret_cast<const __m128i*>(ascii));

    /*
    x = ((x & 0x7F007F007F007F00) >> 1) | (x & 0x007F007F007F007F);
    x = ((x & 0x3FFF00003FFF0000) >> 2) | (x & 0x00003FFF00003FFF);
    x = ((x & 0x0FFFFFFF00000000) >> 4) | (x & 0x000000000FFFFFFF);
    */
    val = _mm_maddubs_epi16(_mm_set1_epi16(0x8001), val);
    val = _mm_madd_epi16(_mm_set1_epi32(0x40000001), val);

    rpart = _mm_and_si128(val, _mm_set1_epi64x(0x000000000FFFFFFF));
    lpart = _mm_and_si128(val, _mm_set1_epi64x(0x0FFFFFFF00000000));
    val = _mm_or_si128(_mm_srli_epi64(lpart, 4), rpart);

    val = _mm_shuffle_epi8(val, control);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bin), val);
    bin += 14;
    ascii += 16;
  }
  return make_pair(ascii, bin);
}

#endif

// Daniel Lemire's function validate_ascii_fast() - under Apache/MIT license.
// See https://github.com/lemire/fastvalidate-utf-8/
// The function returns true (1) if all chars passed in src are
// 7-bit values (0x00..0x7F). Otherwise, it returns false (0).
#ifdef __s390x__
bool validate_ascii_fast(const char* src, size_t len) {
  size_t i = 0;

  // Initialize a vector in which all the elements are set to zero.
  vector unsigned char has_error = vec_splat_s8(0);
  if (len >= 16) {
    for (; i <= len - 16; i += 16) {
      // Load 16 bytes from buffer into a vector.
      vector unsigned char current_bytes = vec_load_len((signed char*)(src + i), 16);
      // Perform a bitwise OR operation between the current and the previously loaded contents.
      has_error = vec_orc(has_error, current_bytes);
    }
  }

  // Initialize a vector in which all the elements are set to an invalid ASCII value.
  vector unsigned char rep_invalid_values = vec_splat_s8(0x80);

  // Perform bitwise AND-complement operation between two vectors.
  vector unsigned char andc_result = vec_andc(rep_invalid_values, has_error);

  // Tests whether any of corresponding elements of the given vectors are not equal.
  // After the bitwise operation, both vectors should be equal if ASCII values.
  if (!vec_all_eq(rep_invalid_values, andc_result)) {
    return false;
  }

  for (; i < len; i++) {
    if (src[i] & 0x80) {
      return false;
    }
  }

  return true;
}
#else
bool validate_ascii_fast(const char* src, size_t len) {
  size_t i = 0;
  __m128i has_error = _mm_setzero_si128();
  if (len >= 16) {
    for (; i <= len - 16; i += 16) {
      __m128i current_bytes = mm_loadu_si128((const __m128i*)(src + i));
      has_error = _mm_or_si128(has_error, current_bytes);
    }
  }
  int error_mask = _mm_movemask_epi8(has_error);

  char tail_has_error = 0;
  for (; i < len; i++) {
    tail_has_error |= src[i];
  }
  error_mask |= (tail_has_error & 0x80);

  return !error_mask;
}
#endif

// len must be at least 16
void ascii_pack(const char* ascii, size_t len, uint8_t* bin) {
  uint64_t val;
  const char* end = ascii + len;

  while (ascii + 8 <= end) {
    val = absl::little_endian::Load64(ascii);
    uint64_t dest = (val & 0xFF);
    for (unsigned i = 1; i <= 7; ++i) {
      val >>= 1;
      dest |= (val & (0x7FUL << 7 * i));
    }
    memcpy(bin, &dest, 7);
    bin += 7;
    ascii += 8;
  }

  // epilog - we do not pack since we have less than 8 bytes.
  while (ascii < end) {
    *bin++ = *ascii++;
  }
}

void ascii_pack2(const char* ascii, size_t len, uint8_t* bin) {
  uint64_t val;
  const char* end = ascii + len;

  while (ascii + 8 <= end) {
    val = absl::little_endian::Load64(ascii);
    val = Compress8x7bit(val);
    memcpy(bin, &val, 7);
    bin += 7;
    ascii += 8;
  }

  // epilog - we do not pack since we have less than 8 bytes.
  while (ascii < end) {
    *bin++ = *ascii++;
  }
}

static inline uint64_t TryPackCompress8x7bit(uint64_t val) {
#if defined(__BMI2__) && defined(__x86_64__)
  return _pext_u64(val, 0x7F7F7F7F7F7F7F7FULL);
#else
  return Compress8x7bit(val);
#endif
}

static inline uint64_t FastExpand8x7bit(uint64_t val) {
#if defined(__BMI2__) && defined(__x86_64__)
  return _pdep_u64(val, 0x7F7F7F7F7F7F7F7FULL);
#else
  val = ((val & 0x00FFFFFFF0000000) << 4) | (val & 0x000000000FFFFFFF);
  val = ((val & 0xFFFFC000FFFFC000) << 2) | (val & 0x00003FFF00003FFF);
  return ((val & 0x7F807F807F807F80) << 1) | (val & 0x007F007F007F007F);
#endif
}

// Copies less than 8 bytes with at most two overlapping loads and stores. Returns the OR of all
// copied bytes so callers can test them for the high bit. Loads precede stores, so `src` and `dest`
// may overlap as long as `dest` is not above `src`.
static inline uint64_t CopyTailBytes(const uint8_t* src, size_t len, uint8_t* dest) {
  if (len >= 4) {
    const uint32_t head = absl::little_endian::Load32(src);
    const uint32_t tail = absl::little_endian::Load32(src + len - 4);
    absl::little_endian::Store32(dest, head);
    absl::little_endian::Store32(dest + len - 4, tail);
    return head | tail;
  }

  if (len >= 2) {
    const uint16_t head = absl::little_endian::Load16(src);
    const uint16_t tail = absl::little_endian::Load16(src + len - 2);
    absl::little_endian::Store16(dest, head);
    absl::little_endian::Store16(dest + len - 2, tail);
    return head | tail;
  }

  if (len == 1) {
    dest[0] = src[0];
    return src[0];
  }

  return 0;
}

constexpr uint64_t kAsciiHighBits = 0x8080808080808080ULL;

// Packs the bytes that are left once no full 128-bit block fits. Returns bits that are set only
// when a non-ascii byte was seen.
static inline __attribute__((always_inline)) uint64_t AsciiTryPackScalarTail(const uint8_t* src,
                                                                             size_t remaining,
                                                                             uint8_t* out) {
  uint64_t has_error = 0;

  while (remaining >= 9) {  // a single 8-byte store, the 8th byte belongs to the next group
    const uint64_t val = absl::little_endian::Load64(src);
    has_error |= val;
    absl::little_endian::Store64(out, TryPackCompress8x7bit(val));
    src += 8;
    out += 7;
    remaining -= 8;
  }

  if (remaining == 8) {
    uint64_t val = absl::little_endian::Load64(src);
    has_error |= val;
    val = TryPackCompress8x7bit(val);
#if defined(ABSL_IS_BIG_ENDIAN)
    val = __builtin_bswap64(val);
#endif
    memcpy(out, &val, 7);
  } else if (remaining) {
    has_error |= CopyTailBytes(src, remaining, out);  // the final partial group stays verbatim
  }

  return has_error;
}

#if defined(__SSE3__) || defined(__aarch64__)
constexpr size_t kAsciiPerTryPackSimdBlock = 16;
constexpr size_t kPackedPerTryPackSimdBlock = 14;
constexpr size_t kTryPackSimdStoreOverlap = 2;
#if defined(__aarch64__)
constexpr size_t kTryPackSimdBulkMin = 2 * kAsciiPerTryPackSimdBlock;
#else
constexpr size_t kTryPackSimdBulkMin = kAsciiPerTryPackSimdBlock + kTryPackSimdStoreOverlap;
#endif

static inline __m128i AsciiTryPackSimdStages(__m128i val) {
#if defined(__aarch64__)
  // Valid 7-bit lanes let SLI fold adjacent fields without masks.
  uint16x8_t v16 = vreinterpretq_u16_m128i(val);
  v16 = vsliq_n_u16(v16, vshrq_n_u16(v16, 8), 7);
  uint32x4_t v32 = vreinterpretq_u32_u16(v16);
  v32 = vsliq_n_u32(v32, vshrq_n_u32(v32, 16), 14);
  uint64x2_t v64 = vreinterpretq_u64_u32(v32);
  v64 = vsliq_n_u64(v64, vshrq_n_u64(v64, 32), 28);
  val = vreinterpretq_m128i_u64(v64);
#else
  val = _mm_maddubs_epi16(_mm_set1_epi16(0x8001), val);
  val = _mm_madd_epi16(_mm_set1_epi32(0x40000001), val);

  const __m128i rpart = _mm_and_si128(val, _mm_set1_epi64x(0x000000000FFFFFFF));
  const __m128i lpart = _mm_and_si128(val, _mm_set1_epi64x(0x0FFFFFFF00000000));
  val = _mm_or_si128(_mm_srli_epi64(lpart, 4), rpart);
#endif

  return val;
}

static inline __m128i AsciiTryPackSimd(__m128i val) {
  const __m128i control = _mm_set_epi8(-1, -1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1, 0);
  return _mm_shuffle_epi8(AsciiTryPackSimdStages(val), control);
}

static inline __attribute__((always_inline)) uint64_t AsciiTryPackSimdBulk(const uint8_t* src,
                                                                           size_t remaining,
                                                                           uint8_t* out) {
  uint64_t has_error = 0;

  if (remaining >= kTryPackSimdBulkMin) {
    __m128i simd_error = _mm_setzero_si128();
    // Each block stores 16 bytes although only 14 of them are packed data: the next block or the
    // tail always overwrites the two extra bytes, which is cheaper than an exact 14-byte store.
    do {
      const __m128i val = mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
      simd_error = _mm_or_si128(simd_error, val);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out), AsciiTryPackSimd(val));
      src += kAsciiPerTryPackSimdBlock;
      out += kPackedPerTryPackSimdBlock;
      remaining -= kAsciiPerTryPackSimdBlock;
    } while (remaining >= kAsciiPerTryPackSimdBlock + kTryPackSimdStoreOverlap);

#if defined(__aarch64__)
    has_error = (vmaxvq_u8(vreinterpretq_u8_m128i(simd_error)) & 0x80) ? kAsciiHighBits : 0;
#else
    has_error = _mm_movemask_epi8(simd_error) ? kAsciiHighBits : 0;
#endif
  }

  // Fewer than kTryPackSimdBulkMin bytes are left, for which the scalar tail is faster.
  return has_error | AsciiTryPackScalarTail(src, remaining, out);
}

#if defined(__AVX2__) && defined(__x86_64__)
constexpr size_t kAsciiPerTryPackAvx2Block = 2 * kAsciiPerTryPackSimdBlock;
constexpr size_t kPackedPerTryPackAvx2Block = 2 * kPackedPerTryPackSimdBlock;

// The 256-bit path has to live in its own function (see AsciiTryPackAvx2Long), so it only pays off
// once its blocks save more than the call and the setup of the wider constants cost.
constexpr size_t kTryPackAvx2Min = 3 * kAsciiPerTryPackAvx2Block;

// Up to this length a single 128-bit block plus the scalar tail covers the whole input.
constexpr size_t kTryPackSimdShortMax = kAsciiPerTryPackSimdBlock + kTryPackSimdBulkMin - 1;

static inline __m256i AsciiTryPackAvx2(__m256i val) {
  val = _mm256_maddubs_epi16(_mm256_set1_epi16(0x8001), val);
  val = _mm256_madd_epi16(_mm256_set1_epi32(0x40000001), val);

  const __m256i rpart = _mm256_and_si256(val, _mm256_set1_epi64x(0x000000000FFFFFFF));
  const __m256i lpart = _mm256_and_si256(val, _mm256_set1_epi64x(0x0FFFFFFF00000000));
  val = _mm256_or_si256(_mm256_srli_epi64(lpart, 4), rpart);

  const __m128i control = _mm_set_epi8(-1, -1, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1, 0);
  return _mm256_shuffle_epi8(val, _mm256_broadcastsi128_si256(control));
}

// Stores the 28 packed bytes of a block without touching anything past them.
static inline void AsciiTryPackAvx2Store28(uint8_t* dest, __m256i packed) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dest), _mm256_castsi256_si128(packed));
  const __m128i high = _mm256_extracti128_si256(packed, 1);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dest + kPackedPerTryPackSimdBlock), high);
  const uint64_t rest = _mm_cvtsi128_si64(_mm_srli_si128(high, 8));
  absl::little_endian::Store32(dest + kPackedPerTryPackSimdBlock + 8, static_cast<uint32_t>(rest));
  absl::little_endian::Store16(dest + kPackedPerTryPackSimdBlock + 12,
                               static_cast<uint16_t>(rest >> 32));
}

// Packs at most kTryPackSimdShortMax bytes: one 128-bit block at most, then the tail. Deliberately
// loop free, a loop here costs both its caller and this path extra induction variables.
static inline __attribute__((always_inline)) uint64_t AsciiTryPackSimdShort(const uint8_t* src,
                                                                            size_t remaining,
                                                                            uint8_t* out) {
  uint64_t has_error = 0;

  if (remaining >= kTryPackSimdBulkMin) {
    const __m128i val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    has_error = _mm_movemask_epi8(val) ? kAsciiHighBits : 0;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), AsciiTryPackSimd(val));
    src += kAsciiPerTryPackSimdBlock;
    out += kPackedPerTryPackSimdBlock;
    remaining -= kAsciiPerTryPackSimdBlock;
  }

  return has_error | AsciiTryPackScalarTail(src, remaining, out);
}

// Packs at least kTryPackAvx2Min bytes. Kept out of line on purpose: 256-bit code inside
// ascii_try_pack would force a 32-byte aligned frame on it and slow down every short string.
static __attribute__((noinline)) uint64_t AsciiTryPackAvx2Long(const uint8_t* src, size_t remaining,
                                                               uint8_t* out) {
  // Two full 16-byte stores are cheaper than an exact 28-byte store, and the two extra bytes they
  // write are always overwritten by the next block or by the tail below.
  const size_t bulk_blocks = (remaining - kTryPackSimdStoreOverlap) / kAsciiPerTryPackAvx2Block;
  __m256i simd_error = _mm256_setzero_si256();

  // A counted loop, so that its body stays free of induction variables the tail below needs.
  for (size_t i = 0; i < bulk_blocks; ++i) {
    __m256i val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    simd_error = _mm256_or_si256(simd_error, val);
    val = AsciiTryPackAvx2(val);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm256_castsi256_si128(val));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + kPackedPerTryPackSimdBlock),
                     _mm256_extracti128_si256(val, 1));
    src += kAsciiPerTryPackAvx2Block;
    out += kPackedPerTryPackAvx2Block;
  }
  remaining -= bulk_blocks * kAsciiPerTryPackAvx2Block;

  // Consume the accumulator before the last block, otherwise it stays live across the loop above.
  uint64_t has_error = _mm256_movemask_epi8(simd_error) ? kAsciiHighBits : 0;

  if (remaining >= kAsciiPerTryPackAvx2Block) {  // nothing follows, store exactly 28 bytes
    const __m256i val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    has_error |= _mm256_movemask_epi8(val) ? kAsciiHighBits : 0;
    AsciiTryPackAvx2Store28(out, AsciiTryPackAvx2(val));
    src += kAsciiPerTryPackAvx2Block;
    out += kPackedPerTryPackAvx2Block;
    remaining -= kAsciiPerTryPackAvx2Block;
  }

  return has_error | AsciiTryPackSimdShort(src, remaining, out);
}
#endif
#endif

size_t ascii_try_pack(const char* ascii, size_t len, uint8_t* bin) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(ascii);
  uint64_t has_error;

#if defined(__AVX2__) && defined(__x86_64__)
  if (len > kTryPackSimdShortMax)
    has_error = len >= kTryPackAvx2Min ? AsciiTryPackAvx2Long(src, len, bin)
                                       : AsciiTryPackSimdBulk(src, len, bin);
  else
    has_error = AsciiTryPackSimdShort(src, len, bin);
#elif defined(__SSE3__) || defined(__aarch64__)
  has_error = AsciiTryPackSimdBulk(src, len, bin);
#else
  has_error = AsciiTryPackScalarTail(src, len, bin);
#endif

  return (has_error & kAsciiHighBits) ? 0 : binpacked_len(len);
}

static inline __attribute__((always_inline)) void AsciiUnpackFastScalar(const uint8_t* bin,
                                                                        size_t ascii_len,
                                                                        char* ascii) {
  constexpr uint64_t kSevenBytes = 0x00FFFFFFFFFFFFFFULL;

  while (ascii_len >= 9) {
    const uint64_t val = absl::little_endian::Load64(bin) & kSevenBytes;
    absl::little_endian::Store64(ascii, FastExpand8x7bit(val));
    bin += 7;
    ascii += 8;
    ascii_len -= 8;
  }

  if (ascii_len == 8) {
    // Two overlapping loads read exactly seven bytes.
    const uint64_t val = absl::little_endian::Load32(bin) |
                         (static_cast<uint64_t>(absl::little_endian::Load32(bin + 3)) << 24);
    absl::little_endian::Store64(ascii, FastExpand8x7bit(val));
    return;
  }

  CopyTailBytes(bin, ascii_len, reinterpret_cast<uint8_t*>(ascii));
}

#if defined(__AVX2__) && defined(__x86_64__)
constexpr size_t kAsciiPerAvx2Block = 32;
constexpr size_t kPackedPerAvx2Block = 28;

static inline __m256i AsciiUnpackAvx2(__m256i val) {
  const __m128i control128 = _mm_set_epi8(14, 13, 12, 11, 10, 9, 8, 7, -1, 6, 5, 4, 3, 2, 1, 0);
  val = _mm256_shuffle_epi8(val, _mm256_broadcastsi128_si256(control128));

  // Every stage splits each field into two halves of equal width. The blend keeps the lower half
  // in place, so only the shifted copy needs selecting, and the bits that shift in from below are
  // dropped by the final mask.
  val = _mm256_blend_epi32(val, _mm256_slli_epi64(val, 4), 0xAA);
  val = _mm256_blend_epi16(val, _mm256_slli_epi32(val, 2), 0xAA);
  val = _mm256_blendv_epi8(val, _mm256_slli_epi16(val, 1),
                           _mm256_set1_epi16(static_cast<int16_t>(0xFF00)));
  return _mm256_and_si256(val, _mm256_set1_epi8(0x7F));
}

__attribute__((noinline)) static void AsciiUnpackAvx2Blocks(const uint8_t* bin, size_t ascii_len,
                                                            char* ascii) {
  while (ascii_len >= kAsciiPerAvx2Block + 2) {
    const __m128i low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bin));
    const __m128i high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bin + 14));
    __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ascii), AsciiUnpackAvx2(val));
    bin += kPackedPerAvx2Block;
    ascii += kAsciiPerAvx2Block;
    ascii_len -= kAsciiPerAvx2Block;
  }
  if (ascii_len >= kAsciiPerAvx2Block) {
    const __m128i low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bin));
    const __m128i high =
        _mm_srli_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(bin + 12)), 2);
    __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ascii), AsciiUnpackAvx2(val));
    bin += kPackedPerAvx2Block;
    ascii += kAsciiPerAvx2Block;
    ascii_len -= kAsciiPerAvx2Block;
  }
  AsciiUnpackFastScalar(bin, ascii_len, ascii);
}

static inline void AsciiUnpackFastImpl(const uint8_t* bin, size_t ascii_len, char* ascii) {
  if (ascii_len >= kAsciiPerAvx2Block) {
    AsciiUnpackAvx2Blocks(bin, ascii_len, ascii);
  } else {
    AsciiUnpackFastScalar(bin, ascii_len, ascii);
  }
}
#elif defined(__BMI2__) && defined(__x86_64__)
static inline void AsciiUnpackFastImpl(const uint8_t* bin, size_t ascii_len, char* ascii) {
  AsciiUnpackFastScalar(bin, ascii_len, ascii);
}
#elif defined(__aarch64__)
static inline uint8x16_t AsciiUnpackNeonBlock(uint8x16_t val, uint8x16_t ascii_mask) {
  uint64x2_t v64 = vreinterpretq_u64_u8(val);
  v64 = vsliq_n_u64(v64, vshrq_n_u64(v64, 28), 32);
  uint32x4_t v32 = vreinterpretq_u32_u64(v64);
  v32 = vsliq_n_u32(v32, vshrq_n_u32(v32, 14), 16);
  uint16x8_t v16 = vreinterpretq_u16_u32(v32);
  v16 = vsliq_n_u16(v16, vshrq_n_u16(v16, 7), 8);
  return vandq_u8(vreinterpretq_u8_u16(v16), ascii_mask);
}

static inline void AsciiUnpackNeonTailBlock(const uint8_t* bin, char* ascii,
                                            uint8x16_t ascii_mask) {
  const uint8x16_t control = {0, 1, 2, 3, 4, 5, 6, 0xFF, 9, 10, 11, 12, 13, 14, 15, 0xFF};
  const uint8x16_t raw = vcombine_u8(vld1_u8(bin), vld1_u8(bin + 6));
  vst1q_u8(reinterpret_cast<uint8_t*>(ascii),
           AsciiUnpackNeonBlock(vqtbl1q_u8(raw, control), ascii_mask));
}

static inline void AsciiUnpackNeonBlocks(const uint8_t* bin, size_t ascii_len, char* ascii) {
  const uint8x16_t ascii_mask = vdupq_n_u8(0x7F);
  const uint8x16_t control = {0, 1, 2, 3, 4, 5, 6, 0xFF, 7, 8, 9, 10, 11, 12, 13, 0xFF};
  const size_t simd_len = ((ascii_len - 2) / 16) * 16;
  const char* end = ascii + simd_len;
  const char* paired_end = ascii + (simd_len & ~size_t{31});

  while (ascii < paired_end) {
    const uint8x16_t low = vqtbl1q_u8(vld1q_u8(bin), control);
    const uint8x16_t high = vqtbl1q_u8(vld1q_u8(bin + 14), control);
    uint8_t* dest = reinterpret_cast<uint8_t*>(ascii);
    vst1q_u8(dest, AsciiUnpackNeonBlock(low, ascii_mask));
    vst1q_u8(dest + 16, AsciiUnpackNeonBlock(high, ascii_mask));
    ascii += 32;
    bin += 28;
  }

  if (ascii < end) {
    vst1q_u8(reinterpret_cast<uint8_t*>(ascii),
             AsciiUnpackNeonBlock(vqtbl1q_u8(vld1q_u8(bin), control), ascii_mask));
    ascii += 16;
    bin += 14;
  }
  ascii_len -= simd_len;

  if (ascii_len >= 16) {
    AsciiUnpackNeonTailBlock(bin, ascii, ascii_mask);
    ascii += 16;
    bin += 14;
    ascii_len -= 16;
  }

  AsciiUnpackFastScalar(bin, ascii_len, ascii);
}

static inline void AsciiUnpackFastImpl(const uint8_t* bin, size_t ascii_len, char* ascii) {
  if (ascii_len < 16) {
    AsciiUnpackFastScalar(bin, ascii_len, ascii);
  } else if (ascii_len < 32) {
    AsciiUnpackNeonTailBlock(bin, ascii, vdupq_n_u8(0x7F));
    AsciiUnpackFastScalar(bin + 14, ascii_len - 16, ascii + 16);
  } else {
    AsciiUnpackNeonBlocks(bin, ascii_len, ascii);
  }
}
#else
static inline void AsciiUnpackFastImpl(const uint8_t* bin, size_t ascii_len, char* ascii) {
  AsciiUnpackFastScalar(bin, ascii_len, ascii);
}
#endif

void ascii_unpack_fast(const uint8_t* bin, size_t ascii_len, char* ascii) {
  AsciiUnpackFastImpl(bin, ascii_len, ascii);
}

// The algo - do in parallel what ascii_pack does on two uint64_t integers
void ascii_pack_simd(const char* ascii, size_t len, uint8_t* bin) {
#if defined(__SSE3__) || defined(__aarch64__)
  // I leave out 16 bytes in addition to 16 that we load in the loop
  // because we store into bin full 16 bytes instead of 14. To prevent data
  // overwrite we finish loop one iteration earlier.
  const char* end = ascii + len - 32;

  tie(ascii, bin) = simd_variant1_pack(ascii, end, bin);

  end += 32;  // Bring back end.
  DCHECK(ascii < end);
  ascii_pack(ascii, end - ascii, bin);
#else
  ascii_pack(ascii, len, bin);
#endif
}

void ascii_pack_simd2(const char* ascii, size_t len, uint8_t* bin) {
#if defined(__SSE3__) || defined(__aarch64__)
  // I leave out 16 bytes in addition to 16 that we load in the loop
  // because we store into bin full 16 bytes instead of 14. To prevent data
  // overwrite we finish loop one iteration earlier.
  const char* end = ascii + len - 32;

  // on arm var
#if defined(__aarch64__)
  tie(ascii, bin) = simd_variant1_pack(ascii, end, bin);
#else
  tie(ascii, bin) = simd_variant2_pack(ascii, end, bin);
#endif

  end += 32;  // Bring back end.
  DCHECK(ascii < end);
  ascii_pack(ascii, end - ascii, bin);
#else
  ascii_pack(ascii, len, bin);
#endif
}

// unpacks 8->7 encoded blob back to ascii.
// generally, we can not unpack inplace because ascii (dest) buffer is 8/7 bigger than
// the source buffer.
// however, if binary data is positioned on the right of the ascii buffer with empty space on the
// left than we can unpack inplace.
void ascii_unpack(const uint8_t* bin, size_t ascii_len, char* ascii) {
  constexpr uint8_t kM = 0x7F;
  uint8_t p = 0;
  unsigned i = 0;

  while (ascii_len >= 8) {
    for (i = 0; i < 7; ++i) {
      uint8_t src = *bin;  // keep on stack in case we unpack inplace.
      *ascii++ = (p >> (8 - i)) | ((src << i) & kM);
      p = src;
      ++bin;
    }

    ascii_len -= 8;
    *ascii++ = p >> 1;
  }

  DCHECK_LT(ascii_len, 8u);
  for (i = 0; i < ascii_len; ++i) {
    *ascii++ = *bin++;
  }
}

uint8_t ascii_unpack_byte(const uint8_t* bin, size_t ascii_len, size_t idx) {
  DCHECK(idx < ascii_len) << "Index oob for ascii byte unpacking: " << idx << " >= " << ascii_len;
  const size_t packed_groups = ascii_len / 8;
  const size_t group = idx / 8;
  const size_t idx_in_group = idx % 8;

  // Tail bytes (after the last full 8-char group) are stored unpacked.
  if (group >= packed_groups) {
    return bin[packed_groups * 7 + idx_in_group];
  }

  // Unpack ascii group and return byte at idx.
  char buf[8];
  ascii_unpack(bin + group * 7, 8, buf);
  return buf[idx_in_group];
}

void ascii_pack_byte(uint8_t* bin, size_t ascii_len, size_t idx, uint8_t val) {
  DCHECK(idx < ascii_len) << "Index oob for ascii byte packing: " << idx << " >= " << ascii_len;
  DCHECK_LT(val, 128u) << "Only 7-bit ASCII values can be packed";

  const size_t packed_groups = ascii_len / 8;
  const size_t group = idx / 8;
  const size_t idx_in_group = idx % 8;

  // Tail bytes (after the last full 8-char group) are stored unpacked.
  if (group >= packed_groups) {
    bin[packed_groups * 7 + idx_in_group] = val;
    return;
  }

  // Unpack ascii group and return, modify byte at idx and pack back.
  uint8_t* group_bin = bin + group * 7;
  char buf[8];
  ascii_unpack(group_bin, 8, buf);
  buf[idx_in_group] = val;
  ascii_pack(buf, 8, group_bin);
}

// See CompactObjectTest.AsanTriggerReadOverflow for more details.
void ascii_unpack_simd(const uint8_t* bin, size_t ascii_len, char* ascii) {
#if defined(__SSE3__) || defined(__aarch64__)

  if (ascii_len < 18) {  // ascii_len >=18 means bin length >=16.
    ascii_unpack(bin, ascii_len, ascii);
    return;
  }

  __m128i val, rpart, lpart;

  // we read 16 bytes from bin even when we need only 14 bytes.
  // So for last iteration we may access 2 bytes outside of the bin buffer.
  // To prevent this we need to round down the length of the bin buffer but since we
  // limit by ascii_len we reduce the ascii_len by two before computing number of iterations.
  size_t simd_len = ((ascii_len - 2) / 16) * 16;
  const char* end = ascii + simd_len;

  // shifts the second 7-byte blob to the left.
  const __m128i control = _mm_set_epi8(14, 13, 12, 11, 10, 9, 8, 7, -1, 6, 5, 4, 3, 2, 1, 0);

  while (ascii < end) {
    val = mm_loadu_si128(reinterpret_cast<const __m128i*>(bin));
    val = _mm_shuffle_epi8(val, control);

    rpart = _mm_and_si128(val, _mm_set1_epi64x(0x000000000FFFFFFF));
    lpart = _mm_and_si128(val, _mm_set1_epi64x(0x00FFFFFFF0000000));
    val = _mm_or_si128(_mm_slli_epi64(lpart, 4), rpart);

    rpart = _mm_and_si128(val, _mm_set1_epi64x(0x00003FFF00003FFF));
    lpart = _mm_and_si128(val, _mm_set1_epi64x(0xFFFFC000FFFFC000));
    val = _mm_or_si128(_mm_slli_epi64(lpart, 2), rpart);

    rpart = _mm_and_si128(val, _mm_set1_epi64x(0x007F007F007F007F));
    lpart = _mm_and_si128(val, _mm_set1_epi64x(0x7F807F807F807F80));
    val = _mm_or_si128(_mm_slli_epi64(lpart, 1), rpart);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(ascii), val);
    ascii += 16;
    bin += 14;
  }

  ascii_len -= simd_len;
  if (ascii_len)
    ascii_unpack(bin, ascii_len, ascii);
#else
  ascii_unpack(bin, ascii_len, ascii);
#endif
}

// compares packed and unpacked strings. packed must be of length = binpacked_len(ascii_len).
bool compare_packed(const uint8_t* packed, const char* ascii, size_t ascii_len) {
  unsigned i = 0;
  bool res = true;
  const char* end = ascii + ascii_len;

  while (ascii + 8 <= end) {
    for (i = 0; i < 7; ++i) {
      uint8_t conv = (ascii[0] >> i) | (ascii[1] << (7 - i));
      res &= (conv == *packed);
      ++ascii;
      ++packed;
    }

    if (!res)
      return false;

    ++ascii;
  }

  while (ascii < end) {
    if (*ascii++ != *packed++) {
      return false;
    }
  }

  return true;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

}  // namespace detail

}  // namespace dfly
