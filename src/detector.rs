//! Incremental repeated-content (loop) detector.
//!
//! For each candidate block length `L` in `[min_block, max_block]` the detector
//! maintains `run[L]` = the length of the longest suffix of the streamed text
//! that has period `L`. A loop is reported as soon as some `L` has
//! `run[L] >= min_repeats * L`, **provided** the trailing `L`-char block is
//! *primitive* (not itself a repetition of a smaller unit shorter than
//! `min_block`). The primitive check prevents e.g. `"AB" x 10` from being
//! misread as a 10-char block repeated twice.
//!
//! Complexity per character: `O(max_block - min_block + 1)` simple comparisons.
//! Memory: `O(max_block)` characters (bounded window) + `O(max_block)` counters.
//! Multilingual: operates on `char` (Unicode scalar values); block lengths are
//! measured in characters, so CJK / emoji count as one unit each.

use std::collections::VecDeque;

/// A detected loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopHit {
    /// Fundamental block length (characters).
    pub block_len: usize,
    /// Length of the trailing repeated region (characters) = `min_repeats * block_len`.
    pub run: usize,
    /// Number of leading characters to keep (prefix + one block). The proxy slices
    /// the accumulated text to this many characters as the deduplicated assistant
    /// content for the continuation request.
    pub kept_chars: usize,
}

/// Incremental loop detector. Feed characters as they arrive; a [`LoopHit`] is
/// returned on the first character that completes a loop.
pub struct LoopDetector {
    min_block: usize,
    max_block: usize,
    min_repeats: usize,
    /// Last `max_block` characters of the stream (bounded).
    window: VecDeque<char>,
    /// `run[i]` = longest period-`(min_block + i)` suffix length.
    run: Vec<usize>,
    /// Total characters fed so far.
    total: usize,
}

// `L` is the established notation for block length in this algorithm.
#[allow(non_snake_case)]
impl LoopDetector {
    pub fn new(min_block: usize, max_block: usize, min_repeats: usize) -> Self {
        assert!(min_block >= 1, "min_block must be >= 1");
        assert!(max_block >= min_block, "max_block must be >= min_block");
        assert!(min_repeats >= 1, "min_repeats must be >= 1");
        let n = max_block - min_block + 1;
        LoopDetector {
            min_block,
            max_block,
            min_repeats,
            window: VecDeque::with_capacity(max_block),
            run: vec![0; n],
            total: 0,
        }
    }

    /// Feed one character; return a [`LoopHit`] if a loop is completed at this point.
    pub fn feed(&mut self, c: char) -> Option<LoopHit> {
        let n = self.total; // 0-based index of the new character
        let start = self.total.saturating_sub(self.max_block);

        // Update run counters for active block lengths (L ascending; break once L > n).
        for (i, L) in (self.min_block..=self.max_block).enumerate() {
            if L > n {
                break;
            }
            let idx = (n - L) - start;
            let prev = self.run[i];
            let base = if prev == 0 { L } else { prev };
            self.run[i] = if c == self.window[idx] { base + 1 } else { L };
        }

        // Append the new character to the bounded window.
        self.window.push_back(c);
        if self.window.len() > self.max_block {
            self.window.pop_front();
        }
        self.total += 1;

        self.check_trigger()
    }

    /// Feed a whole string; return the first hit (if any).
    pub fn feed_str(&mut self, s: &str) -> Option<LoopHit> {
        for c in s.chars() {
            if let Some(hit) = self.feed(c) {
                return Some(hit);
            }
        }
        None
    }

    fn check_trigger(&self) -> Option<LoopHit> {
        for (i, L) in (self.min_block..=self.max_block).enumerate() {
            let r = self.run[i];
            if r >= self.min_repeats * L {
                // Smallest qualifying L. Verify the L-char block is primitive.
                if self.block_is_primitive(L) {
                    // Keep prefix + one block. `r <= total`, so this never underflows.
                    let kept_chars = self.total - r + L;
                    return Some(LoopHit {
                        block_len: L,
                        run: r,
                        kept_chars,
                    });
                }
                // The trailing block is a repetition of a smaller unit
                // (< min_block): a sub-threshold repetition we intentionally
                // ignore. We stop here (rather than scanning larger L) for
                // performance. This can miss a contrived case where a larger
                // block is the true primitive period, but such patterns are not
                // realistic model loops; it never produces a false positive.
                return None;
            }
        }
        None
    }

    /// True if the trailing `L`-char block has no period `p` in `1..min_block`.
    fn block_is_primitive(&self, L: usize) -> bool {
        let wlen = self.window.len();
        let base = wlen - L;
        for p in 1..self.min_block {
            if p >= L {
                break;
            }
            let mut has_period = true;
            for i in p..L {
                if self.window[base + i] != self.window[base + i - p] {
                    has_period = false;
                    break;
                }
            }
            if has_period {
                return false;
            }
        }
        true
    }
}

/// Repeat count for the long tier (fixed by design, not configurable).
pub const LONG_REPEATS: usize = 2;

/// Two-tier loop detector.
///
/// - **Short tier**: block lengths in `[short_min, short_max]`, triggering
///   after `short_repeats` consecutive repetitions. Catches tight, small
///   loops (e.g. a repeated sentence fragment).
/// - **Long tier**: block lengths in `[short_max + 1, long_max]`, triggering
///   after [`LONG_REPEATS`] (2) consecutive repetitions. Catches large-block
///   loops (e.g. a repeated paragraph or code section) where two copies are
///   already obviously a loop.
///
/// Both tiers are fed every character; a hit from either tier is returned
/// (the short tier wins when both fire on the same character, as it is the
/// tighter loop). The long tier is disabled entirely when
/// `long_max <= short_max`.
///
/// The long tier's primitive check rejects blocks whose fundamental period is
/// `<= short_max`, so a short-tier block repeated fewer than `short_repeats`
/// times is never misread as a long-tier loop.
pub struct TieredDetector {
    short: LoopDetector,
    long: Option<LoopDetector>,
}

impl TieredDetector {
    pub fn new(
        short_min: usize,
        short_max: usize,
        short_repeats: usize,
        long_max: usize,
    ) -> Self {
        assert!(short_min >= 1, "short_min must be >= 1");
        assert!(short_max >= short_min, "short_max must be >= short_min");
        assert!(short_repeats >= 1, "short_repeats must be >= 1");
        assert!(long_max >= short_min, "long_max must be >= short_min");
        let long = (long_max > short_max).then(|| {
            LoopDetector::new(short_max + 1, long_max, LONG_REPEATS)
        });
        TieredDetector {
            short: LoopDetector::new(short_min, short_max, short_repeats),
            long,
        }
    }

    /// Feed one character; return a [`LoopHit`] if either tier completes a loop.
    pub fn feed(&mut self, c: char) -> Option<LoopHit> {
        let short_hit = self.short.feed(c);
        let long_hit = self.long.as_mut().and_then(|d| d.feed(c));
        short_hit.or(long_hit)
    }

    /// Feed a whole string; return the first hit (if any).
    pub fn feed_str(&mut self, s: &str) -> Option<LoopHit> {
        for c in s.chars() {
            if let Some(hit) = self.feed(c) {
                return Some(hit);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `input` char by char; return `(trigger_index, hit)` or `None`.
    fn detect(input: &str, min_block: usize, max_block: usize, min_repeats: usize) -> Option<(usize, LoopHit)> {
        let mut d = LoopDetector::new(min_block, max_block, min_repeats);
        for (i, c) in input.chars().enumerate() {
            if let Some(hit) = d.feed(c) {
                return Some((i, hit));
            }
        }
        None
    }

    /// Feed `input` in chunks of `chunk_size` chars; return `(trigger_index, hit)` or `None`.
    fn detect_chunked(
        input: &str,
        chunk_size: usize,
        min_block: usize,
        max_block: usize,
        min_repeats: usize,
    ) -> Option<(usize, LoopHit)> {
        let mut d = LoopDetector::new(min_block, max_block, min_repeats);
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let end = (i + chunk_size).min(chars.len());
            for (offset, &c) in chars[i..end].iter().enumerate() {
                if let Some(hit) = d.feed(c) {
                    return Some((i + offset, hit));
                }
            }
            i = end;
        }
        None
    }

    /// First `hit.kept_chars` characters of `input`.
    fn kept(input: &str, hit: &LoopHit) -> String {
        input.chars().take(hit.kept_chars).collect()
    }

    // A1: basic loop.
    #[test]
    fn a1_basic_loop() {
        let (i, hit) = detect("ABCDABCDABCD", 4, 10, 2).unwrap();
        assert_eq!(i, 7, "trigger at 8th char");
        assert_eq!(hit.block_len, 4);
        assert_eq!(hit.run, 8);
        assert_eq!(kept("ABCDABCDABCD", &hit), "ABCD");
    }

    // A2: no loop.
    #[test]
    fn a2_no_loop() {
        assert!(detect("The quick brown fox jumps over the lazy dog near the river bank.", 10, 1024, 2).is_none());
    }

    // A3: fundamental block smaller than min_block -> no trigger.
    #[test]
    fn a3_block_below_min() {
        // "AB" x 10 has fundamental period 2 (< min_block=10); must NOT trigger.
        assert!(detect("ABABABABABABABABABAB", 10, 1024, 2).is_none());
    }

    // A4: block larger than max_block -> no trigger.
    #[test]
    fn a4_block_above_max() {
        // 2000-char primitive block repeated twice; fundamental period 2000 > max_block.
        let block: String = std::iter::repeat_n('A', 1).chain(std::iter::repeat_n('B', 1999)).collect();
        let input = format!("{block}{block}");
        assert_eq!(input.chars().count(), 4000);
        assert!(detect(&input, 10, 1024, 2).is_none());
    }

    // A5: boundary not reached.
    #[test]
    fn a5_boundary_not_reached() {
        assert!(detect("WXYZWXY", 4, 10, 2).is_none());
    }

    // A6: boundary exact.
    #[test]
    fn a6_boundary_exact() {
        let (i, hit) = detect("WXYZWXYZ", 4, 10, 2).unwrap();
        assert_eq!(i, 7);
        assert_eq!(hit.block_len, 4);
        assert_eq!(kept("WXYZWXYZ", &hit), "WXYZ");
    }

    // A7: minimal period.
    #[test]
    fn a7_minimal_period() {
        let (i, hit) = detect("ABABABAB", 2, 10, 2).unwrap();
        assert_eq!(i, 3);
        assert_eq!(hit.block_len, 2, "fundamental period is 2, not 4");
        assert_eq!(kept("ABABABAB", &hit), "AB");
    }

    // A8: nested period.
    #[test]
    fn a8_nested_period() {
        let (i, hit) = detect("ABCABCABCABC", 3, 10, 2).unwrap();
        assert_eq!(i, 5);
        assert_eq!(hit.block_len, 3);
        assert_eq!(kept("ABCABCABCABC", &hit), "ABC");
    }

    // A9: mismatch resets the run.
    #[test]
    fn a9_mismatch_reset() {
        assert!(detect("AAAABAAAA", 4, 10, 2).is_none());
    }

    // A10: triggers on the 2nd block, not later.
    #[test]
    fn a10_early_trigger() {
        let (i, _) = detect("ABCDABCDABCDABCDABCD", 4, 10, 2).unwrap();
        assert_eq!(i, 7, "trigger at 2nd block, not 5th");
    }

    // A11: CJK (1 char per code point).
    #[test]
    fn a11_cjk() {
        let (i, hit) = detect("你好世界你好世界你好世界", 4, 10, 2).unwrap();
        assert_eq!(i, 7);
        assert_eq!(hit.block_len, 4);
        assert_eq!(kept("你好世界你好世界你好世界", &hit), "你好世界");
    }

    // A12: non-BMP emoji (1 char each, 4 bytes UTF-8).
    #[test]
    fn a12_emoji_non_bmp() {
        let input = "🦀🐙🦀🐙🦀🐙🦀🐙";
        assert_eq!(input.chars().count(), 8);
        let (i, hit) = detect(input, 2, 10, 2).unwrap();
        assert_eq!(i, 3);
        assert_eq!(hit.block_len, 2);
        assert_eq!(kept(input, &hit), "🦀🐙");
    }

    // A13: mixed Latin + CJK.
    #[test]
    fn a13_mixed() {
        let input = "hello你好hello你好";
        let (i, hit) = detect(input, 7, 10, 2).unwrap();
        assert_eq!(i, 13);
        assert_eq!(hit.block_len, 7);
        assert_eq!(kept(input, &hit), "hello你好");
    }

    // A14: chunking invariance.
    #[test]
    fn a14_chunking_invariance() {
        let input = "ABCDABCDABCD";
        let base = detect(input, 4, 10, 2).unwrap();
        for chunk in [1usize, 3, 7, 12] {
            let got = detect_chunked(input, chunk, 4, 10, 2).unwrap();
            assert_eq!(got, base, "chunk size {chunk} must match char-by-char");
        }
    }

    // A15: short input.
    #[test]
    fn a15_short_input() {
        assert!(detect("ABC", 4, 10, 2).is_none());
    }

    // A16: min_repeats = 3.
    #[test]
    fn a16_min_repeats_3() {
        assert!(detect("ABCDABCD", 4, 10, 3).is_none(), "2 copies < 3");
        let (i, hit) = detect("ABCDABCDABCD", 4, 10, 3).unwrap();
        assert_eq!(i, 11, "trigger at 3rd copy");
        assert_eq!(hit.block_len, 4);
        assert_eq!(kept("ABCDABCDABCD", &hit), "ABCD");
    }

    // A17: prefix + loop.
    #[test]
    fn a17_prefix_plus_loop() {
        let input = "Hello, world!world!world!";
        let (i, hit) = detect(input, 6, 10, 2).unwrap();
        assert_eq!(i, 18);
        assert_eq!(hit.block_len, 6);
        assert_eq!(kept(input, &hit), "Hello, world!");
    }

    // A18: partial 3rd block is dropped.
    #[test]
    fn a18_partial_third_block() {
        let input = "ABCDABCDAB";
        let (i, hit) = detect(input, 4, 10, 2).unwrap();
        assert_eq!(i, 7, "trigger at end of 2nd block");
        assert_eq!(kept(input, &hit), "ABCD", "partial 3rd block dropped");
    }

    // A19: window eviction (long prefix longer than max_block, then a loop).
    #[test]
    fn a19_window_eviction() {
        let input = format!("{}ABCDABCDABCD", "X".repeat(5000));
        let (i, hit) = detect(&input, 4, 1024, 2).unwrap();
        assert_eq!(i, 5007, "loop detected after 5000-char prefix");
        assert_eq!(hit.block_len, 4);
        assert_eq!(hit.kept_chars, 5004);
        // The all-X prefix block is non-primitive, so it never triggered on its own.
    }

    // ---- TieredDetector ----

    /// A primitive block of `len` chars with all characters distinct, so the
    /// block has **no period at all** (`p < len`). This is required for two
    /// reasons: (1) the long tier's primitive check rejects any block with a
    /// period `p < short_max + 1`, and (2) a residual small period would let the
    /// short tier fire first. ASCII has too few distinct chars for `len > 95`,
    /// so we use CJK ideographs (each a single `char`, counted as one unit).
    fn primitive_block(len: usize) -> String {
        (0..len)
            .map(|i| char::from_u32(0x4E00 + i as u32).unwrap())
            .collect()
    }

    // T1: short-tier loop (8-char block x3 in [8, 64] x3).
    #[test]
    fn t1_short_tier() {
        let mut d = TieredDetector::new(8, 64, 3, 2048);
        let hit = d.feed_str(&"ABCDEFGH".repeat(3)).unwrap();
        assert_eq!(hit.block_len, 8);
        assert_eq!(hit.run, 24);
    }

    // T2: long-tier loop (100-char primitive block x2 in [65, 2048] x2).
    #[test]
    fn t2_long_tier() {
        let block = primitive_block(100);
        let mut d = TieredDetector::new(8, 64, 3, 2048);
        let hit = d.feed_str(&format!("{block}{block}")).unwrap();
        assert_eq!(hit.block_len, 100);
        assert_eq!(hit.run, 200);
    }

    // T3: short block repeated only twice (< short_repeats=3) -> no trigger
    // from either tier (the long tier sees a non-primitive 16-char block).
    #[test]
    fn t3_short_below_repeats() {
        let mut d = TieredDetector::new(8, 64, 3, 2048);
        assert!(d.feed_str(&"ABCDEFGH".repeat(2)).is_none());
    }

    // T4: the 64-char boundary belongs to the SHORT tier: x2 does not trigger,
    // x3 does.
    #[test]
    fn t4_boundary_64_is_short() {
        let block = primitive_block(64);
        let mut d = TieredDetector::new(8, 64, 3, 2048);
        assert!(
            d.feed_str(&format!("{block}{block}")).is_none(),
            "64-char block x2 must not trigger"
        );
        let mut d = TieredDetector::new(8, 64, 3, 2048);
        let hit = d.feed_str(&format!("{block}{block}{block}")).unwrap();
        assert_eq!(hit.block_len, 64);
        assert_eq!(hit.run, 192);
    }

    // T5: long_max <= short_max -> long tier disabled.
    #[test]
    fn t5_long_tier_disabled() {
        let block = primitive_block(100);
        let mut d = TieredDetector::new(8, 64, 3, 64);
        assert!(d.feed_str(&format!("{block}{block}")).is_none());
    }

    // T6: chunking invariance (long-tier hit).
    #[test]
    fn t6_chunk_invariance() {
        let block = primitive_block(100);
        let input = format!("{block}{block}");
        let mut d = TieredDetector::new(8, 64, 3, 2048);
        let base = d.feed_str(&input).unwrap();
        for chunk in [1usize, 7, 100, 250] {
            let mut d = TieredDetector::new(8, 64, 3, 2048);
            let chars: Vec<char> = input.chars().collect();
            let mut i = 0;
            let mut got = None;
            while i < chars.len() && got.is_none() {
                let end = (i + chunk).min(chars.len());
                for &c in &chars[i..end] {
                    if let Some(h) = d.feed(c) {
                        got = Some(h);
                        break;
                    }
                }
                i = end;
            }
            assert_eq!(got, Some(base.clone()), "chunk size {chunk} must match");
        }
    }
}
