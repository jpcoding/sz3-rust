# sz3-rust

Rust port of SZ3 (reference: C++ SZ3 3.4.0). See PLAN.md for the roadmap and README.md for usage.

## Commits

- Commits are authored by the repository owner; Claude is a collaborator, never the author.
  Do not change `user.name` / `user.email`.
- End every commit message with this trailer (the template in `.gitmessage` pre-fills it):

  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

## Working on the code

- Keep SZ3's modular design: `Predictor`, `Quantizer`, `Decomposition`, `Encoder`, `Lossless`,
  composed by `GenericCompressor`. Everything above zstd is native Rust.
- Streams must stay byte-identical to C++ SZ3. Replicate the C++ evaluation order and float/double
  promotion exactly. After any change to the pipeline, run `scripts/crosscheck.py` against a C++
  build linked to the same zstd (1.5.7, `-DSZ3_USE_BUNDLED_ZSTD=OFF`).
- No `unsafe`; untrusted input returns `Err`, never panics or allocates without bound.
- Before committing: `cargo fmt`, `cargo clippy --all-targets` (no warnings), `cargo test --release`.
