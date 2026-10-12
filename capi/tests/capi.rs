use std::ffi::c_void;

use SZ3c::{SZ_compress_args, SZ_decompress, free_buf};

#[test]
fn round_trip_and_errors() {
    let (r1, r2) = (300usize, 200usize);
    let data: Vec<f32> = (0..r1 * r2)
        .map(|i| ((i % r1) as f32 * 0.05).sin() + (i / r1) as f32 * 0.01)
        .collect();
    let mut size = 0usize;
    // SAFETY: data holds r1 * r2 floats; the returned buffers are freed with free_buf.
    unsafe {
        let cmp = SZ_compress_args(
            0,
            data.as_ptr() as *mut c_void,
            &mut size,
            0,
            1e-3,
            0.0,
            0.0,
            0,
            0,
            0,
            r2,
            r1,
        );
        assert!(!cmp.is_null() && size > 0);
        let dec = SZ_decompress(0, cmp, size, 0, 0, 0, r2, r1) as *const f32;
        assert!(!dec.is_null());
        let out = std::slice::from_raw_parts(dec, r1 * r2);
        assert!(data.iter().zip(out).all(|(a, b)| (a - b).abs() <= 1e-3));
        // dimensions that do not match the stream
        assert!(SZ_decompress(0, cmp, size, 0, 0, 0, r2, r1 + 1).is_null());
        free_buf(dec as *mut c_void);
        free_buf(cmp as *mut c_void);
        assert!(
            SZ_compress_args(
                5,
                data.as_ptr() as *mut c_void,
                &mut size,
                0,
                1e-3,
                0.0,
                0.0,
                0,
                0,
                0,
                r2,
                r1
            )
            .is_null()
        );
    }
}
