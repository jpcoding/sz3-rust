//! `mdz`: compresses a molecular-dynamics trajectory in memory and reports ratio, speed and error
//! (SZ3's tools/mdz).

use std::process::ExitCode;

use sz3::config::{Config, eb, fmt_g_prec};
use sz3::mdz::{COMPRESSOR_NAMES, MdzReport, mdz_compress_2d, mdz_compress_3d};
use sz3::stats;

fn usage() -> ! {
    println!("Usage: ");
    println!("For 2D input:   mdz file_path -2 n_frames n_atoms        -r reb");
    println!("For 3D input:   mdz file_path -3 n_frames n_atoms n_dims(x,y,z) -r reb");
    println!("Optional after the bound: batch_size method(0 VQ, 1 VQT, 2 MT, 3 LR, 4 TS, -1 auto) quant_bins");
    std::process::exit(0);
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        usage();
    }
    let n: usize = args[2].get(1..).and_then(|s| s.parse().ok()).unwrap_or(0);
    if !(2..=3).contains(&n) {
        usage();
    }
    let mut p = 3;
    let mut dims = Vec::new();
    for _ in 0..n {
        dims.push(
            args.get(p)
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or("bad dimension")?,
        );
        p += 1;
    }
    let mut conf = Config::new(&dims);
    if conf.dims.len() != n {
        return Err("every dimension must be larger than 1".into());
    }
    let Some(op) = args.get(p) else { usage() };
    let bound: f64 = args.get(p + 1).and_then(|s| s.parse().ok()).ok_or("bad error bound")?;
    p += 2;
    if op.as_bytes().get(1) == Some(&b'a') {
        conf.error_bound_mode = eb::ABS;
        conf.abs_error_bound = bound;
    } else {
        conf.error_bound_mode = eb::REL;
        conf.rel_error_bound = bound;
    }
    let batch_size: usize = args.get(p).map_or(Ok(0), |s| s.parse()).map_err(|_| "bad batch size")?;
    let method: i32 = args
        .get(p + 1)
        .map_or(Ok(-1), |s| s.parse())
        .map_err(|_| "bad method")?;
    conf.block_size = 128;
    conf.quantbin_cnt = args
        .get(p + 2)
        .map_or(Ok(1024), |s| s.parse())
        .map_err(|_| "bad bin count")?;

    let bytes = std::fs::read(&args[1]).map_err(|e| format!("cannot read {}: {e}", args[1]))?;
    if bytes.len() < conf.num * 4 {
        return Err(format!("{}: fewer than {} floats", args[1], conf.num));
    }
    let input: Vec<f32> = bytes[..conf.num * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut dec = vec![0f32; conf.num];
    let mut report = MdzReport::default();
    if n == 2 {
        mdz_compress_2d(conf.clone(), &input, &mut dec, batch_size, method, &mut report)
    } else {
        mdz_compress_3d(&conf, &input, &mut dec, batch_size, method, &mut report)
    }
    .map_err(|e| e.to_string())?;
    for (a, b, m) in &report.batches {
        println!("From {a} to {b} , Compressor = {}", COMPRESSOR_NAMES[*m]);
    }
    let ratio = (conf.num as f32 * 4.0) / report.compressed_size as f32;
    println!(
        "\nBatch={}\nCompression ratio={ratio:.3}\nCompression time={:.3}\nDecompression time={:.3}",
        if batch_size == 0 { dims[0] } else { batch_size },
        report.compress_time,
        report.decompress_time
    );
    let o: Vec<f64> = input.iter().map(|&x| x as f64).collect();
    let d: Vec<f64> = dec.iter().map(|&x| x as f64).collect();
    let q = stats::verify(&o, &d);
    println!("Max absolute error = {}", fmt_g_prec(q.max_abs_err, 2));
    println!("PSNR = {:.6}, NRMSE= {}", q.psnr, fmt_g_prec(q.nrmse, 10));
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mdz: {e}");
            ExitCode::FAILURE
        }
    }
}
