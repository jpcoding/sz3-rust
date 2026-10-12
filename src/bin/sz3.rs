//! `sz3` command line tool, argument-compatible with SZ3's C++ `sz3`.

use std::process::ExitCode;
use std::time::Instant;

use sz3::bytes::ByteWriter;
use sz3::config::{EB_MAP, SZ3_DATA_VER, SZ3_VERSION, eb, fmt_g_prec, match_enum};
use sz3::{Config, SzData, compress_with, decompress, stats};

fn usage() -> ! {
    print!(
        "Note: SZ3 command line arguments are backward compatible with SZ2, \n\
         \x20     use -h2 to show the supported SZ2 command line arguments. \n\
         Usage: sz3 <options>\n\
         Options:\n\
         * general options:\n\
         \t-h: print the help information\n\
         \t-h2: print the help information for SZ2 style command line\n\
         \t-v: print the version number\n\
         \t-a : print compression results such as distortions\n\
         * input and output:\n\
         \t-i <path> : original input file in binary format\n\
         \t-o <path> : decompressed file in binary format\n\
         \t-z <path> : compressed file\n\
         \t-t : store decompressed file in text format\n\
         * data type:\n\
         \t-f: single precision (float type)\n\
         \t-d: double precision (double type)\n\
         \t-I <width>: integer type (width = 32 or 64)\n\
         * configuration file: \n\
         \t-c <configuration file> : configuration file sz.config\n\
         * error control: (the error control parameters here will overwrite the setting in sz.config)\n\
         \t-M <error control mode> <error bound (optional)> \n\
         \terror control mode as follows: \n\
         \t\tABS (absolute error bound)\n\
         \t\tREL (value range based error bound, so a.k.a., VR_REL)\n\
         \t\tPSNR (peak signal-to-noise ratio)\n\
         \t\tNORM (norm2 error : sqrt(sum(xi-xi')^2)\n\
         \t\tABS_AND_REL (using min{{ABS, REL}})\n\
         \t\tABS_OR_REL (using max{{ABS, REL}})\n\
         \terror bound can be set directly after the error control mode, or separately with the following options:\n\
         \t\t-A <absolute error bound>: specifying absolute error bound\n\
         \t\t-R <value_range based relative error bound>: specifying relative error bound\n\
         \t\t-S <PSNR>: specifying PSNR\n\
         \t\t-N <normErr>: specifying normErr\n\
         * dimensions: \n\
         \t-1 <nx> : dimension for 1D data such as data[nx]\n\
         \t-2 <nx> <ny> : dimensions for 2D data such as data[ny][nx]\n\
         \t-3 <nx> <ny> <nz> : dimensions for 3D data such as data[nz][ny][nx] \n\
         \t-4 <nx> <ny> <nz> <np>: dimensions for 4D data such as data[np][nz][ny][nx] \n\
         * examples: \n\
         \tsz3 -f -i test.dat    -z test.dat.sz     -3 8 8 128 -M ABS 1e-3 \n\
         \tsz3 -f -z test.dat.sz -o test.dat.sz.out -3 8 8 128 -M REL 1e-3 -a \n\
         \tsz3 -f -i test.dat    -o test.dat.sz.out -3 8 8 128 -M ABS_AND_REL -A 1 -R 1e-3 -a \n\
         \tsz3 -f -i test.dat    -o test.dat.sz.out -3 8 8 128 -c sz.config \n"
    );
    std::process::exit(0);
}

#[derive(Clone, Copy, PartialEq)]
enum DType {
    F32,
    F64,
    I32,
    I64,
}

struct Args {
    binary_output: bool,
    print_cmp_results: bool,
    print_meta: bool,
    compression: bool,
    decompression: bool,
    dtype: DType,
    in_path: Option<String>,
    cmp_path: Option<String>,
    con_path: Option<String>,
    dec_path: Option<String>,
    del_cmp_path: bool,
    eb_mode: Option<String>,
    eb: Option<String>,
    abs: Option<String>,
    rel: Option<String>,
    psnr: Option<String>,
    norm: Option<String>,
    dims: Vec<usize>,
}

fn parse_dim(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(v) if v > 0 && s.as_bytes()[0].is_ascii_digit() => Ok(v),
        _ => Err(format!("invalid dimension '{s}'")),
    }
}

/// C `atof`: the longest numeric prefix, or 0.
fn atof(s: &str) -> f64 {
    let s = s.trim_start();
    (1..=s.len())
        .rev()
        .filter(|&i| s.is_char_boundary(i))
        .find_map(|i| s[..i].parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        binary_output: true,
        print_cmp_results: false,
        print_meta: false,
        compression: false,
        decompression: false,
        dtype: DType::F32,
        in_path: None,
        cmp_path: None,
        con_path: None,
        dec_path: None,
        del_cmp_path: false,
        eb_mode: None,
        eb: None,
        abs: None,
        rel: None,
        psnr: None,
        norm: None,
        dims: Vec::new(),
    };
    if argv.len() == 1 {
        usage();
    }
    let mut sz2mode = false;
    let mut i = 1;
    let next = |i: &mut usize, what: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("missing value for {what}"))
    };
    while i < argv.len() {
        let arg = &argv[i];
        let b = arg.as_bytes();
        if b.first() != Some(&b'-') || b.len() != 2 {
            if arg == "-h2" {
                println!("Note: SZ2-style arguments -z/-x/-s are supported; see -h.");
                std::process::exit(0);
            } else if arg == "--help" {
                usage();
            }
            return Err(format!("unrecognized argument: {arg}"));
        }
        match b[1] {
            b'h' => usage(),
            b'v' => {
                println!("SZ3 Version: {SZ3_VERSION}");
                println!("SZ3 Data Format Version: {SZ3_DATA_VER}");
                std::process::exit(0);
            }
            b'b' => a.binary_output = true,
            b't' => a.binary_output = false,
            b'a' => a.print_cmp_results = true,
            b'p' => a.print_meta = true,
            b'z' => {
                a.compression = true;
                if let Some(p) = argv.get(i + 1).filter(|p| !p.starts_with('-')) {
                    a.cmp_path = Some(p.clone());
                    i += 1;
                }
            }
            b'x' => {
                sz2mode = true;
                a.decompression = true;
                if let Some(p) = argv.get(i + 1).filter(|p| !p.starts_with('-')) {
                    a.dec_path = Some(p.clone());
                    i += 1;
                }
            }
            b'f' => a.dtype = DType::F32,
            b'd' => a.dtype = DType::F64,
            b'I' => {
                let w = next(&mut i, "-I")?;
                a.dtype = match w.parse::<i32>() {
                    Ok(32) => DType::I32,
                    Ok(64) => DType::I64,
                    _ => return Err(format!("integer width must be 32 or 64, got '{w}'")),
                };
            }
            b'i' => a.in_path = Some(next(&mut i, "-i")?),
            b'o' => a.dec_path = Some(next(&mut i, "-o")?),
            b's' => a.cmp_path = Some(next(&mut i, "-s")?),
            b'c' => a.con_path = Some(next(&mut i, "-c")?),
            c @ b'1'..=b'4' => {
                let n = (c - b'0') as usize;
                a.dims.clear();
                for _ in 0..n {
                    a.dims.push(parse_dim(&next(&mut i, "dimension")?)?);
                }
            }
            b'M' => {
                a.eb_mode = Some(next(&mut i, "-M")?);
                if let Some(p) = argv.get(i + 1).filter(|p| !p.starts_with('-')) {
                    a.eb = Some(p.clone());
                    i += 1;
                }
            }
            b'A' => a.abs = Some(next(&mut i, "-A")?),
            b'R' => a.rel = Some(next(&mut i, "-R")?),
            b'N' => a.norm = Some(next(&mut i, "-N")?),
            b'S' => a.psnr = Some(next(&mut i, "-S")?),
            _ => return Err(format!("unknown option: {arg}")),
        }
        i += 1;
    }
    if a.in_path.is_none() && a.cmp_path.is_none() {
        return Err(
            "you need to specify either a raw binary data file (-i) or a compressed data file (-s/-z) as input".into(),
        );
    }
    if !sz2mode && a.in_path.is_some() && a.cmp_path.is_some() {
        a.compression = true;
    }
    if a.cmp_path.is_some() && a.dec_path.is_some() {
        a.decompression = true;
    }
    if let (Some(inp), None, Some(_)) = (&a.in_path, &a.cmp_path, &a.dec_path) {
        a.compression = true;
        a.decompression = true;
        a.cmp_path = Some(format!("{inp}.sz.tmp"));
        a.del_cmp_path = true;
    }
    if a.in_path.is_none() || (a.eb_mode.is_none() && a.con_path.is_none()) {
        a.compression = false;
    }
    if !a.compression && !a.decompression {
        return Err(
            "nothing to do: specify compression (-i with -z/-o and an error bound) or decompression (-s/-x)".into(),
        );
    }
    Ok(a)
}

fn read_values<T: SzData>(path: &str) -> Result<Vec<T>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    if bytes.len() % T::SIZE != 0 {
        return Err(format!("{path}: size is not a multiple of {} bytes", T::SIZE));
    }
    Ok(bytes.chunks_exact(T::SIZE).map(T::read_le).collect())
}

fn write_values<T: SzData + std::fmt::Display>(path: &str, v: &[T], binary: bool) -> Result<(), String> {
    if binary {
        // in pieces, so the output is never held twice
        let mut f = std::fs::File::create(path).map_err(|e| format!("cannot write {path}: {e}"))?;
        let mut b = Vec::new();
        for c in v.chunks(1 << 16) {
            b.clear();
            b.put_slice(c);
            std::io::Write::write_all(&mut f, &b).map_err(|e| format!("cannot write {path}: {e}"))?;
        }
        return Ok(());
    }
    let mut s = String::new();
    for x in v {
        s += &format!("{}\n", fmt_text(x));
    }
    std::fs::write(path, s).map_err(|e| format!("cannot write {path}: {e}"))
}

/// `ostream << value`: integers as is, floats in `%g`.
fn fmt_text<T: SzData + std::fmt::Display>(x: &T) -> String {
    if T::IS_FLOAT {
        fmt_g_prec(x.to_f64(), 6)
    } else {
        x.to_string()
    }
}

fn run_compress<T: SzData>(a: &Args, conf: &mut Config) -> Result<(), String> {
    let inp = a.in_path.as_deref().unwrap();
    let data = read_values::<T>(inp)?;
    if data.len() < conf.num {
        return Err(format!(
            "{inp}: {} values, but the dimensions need {}",
            data.len(),
            conf.num
        ));
    }
    let t = Instant::now();
    let bytes = compress_with(conf, &data).map_err(|e| e.to_string())?;
    let secs = t.elapsed().as_secs_f64();
    let out = a.cmp_path.clone().unwrap_or_else(|| format!("{inp}.sz"));
    std::fs::write(&out, &bytes).map_err(|e| format!("cannot write {out}: {e}"))?;
    println!(
        "compression ratio = {:.2} ",
        (conf.num * T::SIZE) as f64 / bytes.len() as f64
    );
    println!("compression time = {secs:.6}");
    println!("compressed data file = {out}");
    Ok(())
}

fn run_decompress<T: SzData + std::fmt::Display>(a: &Args, conf: &mut Config) -> Result<(), String> {
    let cmp_path = a.cmp_path.as_deref().unwrap();
    let cmp = std::fs::read(cmp_path).map_err(|e| format!("cannot read {cmp_path}: {e}"))?;
    let t = Instant::now();
    let (c, dec) = decompress::<T>(&cmp).map_err(|e| e.to_string())?;
    let secs = t.elapsed().as_secs_f64();
    *conf = c;
    let out = a.dec_path.clone().unwrap_or_else(|| format!("{cmp_path}.out"));
    write_values(&out, &dec, a.binary_output)?;
    if a.print_cmp_results {
        let ori = read_values::<T>(a.in_path.as_deref().unwrap())?;
        if ori.len() != dec.len() {
            return Err("the original data has a different number of values".into());
        }
        let o: Vec<f64> = ori.iter().map(|x| x.to_f64()).collect();
        let d: Vec<f64> = dec.iter().map(|x| x.to_f64()).collect();
        let q = stats::verify(&o, &d);
        println!("Min={:.20e}, Max={:.20e}, range={:.20e}", q.min, q.max, q.range);
        println!("Max absolute error = {}", fmt_g_prec(q.max_abs_err, 2));
        println!("Max relative error = {}", fmt_g_prec(q.max_rel_err, 2));
        println!("Max pw relative error = {}", fmt_g_prec(q.max_pw_rel_err, 2));
        println!("PSNR = {:.6}, NRMSE= {}", q.psnr, fmt_g_prec(q.nrmse, 10));
        println!("normError = {:.6}, normErr_norm = {:.6}", q.norm_err, q.norm_err_norm);
        println!("acEff={:.6}", q.ac_eff);
    }
    println!(
        "compression ratio = {:.6}",
        (conf.num * T::SIZE) as f64 / cmp.len() as f64
    );
    println!("decompression time = {secs:.6} seconds.");
    println!("decompressed file = {out}");
    Ok(())
}

fn run(argv: &[String]) -> Result<(), String> {
    let a = parse_args(argv)?;
    // -1 nx / -2 nx ny / ...: the fastest dimension comes first on the command line.
    let rev: Vec<usize> = a.dims.iter().rev().copied().collect();
    let mut conf = Config::new(if rev.is_empty() { &[0] } else { &rev });
    if a.compression
        && let Some(p) = &a.con_path
    {
        conf.load_cfg(p).map_err(|e| e.to_string())?;
    }
    if let Some(mode) = &a.eb_mode {
        if let Some(v) = &a.rel {
            conf.rel_error_bound = atof(v);
        }
        if let Some(v) = &a.abs {
            conf.abs_error_bound = atof(v);
        }
        if let Some(v) = &a.psnr {
            conf.psnr_error_bound = atof(v);
        }
        if let Some(v) = &a.norm {
            conf.l2norm_error_bound = atof(v);
        }
        match_enum(mode, EB_MAP, &mut conf.error_bound_mode);
        if mode == "VR_REL" {
            conf.error_bound_mode = eb::REL;
        }
        let v = a.eb.as_deref().map(atof);
        match conf.error_bound_mode {
            eb::ABS => conf.abs_error_bound = v.unwrap_or(conf.abs_error_bound),
            eb::REL => conf.rel_error_bound = v.unwrap_or(conf.rel_error_bound),
            eb::PSNR => conf.psnr_error_bound = v.unwrap_or(conf.psnr_error_bound),
            eb::L2NORM => conf.l2norm_error_bound = v.unwrap_or(conf.l2norm_error_bound),
            eb::ABS_AND_REL | eb::ABS_OR_REL => {}
            _ => return Err("wrong error bound mode setting by using the option '-M'".into()),
        }
    }
    if a.compression {
        if a.dims.is_empty() {
            return Err("compression needs the dimensions (-1/-2/-3/-4)".into());
        }
        match a.dtype {
            DType::F32 => run_compress::<f32>(&a, &mut conf)?,
            DType::F64 => run_compress::<f64>(&a, &mut conf)?,
            DType::I32 => run_compress::<i32>(&a, &mut conf)?,
            DType::I64 => run_compress::<i64>(&a, &mut conf)?,
        }
    }
    if a.decompression {
        if a.print_cmp_results && a.in_path.is_none() {
            return Err("the -a option (analysis) needs the original data path, specify it with -i <path>".into());
        }
        match a.dtype {
            DType::F32 => run_decompress::<f32>(&a, &mut conf)?,
            DType::F64 => run_decompress::<f64>(&a, &mut conf)?,
            DType::I32 => run_decompress::<i32>(&a, &mut conf)?,
            DType::I64 => run_decompress::<i64>(&a, &mut conf)?,
        }
    }
    if a.print_meta {
        conf.print();
    }
    if a.del_cmp_path {
        let _ = std::fs::remove_file(a.cmp_path.as_deref().unwrap());
    }
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    match run(&argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sz3: {e}");
            ExitCode::FAILURE
        }
    }
}
