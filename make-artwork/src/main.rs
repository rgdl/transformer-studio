use std::env;

use rust_perlin::perlin;

struct Config {
    rows: usize,
    cols: usize,
    scale: f32,
}

impl Config {
    pub fn from_args_vector(args: Vec<String>) -> Self {
        if args.len() != 4 {
            panic!("There must be exactly 3 command line args");
        }

        Config {
            rows: args[1].parse().expect("Invalid rows"),
            cols: args[2].parse().expect("Invalid cols"),
            scale: args[3].parse().expect("Invalid scale"),
        }
    }
}

fn main() -> () {
    let args_vec: Vec<String> = env::args().collect();
    let config = Config::from_args_vector(args_vec);
    println!("{:?}", perlin(config.rows, config.cols, config.scale));
}
