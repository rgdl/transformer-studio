use std::time::SystemTime;

use pyo3::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Normal, Distribution};

// Currently running slower than the python equivalent, possibly due to the overhead of moving data
// between processes. Would be good to profile it, use parallelisation, etc.

enum Num32 {
    Integer(i32),
    Float(f32),
}

type Array = Vec<Vec<f32>>;
type UsizeArray = Vec<Vec<usize>>;
type AnyArray = Vec<Vec<Num32>>;
type Tensor3 = Vec<Vec<Vec<f32>>>;

fn smoothstep(x: Array) -> Array {
    x.iter().map(
        |row| row.iter().map(
            |&i| {
                if i < 0.0 || i > 1.0 {
                    panic!("Value {} is out of range", i);
                }
                i * i * (3.0 - 2.0 * i)
            }
        ).collect()
    ).collect()
}

fn interpolate(dx0: Array, dy0: Array, dot00: Array, dot10: Array, dot01: Array, dot11: Array) -> Array {
    let u = smoothstep(dx0);
    let v = smoothstep(dy0);

    let interpolate0 = add_grid(
        &dot00,
        &hadamard_product(&u, &subtract_grid(&dot10, &dot00)),
    );

    let interpolate1 = add_grid(
        &dot01,
        &hadamard_product(&u, &subtract_grid(&dot11, &dot01)),
    );

    let final_noise = add_grid(
        &interpolate0,
        &hadamard_product(&v, &subtract_grid(&interpolate1, &interpolate0)),
    );

    let (min, max) = grid_min_max(&final_noise);

    final_noise.iter().map(
        |row| row.iter().map(
            |x| (x - min) / (max - min)
        ).collect()
    ).collect()
}

fn grid_min_max(grid: &Array) -> (f32, f32) {
    let mut min = grid[0][0];
    let mut max = grid[0][0];

    for row in grid {
        for val in row {
            if val < &min { min = *val }
            if val > &max { max = *val }
        }
    }

    (min, max)
}

// TODO turn on strict linting, write doc strings, more abstractions. Once I know exactly what kind
// of functionality I need, switch to a 3rd party crate for data structures etc.
// TODO: definitely try to abstract away iteration over a whole Array or pair of Arrays

fn dot_product(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

fn hadamard_product(array1: &Array, array2: &Array) -> Array {
    array1.iter().zip(array2.iter()).map(
        |(row1, row2)| row1.iter().zip(row2.iter()).map(
            |(a, b)| a * b
        ).collect()
    ).collect()
}

fn dot_product_grid(grid1: Tensor3, grid2: Tensor3) -> Array {
    grid1.iter().zip(grid2.iter()).map(
        |(row1, row2)| row1.iter().zip(row2.iter()).map(
            |(&ref vec1, &ref vec2)| dot_product(vec1, vec2)
        ).collect()
    ).collect()
}


fn random_normal(rows: usize, cols: usize) -> Tensor3 {
    let seed: [u8; 32] = [1; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut result = Vec::with_capacity(rows);

    for _ in 0..=rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..=cols {
            let val = vec![normal.sample(&mut rng), normal.sample(&mut rng)];
            row.push(val);
        }
        result.push(row);
    }

    result
}

// TODO: any for loops that can be refactored to iter() patterns
// TODO: reduce cloning once it's all in rust

fn make_grids(rows: usize, cols: usize, scale: f32) -> (Array, Array) {
    let x_row: Vec<f32> = (0..cols).map(|i| scale * (i as f32) / (cols as f32)).collect();

    let mut x_grid = Vec::with_capacity(rows);

    for _ in 0..rows {
        x_grid.push(x_row.clone());
    }

    let mut y_grid = Vec::with_capacity(rows);
    for row in 0..rows {
        y_grid.push(vec![scale * (row as f32) / (rows as f32); cols])
    }

    (x_grid, y_grid)
}

fn quantise_grid(grid: &Array, quantise_up: bool) -> UsizeArray {
    grid.iter().map(
        |row| row.iter().map(
            // N.B. val.ceil() rounded up ~0 to 1, hence using val.floor() + 1.0
            |val| if quantise_up { val.floor() as usize + 1 } else { val.floor() as usize}
        ).collect()
    ).collect()
}

// TODO: once we've got the whole thing moved to rust, do a clever version using enums that can
// contain either floats or ints
fn multiply_grid(scalar: f32, grid: &Array) -> Array {
    grid.iter().map(
        |row| row.iter().map(
            |val| scalar * val
        ).collect()
    ).collect()
}

fn multiply_int_grid(scalar: f32, grid: &UsizeArray) -> Array {
    grid.iter().map(
        |row| row.iter().map(
            |&val| scalar * val as f32
        ).collect()
    ).collect()
}

fn add_grid(grid1: &Array, grid2: &Array) -> Array {
    if grid1.len() != grid2.len() || grid1[0].len() != grid2[0].len() {
        panic!("Mismatching grid shapes");
    }

    grid1.iter().zip(grid2.iter()).map(
        |(row1, row2)| row1.iter().zip(row2.iter()).map(
            |(val1, val2)| val1 + val2
        ).collect()
    ).collect()
}

fn subtract_grid(grid1: &Array, grid2: &Array) -> Array {
    add_grid(&grid1, &multiply_grid(-1.0, &grid2))
}

// TODO: slowest function
//
fn get_corner_gradients(gradients: &Tensor3, quantised_rows: &UsizeArray, quantised_cols: &UsizeArray) -> Tensor3 {
    quantised_rows.iter().zip(quantised_cols.iter()).map(
        |(qr_row, qc_row)| qr_row.iter().zip(qc_row.iter()).map(
            |(qr_val, qc_val)| gradients[*qr_val][*qc_val].clone()
        ).collect()
    ).collect()
}

fn stack_arrays(array1: &Array, array2: &Array) -> Tensor3 {
    let n_rows = array1.len();
    let n_cols = array1[0].len();

    if n_rows != array2.len() || n_cols != array2[0].len() {
        panic!("Array shapes must match!");
    }

    // TODO: do this with zips/iters instead?
    let mut result = Vec::with_capacity(n_rows);

    for r in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);

        for c in 0..n_cols {
            row.push(vec![array1[r][c], array2[r][c]]);
        }

        result.push(row);
    }

    result
}

#[pyfunction]
fn perlin(rows: usize, cols: usize, scale: f32) -> Array {
    let mut t0 = SystemTime::now();

    let gradients = random_normal(rows, cols);

    println!("***");
    println!("get gradients: {:?}", SystemTime::now().duration_since(t0));
    let mut t0 = SystemTime::now();

    let (x_grid, y_grid) = make_grids(rows, cols, scale);

    println!("make_grids: {:?}", SystemTime::now().duration_since(t0));
    let mut t0 = SystemTime::now();

    // Indices for nearest grid points

    let x0 = quantise_grid(&x_grid, false);
    let x1 = quantise_grid(&x_grid, true);
    let y0 = quantise_grid(&y_grid, false);
    let y1 = quantise_grid(&y_grid, true);

    println!("quantise_grid: {:?}", SystemTime::now().duration_since(t0));
    let mut t0 = SystemTime::now();

    // Calculate the distance vectors from the grid points to the coordinates

    let dx0 = add_grid(&x_grid, &multiply_int_grid(-1.0, &x0));
    let dx1 = add_grid(&x_grid, &multiply_int_grid(-1.0, &x1));
    let dy0 = add_grid(&y_grid, &multiply_int_grid(-1.0, &y0));
    let dy1 = add_grid(&y_grid, &multiply_int_grid(-1.0, &y1));

    println!("distance vectors: {:?}", SystemTime::now().duration_since(t0));
    let mut t0 = SystemTime::now();

    // Calculate the dot products between the gradient vectors and the distance vectors
    // TODO: this is the slow bit (mainly get_corner_gradients)
    let dot00 = dot_product_grid(
        get_corner_gradients(&gradients, &y0, &x0), stack_arrays(&dx0, &dy0)
    );

    let dot10 = dot_product_grid(
        get_corner_gradients(&gradients, &y0, &x1), stack_arrays(&dx1, &dy0)
    );

    let dot01 = dot_product_grid(
        get_corner_gradients(&gradients, &y1, &x0), stack_arrays(&dx0, &dy1)
    );

    let dot11 = dot_product_grid(
        get_corner_gradients(&gradients, &y1, &x1), stack_arrays(&dx1, &dy1)
    );

    println!("dot products: {:?}", SystemTime::now().duration_since(t0));
    let mut t0 = SystemTime::now();

    let final_noise = interpolate(dx0, dy0, dot00, dot10, dot01, dot11);

    println!("interpolate: {:?}", SystemTime::now().duration_since(t0));
    final_noise
}

#[pymodule]
fn rust_perlin(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perlin, m)?)?;

    Ok(())
}
