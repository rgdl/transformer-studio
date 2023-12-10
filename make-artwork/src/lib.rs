use pyo3::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Normal, Distribution};

enum Num32 {
    Integer(i32),
    Float(f32),
}

type Array = Vec<Vec<f32>>;
type IntArray = Vec<Vec<i32>>;
type AnyArray = Vec<Vec<Num32>>;
type Tensor3 = Vec<Vec<Vec<f32>>>;

#[pyfunction]
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

#[pyfunction]
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

#[pyfunction]
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

#[pyfunction]
fn quantise_grid(grid: Array, quantise_up: bool) -> IntArray {
    grid.iter().map(
        |row| row.iter().map(
            // N.B. val.ceil() rounded up ~0 to 1, hence using val.floor() + 1.0
            |val| if quantise_up { val.floor() as i32 + 1 } else { val.floor() as i32}
        ).collect()
    ).collect()
}

// TODO: once we've got the whole thing moved to rust, do a clever version using enums that can
// contain either floats or ints
#[pyfunction]
fn multiply_grid(scalar: f32, grid: Array) -> Array {
    grid.iter().map(
        |row| row.iter().map(
            |val| scalar * val
        ).collect()
    ).collect()
}

#[pyfunction]
fn add_grid(grid1: Array, grid2: Array) -> Array {
    if grid1.len() != grid2.len() || grid1[0].len() != grid2[0].len() {
        panic!("Mismatching grid shapes");
    }

    grid1.iter().zip(grid2.iter()).map(
        |(row1, row2)| row1.iter().zip(row2.iter()).map(
            |(val1, val2)| val1 + val2
        ).collect()
    ).collect()
}

//#[pyfunction]
//fn multiply_grid(scalar: Num32, grid: AnyArray) -> Array {
//    grid.iter().map(
//        |row| row.iter().map(
//            |val| scalar * val as i32
//        ).collect()
//    ).collect()
//}

#[pymodule]
fn rust_perlin(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smoothstep, m)?)?;
    m.add_function(wrap_pyfunction!(random_normal, m)?)?;
    m.add_function(wrap_pyfunction!(make_grids, m)?)?;
    m.add_function(wrap_pyfunction!(quantise_grid, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_grid, m)?)?;
    m.add_function(wrap_pyfunction!(add_grid, m)?)?;

    Ok(())
}
