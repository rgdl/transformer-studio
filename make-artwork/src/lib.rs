use pyo3::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;

// Currently running slower than the python equivalent, possibly due to the overhead of moving data
// between processes. Would be good to profile it, use parallelisation, etc.

enum Number {
    Integer(usize),
    Float(f32),
}

struct _Array {
    data: Vec<Vec<f32>>,
    rows: usize,
    cols: usize,
}

impl _Array {
    fn new(data: Vec<Vec<f32>>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        Self { data: data, rows, cols }
    }

    fn check_range(&self, min: f32, max: f32) -> &Self {
        for r in 0..self.rows {
            for c in 0..self.cols {
                let val = self.data[r][c];

                if val < min || val > max {
                    panic!("Value {} is out of range", val);
                }
            }
        }

        self
    }

    fn apply<F: Sync + Send>(&self, func: F) -> Self
    where F: Fn(&f32) -> f32, {
        Self::new(
            self.data.par_iter().map(
                |row| row.par_iter().map(|val| func(&val)).collect()
            ).collect()
        )
    }

    fn zip_apply<F: Sync + Send>(&self, other: &Self, func: F) -> Self
    where F: Fn(&f32, &f32) -> f32, {
        Self::new(
            self.data.par_iter().zip(other.data.par_iter()).map(
                |(row1, row2)| row1.par_iter().zip(row2.par_iter()).map(
                    |(val1, val2)| func(val1, val2)
                ).collect()
            ).collect()
        )
    }

    fn check_same_size(&self, other: &Self) -> &Self {
        if self.cols != other.cols || self.rows != other.rows {
            panic!("Size mismatch!");
        }
        self
    }

    fn add(&self, other: &Self) -> Self {
        self.check_same_size(other);
        self.zip_apply(&other, |val1, val2| val1 + val2)
    }

    fn multiply(&self, scalar: f32) -> Self {
        self.apply(|val| scalar * val)
    }

    fn hadamard(&self, other: &Self) -> Self {
        self.check_same_size(other);
        self.zip_apply(&other, |val1, val2| val1 * val2)
    }

    fn subtract(&self, other: &Self) -> Self {
        self.add(&other.multiply(-1.0))
    }


    fn smoothstep(&self) -> Self {
        self.check_range(0.0, 1.0).apply(|&i| i * i * (3.0 - 2.0 * i))
    }

    fn min_max(&self) -> (f32, f32) {
        let mut min = &self.data[0][0];
        let mut max = &self.data[0][0];

        for row in &self.data {
            for val in row {
                if val < min { min = val }
                if val > max { max = val }
            }
        }

        (*min, *max)
    }
}

type Array = Vec<Vec<f32>>;
type UsizeArray = Vec<Vec<usize>>;
// type AnyArray = Vec<Vec<Num32>>;
type Tensor3 = Vec<Vec<Vec<f32>>>;

fn apply<T: Sync + Send, R: Sync + Send, F: Sync + Send>(array: &Vec<Vec<T>>, func: F) -> Vec<Vec<R>>
where F: Fn(&T) -> R, {
    array.par_iter().map(
        |row| row.par_iter().map(|val| func(&val)).collect()
    ).collect()
}

fn zip_apply<T: Sync + Send, R: Sync + Send, F: Sync + Send>(array1: &Vec<Vec<T>>, array2: &Vec<Vec<T>>, func: F) -> Vec<Vec<R>>
where F: Fn(&T, &T) -> R, {
    array1.par_iter().zip(array2.par_iter()).map(
        |(row1, row2)| row1.par_iter().zip(row2.par_iter()).map(
            |(val1, val2)| func(val1, val2)
        ).collect()
    ).collect()
}

fn interpolate(dx0: _Array, dy0: _Array, dot00: _Array, dot10: _Array, dot01: _Array, dot11: _Array) -> _Array {
    let u = dx0.smoothstep();
    let v = dy0.smoothstep();

    let interpolate0 = dot00.add(
        &u.hadamard(
            &dot10.subtract(
                &dot00
            )
        )
    );

    let interpolate1 = dot01.add(
        &u.hadamard(
            &dot11.subtract(
                &dot01
            )
        )
    );

    let final_noise = interpolate0.add(
        &v.hadamard(
            &interpolate1.subtract(
                &interpolate0
            )
        )
    );

    let (min, max) = final_noise.min_max();

    final_noise.apply(|x| (x - min) / (max - min))
}

// TODO turn on strict linting, write doc strings, more abstractions. Once I know exactly what kind
// of functionality I need, switch to a 3rd party crate for data structures etc.

fn dot_product(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    vec1.par_iter().zip(vec2.par_iter()).map(|(a, b)| a * b).sum()
}

fn hadamard_product(array1: &Array, array2: &Array) -> Array {
    zip_apply(array1, array2, |a, b| a * b)
}

fn dot_product_grid(grid1: Vec<Vec<&Vec<f32>>>, grid2: Vec<Vec<&Vec<f32>>>) -> Array {
    zip_apply(&grid1, &grid2, |&vec1, &vec2| dot_product(vec1, vec2))
}

fn random_normal(rows: usize, cols: usize) -> Tensor3 {
    let seed: [u8; 32] = [1; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    (0..=rows).map(
        |_| (0..=cols).map(
            |_| vec![
                normal.sample(&mut rng), normal.sample(&mut rng)
            ]
        ).collect()
    ).collect()
}

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
    apply(
        grid,
        |val| if quantise_up {
            val.floor() as usize + 1
        } else {
            val.floor() as usize
        },
    )
}

// TODO: once we've got the whole thing moved to rust, do a clever version using enums that can
// contain either floats or ints
fn multiply_grid(scalar: f32, grid: &Array) -> Array {
    apply(grid, |val| scalar * val)
}

fn multiply_int_grid(scalar: f32, grid: &UsizeArray) -> Array {
    apply(&grid, |&val| scalar * val as f32)
}

fn add_grid(grid1: &Array, grid2: &Array) -> Array {
    _Array::new(grid1.clone()).add(&_Array::new(grid2.clone())).data
}

fn subtract_grid(grid1: &Array, grid2: &Array) -> Array {
    add_grid(&grid1, &multiply_grid(-1.0, &grid2))
}

fn get_corner_gradients<'a>(gradients: &'a Tensor3, quantised_rows: &UsizeArray, quantised_cols: &UsizeArray) -> Vec<Vec<&'a Vec<f32>>> {
    zip_apply(
        &quantised_rows,
        &quantised_cols, 
        |qr_val, qc_val| &gradients[*qr_val][*qc_val]
    )
}

fn stack_arrays(array1: &Array, array2: &Array) -> Tensor3 {
    let n_rows = array1.len();
    let n_cols = array1[0].len();

    if n_rows != array2.len() || n_cols != array2[0].len() {
        panic!("Array shapes must match!");
    }

    zip_apply(&array1, &array2, |a, b| vec![*a, *b])
}

#[pyfunction]
pub fn perlin(rows: usize, cols: usize, scale: f32) -> Array {
    let gradients = random_normal(rows, cols);
    let (x_grid, y_grid) = make_grids(rows, cols, scale);

    // Indices for nearest grid points

    let x0 = quantise_grid(&x_grid, false);
    let x1 = quantise_grid(&x_grid, true);
    let y0 = quantise_grid(&y_grid, false);
    let y1 = quantise_grid(&y_grid, true);

    // Calculate the distance vectors from the grid points to the coordinates

    let dx0 = add_grid(&x_grid, &multiply_int_grid(-1.0, &x0));
    let dx1 = add_grid(&x_grid, &multiply_int_grid(-1.0, &x1));
    let dy0 = add_grid(&y_grid, &multiply_int_grid(-1.0, &y0));
    let dy1 = add_grid(&y_grid, &multiply_int_grid(-1.0, &y1));

    // Calculate the dot products between the gradient vectors and the distance vectors

    // TODO: this is the slow bit (mainly get_corner_gradients)
    let dot00 = dot_product_grid(
        get_corner_gradients(&gradients, &y0, &x0),
        // Turn inner vectors into references
        stack_arrays(&dx0, &dy0).iter().map(|row| row.iter().collect()).collect()
    );

    let dot10 = dot_product_grid(
        get_corner_gradients(&gradients, &y0, &x1),
        // Turn inner vectors into references
        stack_arrays(&dx1, &dy0).iter().map(|row| row.iter().collect()).collect()
    );

    let dot01 = dot_product_grid(
        get_corner_gradients(&gradients, &y1, &x0),
        // Turn inner vectors into references
        stack_arrays(&dx0, &dy1).iter().map(|row| row.iter().collect()).collect()
    );

    //let dot11 = dot_product_grid(
    //    get_corner_gradients(&gradients, &y1, &x1),
    //    // Turn inner vectors into references
    //    stack_arrays(&dx1, &dy1).iter().map(|row| row.iter().collect()).collect()
    //);

    let _dot11 = dot_product_grid(
        get_corner_gradients(&gradients, &y1, &x1),
        // Turn inner vectors into references
        stack_arrays(&dx1, &dy1).iter().map(|row| row.iter().collect()).collect()
    );

    interpolate(
        _Array::new(dx0),
        _Array::new(dy0),
        _Array::new(dot00),
        _Array::new(dot10),
        _Array::new(dot01),
        _Array::new(dot11),
    ).data
}

#[pymodule]
fn rust_perlin(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perlin, m)?)?;

    Ok(())
}
