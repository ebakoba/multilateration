//! [![github]](https://github.com/ebakoba/multilateration)&ensp;[![crates-io]](https://crates.io/crates/multilateration)&ensp;[![docs-rs]](https://docs.rs/multilateration)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K
//!
//! <br>
//!
//! This is a multilateration library implemented in Rust, which is loosly
//! based on Java [trilateration library](https://github.com/lemmingapex/trilateration).
//!
//!
//! <br>
//!
//! # Details
//!
//! - It exposes one function that uses a vector of `Measurement` struct as an input.
//!   Output will be the tuple struct called `Point` which contains the output coordinates
//!   as a vector in the first tuple slot
//!   
//!
//!   ```
//!   use multilateration::{multilaterate, Measurement, Point};
//!
//!   fn main() -> Result<()> {
//!     let measurements = vec![
//!       Measurement::new(Point(vec![1.0, 1.0, 1.0]), 1.0),
//!       Measurement::new(Point(vec![3.0, 1.0, 1.0]), 1.0),
//!       Measurement::new(Point(vec![2.0, 2.0, 1.0]), 1.0),
//!     ];
//!
//!     let coordinates = multilaterate(measurements).unwrap().0;
//!     println!("Coordinates are: {:?}", coordinates);
//!     Ok(())
//!   }
//!   ```
//!
//!   ```console
//!   Coordinates are: [2.0, 1.0000157132198315, 0.9943941804736127]
//!   ```
//!
//! <br>
//!
//! # Error conditions
//!
//! - Points have different dimensions
//! 
//! ```
//!   use multilateration::{multilaterate, Measurement, Point};
//!
//!   fn main() -> Result<()> {
//!     let measurements = vec![
//!       Measurement::new(Point(vec![1.0, 1.0]), 1.0),
//!       Measurement::new(Point(vec![3.0, 1.0, 1.0]), 1.0),
//!       Measurement::new(Point(vec![2.0, 2.0, 1.0]), 1.0),
//!     ];
//!
//!     let result = multilaterate(measurements);
//!     println!("Result is: {:?}", result);
//!     Ok(())
//!   }
//!   ```
//!
//!   ```console
//!   Result is: Err(All points must have the same dimensions)
//!   ```
//! - Points have no dimensions
//! 
//! ```
//!   use multilateration::{multilaterate, Measurement, Point};
//!
//!   fn main() -> Result<()> {
//!     let measurements = vec![
//!       Measurement::new(Point(vec![]), 1.0),
//!       Measurement::new(Point(vec![]), 1.0),
//!       Measurement::new(Point(vec![]), 1.0),
//!     ];
//!
//!     let result = multilaterate(measurements);
//!     println!("Result is: {:?}", result);
//!     Ok(())
//!   }
//!   ```
//!
//!   ```console
//!   Result is: Err(Points must contain at least one dimension)
//!   ```


use mathru::optimization::{
    LevenbergMarquardt,
    Optim
};
use mathru::algebra::linear::vector::vector::Vector;
use mathru::algebra::linear::matrix::matrix::Matrix;
use anyhow::{Result, anyhow, bail};

struct MultilaterationFunction {
    measurements: Vec<Measurement>,
}

impl MultilaterationFunction {
    pub fn new(measurements: Vec<Measurement>) -> MultilaterationFunction {
        MultilaterationFunction{
            measurements,
        }
    }

    pub fn estimate_intial_point(&self) -> Point {
        let position_dimensions = self.measurements[0].point.0.len();
        let number_of_measurements = self.measurements.len();

        let mut initial_position = vec![0f64; position_dimensions];
        for i in 0..number_of_measurements {
            for j in 0..position_dimensions {
                initial_position[j] = self
                    .measurements[i].point.0[j];
            }
        }
        for i in 0..position_dimensions {
            initial_position[i] /= number_of_measurements as f64;
        }

        Point(initial_position)
    }
}

impl Optim<f64> for MultilaterationFunction {
    fn eval(&self, input: &Vector<f64>) -> Vector<f64> {
        let mut result = vec![0f64; self.measurements.len()];

        for i in 0..self.measurements.len() {
            result[i] = 0f64;
            for j in 0..input.clone().convert_to_vec().len() {
                result[i] +=
                    f64::powf(
                        *input.get(j) -
                        self.measurements[i].point.0[j],
                    2f64
                );
            }
            result[i] -= f64::powf(self.measurements[i].distance, 2f64);
        }

        Vector::new_column(self.measurements.len(), result)
    }

    fn jacobian(&self, input: &Vector<f64>) -> Matrix<f64> {
        let input_length = input.clone().convert_to_vec().len();
        let data = vec![
            0f64;
            self.measurements.len() * input_length
        ];
        let mut matrix = Matrix::new(
            self.measurements.len(),
            input_length,
            data
        );

        for i in 0..self.measurements.len() {
            for j in 0..input_length {
                *matrix.get_mut(i, j) =
                    2f64 * input.get(j) - 
                    2f64 * self.measurements[i].point.0[j];
            }
        }

        matrix
    }
}

#[derive(Debug, Clone)]
pub struct Point(pub Vec<f64>);

#[derive(Debug)]
pub struct Measurement {
    point: Point,
    distance: f64
}

impl Measurement {
    pub fn new(point: Point, distance: f64) -> Measurement {
        Measurement {
            point,
            distance
        }
    }
}

fn validate_measurements(measurements: &[Measurement]) -> Result<()> {
    let point_dimensions: Vec<usize> = measurements.iter().map(|measurement| measurement.point.0.len()).collect();
    let min_length = *point_dimensions.iter().min().ok_or(anyhow!("Failed to calculate minimal dimension"))?;
    let max_length = *point_dimensions.iter().max().ok_or(anyhow!("Failed to calculate maximal dimension"))?;

    if min_length != max_length {
        bail!("All points must have the same dimensions");
    }
    if min_length < 1usize {
        bail!("Points must contain at least one dimension");
    }
    Ok(())
}

pub fn multilaterate(
    measurements: Vec<Measurement>
) -> Result<Point> {
    validate_measurements(&measurements)?;
    let multilateration_function = MultilaterationFunction::new(measurements);
    let optimization = LevenbergMarquardt::new(1000, -1f64, 1f64);
    let initial_point = multilateration_function.estimate_intial_point();
    let result = optimization.minimize(
        &multilateration_function,
        &Vector::new_column(
            initial_point.0.len(),
            initial_point.0
        )
    ).map_err(|_| anyhow!("Failed to calculate a result"))?;

    Ok(Point(result.arg().convert_to_vec()))
}

#[cfg(test)]
mod tests {
    use super::Point;
    use super::Measurement;
    use super::multilaterate;

    fn is_in_delta(
        delta: f64,
        value: f64,
        comparison: f64
    ) -> bool {
        value >= (comparison - delta) && value <= (comparison + delta)
    }

    #[test]
    fn multilat_1() {
        let measurements = vec![
            Measurement {
                point: Point(vec![5.0, -6.0]),
                distance: 8.06
            },
            Measurement {
                point: Point(vec![13.0, -15.0]),
                distance: 13.97
            },
            Measurement {
                point: Point(vec![21.0, -3.0]),
                distance: 23.32
            },
            Measurement {
                point: Point(vec![12.4, -21.2]),
                distance: 15.31
            },
        ];

        let result = multilaterate(measurements).unwrap().0;
        assert_eq!(result.len(), 2);
        let delta = 1f64;
        assert!(is_in_delta(delta, result[0], -0.6));
        assert!(is_in_delta(delta, result[1], -11.8));
    }

    #[test]
    fn multilat_2() {
        let measurements = vec![
            Measurement {
                point: Point(vec![1.0, 1.0]),
                distance: 0.5
            },
            Measurement {
                point: Point(vec![3.0, 1.0]),
                distance: 0.5
            },
            Measurement {
                point: Point(vec![2.0, 2.0]),
                distance: 0.5
            }
        ];

        let result = multilaterate(measurements).unwrap().0;
        assert_eq!(result.len(), 2);
        let delta = 0.4;
        assert!(is_in_delta(delta, result[0], 2.0));
        assert!(is_in_delta(delta, result[1], 1.0));
    }

    #[test]
    fn multilat_3() {
        let measurements = vec![
            Measurement {
                point: Point(vec![1.0, 1.0]),
                distance: 2.0
            },
            Measurement {
                point: Point(vec![3.0, 1.0]),
                distance: 2.0
            },
            Measurement {
                point: Point(vec![2.0, 2.0]),
                distance: 2.0
            }
        ];

        let result = multilaterate(measurements).unwrap().0;
        assert_eq!(result.len(), 2);
        let delta = 2.0;
        assert!(is_in_delta(delta, result[0], 2.0));
        assert!(is_in_delta(delta, result[1], 1.0));
    }

    #[test]
    fn multilat_4() {
        let measurements = vec![
            Measurement {
                point: Point(vec![1.0, 1.0]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![1.0, 1.0]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![3.0, 1.0]),
                distance: 1.0
            }
        ];

        let result = multilaterate(measurements).unwrap().0;
        assert_eq!(result.len(), 2);
        let delta = 0.5;
        assert!(is_in_delta(delta, result[0], 2.0));
        assert!(is_in_delta(delta, result[1], 1.0));
    }

    #[test]
    fn multilat_5() {
        let measurements = vec![
            Measurement {
                point: Point(vec![1.0, 1.0]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![3.0, 1.0]),
                distance: 1.0
            }
        ];

        let result = multilaterate(measurements).unwrap().0;
        assert_eq!(result.len(), 2);
        let delta = 0.5;
        assert!(is_in_delta(delta, result[0], 2.0));
        assert!(is_in_delta(delta, result[1], 1.0));
    }

    #[test]
    fn multilat_6() {
        let measurements = vec![
            Measurement {
                point: Point(vec![1.0, 1.0]),
                distance: 0.9
            },
            Measurement {
                point: Point(vec![3.0, 1.0]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![2.0, 2.0]),
                distance: 1.0
            }
        ];

        let result = multilaterate(measurements).unwrap().0;
        assert_eq!(result.len(), 2);
        let delta = 0.1;
        assert!(is_in_delta(delta, result[0], 2.0));
        assert!(is_in_delta(delta, result[1], 1.0));
    }

    #[test]
    fn multilat_7() {
        let measurements = vec![
            Measurement {
                point: Point(vec![1.0, 1.0, 1.0]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![3.0, 1.0, 1.0]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![2.0, 2.0, 1.0]),
                distance: 1.0
            }
        ];

        let result = multilaterate(measurements).unwrap().0;
        assert_eq!(result.len(), 3);
        let delta = 0.1;
        assert!(is_in_delta(delta, result[0], 2.0));
        assert!(is_in_delta(delta, result[1], 1.0));
        assert!(is_in_delta(delta, result[2], 1.0));
    }

    #[test]
    fn multilat_fails_on_different_dimension() {
        let measurements = vec![
            Measurement {
                point: Point(vec![1.0, 1.0 ]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![3.0, 1.0, 1.0]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![2.0, 1.0]),
                distance: 1.0
            }
        ];

        let result = multilaterate(measurements);
        assert!(result.is_err());
    }

    #[test]
    fn multilat_fails_on_zero_dimensions() {
        let measurements = vec![
            Measurement {
                point: Point(vec![]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![]),
                distance: 1.0
            },
            Measurement {
                point: Point(vec![]),
                distance: 1.0
            }
        ];

        let result = multilaterate(measurements);
        assert!(result.is_err());
    }
}
