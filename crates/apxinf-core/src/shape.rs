use crate::{Error, Result};

/// N-dimensional shape for tensors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Scalar (0-dimensional).
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// The dimension sizes.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Size of a specific dimension. Supports negative indexing.
    pub fn dim(&self, i: isize) -> Result<usize> {
        let idx = if i < 0 {
            let pos = self.ndim() as isize + i;
            if pos < 0 {
                return Err(Error::InvalidAxis {
                    axis: i as usize,
                    ndim: self.ndim(),
                });
            }
            pos as usize
        } else {
            i as usize
        };
        self.dims.get(idx).copied().ok_or(Error::InvalidAxis {
            axis: idx,
            ndim: self.ndim(),
        })
    }

    /// Compute row-major strides.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.ndim()];
        if self.ndim() > 0 {
            strides[self.ndim() - 1] = 1;
            for i in (0..self.ndim() - 1).rev() {
                strides[i] = strides[i + 1] * self.dims[i + 1];
            }
        }
        strides
    }

    /// Total bytes needed to store elements of this shape.
    pub fn total_bytes(&self, elem_size: usize) -> usize {
        self.numel() * elem_size
    }

    /// Check if two shapes are matmul-compatible: [..., M, K] @ [..., K, N] -> [..., M, N].
    pub fn matmul_shape(&self, other: &Shape) -> Result<Shape> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(Error::Other(
                "matmul requires at least 2D tensors".to_string(),
            ));
        }
        let m = self.dims[self.ndim() - 2];
        let k1 = self.dims[self.ndim() - 1];
        let k2 = other.dims[other.ndim() - 2];
        let n = other.dims[other.ndim() - 1];

        if k1 != k2 {
            return Err(Error::MatmulDimMismatch { m, k1, k2, n });
        }

        // For simplicity, output shape takes batch dims from self.
        let mut out_dims: Vec<usize> = self.dims[..self.ndim() - 2].to_vec();
        out_dims.push(m);
        out_dims.push(n);
        Ok(Shape::new(out_dims))
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec())
    }
}

// Convenience: Shape::from((m, n))
impl From<(usize, usize)> for Shape {
    fn from((a, b): (usize, usize)) -> Self {
        Shape::new(vec![a, b])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from((a, b, c): (usize, usize, usize)) -> Self {
        Shape::new(vec![a, b, c])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from((a, b, c, d): (usize, usize, usize, usize)) -> Self {
        Shape::new(vec![a, b, c, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numel() {
        assert_eq!(Shape::new(vec![2, 3, 4]).numel(), 24);
        assert_eq!(Shape::new(vec![1]).numel(), 1);
        assert_eq!(Shape::scalar().numel(), 1);
    }

    #[test]
    fn test_strides() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_dim_negative() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.dim(-1).unwrap(), 4);
        assert_eq!(s.dim(-2).unwrap(), 3);
        assert_eq!(s.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_matmul_shape() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![3, 4]);
        assert_eq!(a.matmul_shape(&b).unwrap(), Shape::new(vec![2, 4]));
    }

    #[test]
    fn test_matmul_shape_batch() {
        let a = Shape::new(vec![8, 2, 3]);
        let b = Shape::new(vec![8, 3, 4]);
        assert_eq!(a.matmul_shape(&b).unwrap(), Shape::new(vec![8, 2, 4]));
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![4, 5]);
        assert!(a.matmul_shape(&b).is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Shape::new(vec![2, 3, 4])), "[2, 3, 4]");
    }
}
