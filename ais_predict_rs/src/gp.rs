use nalgebra as na;
#[derive(Clone, Debug)]
pub struct GaussianProcess<T: na::ComplexField> {
    chol_l: na::Cholesky<T, na::Dynamic>,
    alpha: na::MatrixXx2<T>,
    train_x: na::MatrixXx2<T>,
    lengthscale: T,
}

impl<T: na::ComplexField> GaussianProcess<T> {
    fn kernel_cov<'a, 'b, R1, R2, S1, S2>(
        x1: &'a na::Matrix<T, R1, na::U2, S1>,
        x2: &'b na::Matrix<T, R2, na::U2, S2>,
        lengthscale: T,
    ) -> na::OMatrix<T, R1, R2>
    where
        R1: na::Dim,
        R2: na::Dim,
        S1: na::storage::Storage<T, R1, na::U2>,
        S2: na::storage::Storage<T, R2, na::U2>,
        na::DefaultAllocator: na::allocator::Allocator<T, R1, R2>,
    {
        let iter = x1
            .row_iter()
            .flat_map(|r1| x2.row_iter().map(move |r2| (r1 - r2) / lengthscale))
            .map(|d| (-d.dot(&d)).exp());
        return na::OMatrix::<T, R1, R2>::from_iterator_generic(
            na::Dim::from_usize(x1.nrows()),
            na::Dim::from_usize(x2.nrows()),
            iter,
        );
    }

    pub fn new(
        train_x: na::MatrixXx2<T>,
        train_y: na::MatrixXx2<T>,
        lengthscale: T,
        noise: T,
    ) -> Option<Self> {
        let k = Self::kernel_cov(&train_x, &train_x, lengthscale)
            + na::DMatrix::<T>::from_diagonal_element(
                train_x.nrows(),
                train_x.nrows(),
                noise * noise,
            );
        let train_mat = na::Cholesky::new_unchecked(k);
        let mut alpha = train_y;
        train_mat.solve_mut(&mut alpha);
        return Some(GaussianProcess::<T> {
            chol_l: train_mat,
            train_x: train_x,
            alpha: alpha,
            lengthscale: lengthscale,
        });
    }
    pub fn predict_f<S>(
        &self,
        x: &na::Matrix<T, na::U1, na::U2, S>,
    ) -> Option<(na::RowVector2<T>, na::Matrix2<T>)>
    where
        S: na::storage::Storage<T, na::U1, na::U2>,
    {
        let kernel = GaussianProcess::kernel_cov(x, &self.train_x, self.lengthscale);
        let f = &kernel * (&self.alpha);
        let v = self
            .chol_l
            .l_dirty()
            .solve_lower_triangular(&kernel.transpose())?;
        let std = GaussianProcess::kernel_cov(x, x, self.lengthscale).x - v.dot(&v);
        let std = na::Matrix2::<T>::from_diagonal_element(std);
        return Some((f, std));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gp() {
        let train_x = na::MatrixXx2::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let train_y = 2.0 * train_x.clone();

        let gp = GaussianProcess::<f64>::new(train_x, train_y, 2.0).expect("Unable to create gp");
        let (f, std) = gp
            .predict_f(&na::RowVector2::<f64>::new(1.0, 2.0))
            .expect("Unable to predict");
    }
}
