use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_sum() {
        let mut a = Mat {
            rows: 2,
            cols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        let b = Mat {
            rows: 2,
            cols: 2,
            data: vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        };

        Mat::sum(&mut a, &b);

        assert_eq!(a.data[0][0], 6.0);
        assert_eq!(a.data[0][1], 8.0);
        assert_eq!(a.data[1][0], 10.0);
        assert_eq!(a.data[1][1], 12.0);
    }

    #[test]
    fn test_mat_dot() {
        let a = Mat {
            rows: 2,
            cols: 3,
            data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        };
        let b = Mat {
            rows: 3,
            cols: 2,
            data: vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]],
        };
        let mut c = Mat {
            rows: 2,
            cols: 2,
            data: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        };

        Mat::dot(&mut c, &a, &b);

        assert_eq!(c.data[0][0], 58.0);
        assert_eq!(c.data[0][1], 64.0);
        assert_eq!(c.data[1][0], 139.0);
        assert_eq!(c.data[1][1], 154.0);
    }

    #[test]
    fn test_mat_fill() {
        let mut a = Mat {
            rows: 2,
            cols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };

        Mat::fill(&mut a, 5.0);

        assert_eq!(a.data[0][0], 5.0);
        assert_eq!(a.data[0][1], 5.0);
        assert_eq!(a.data[1][0], 5.0);
        assert_eq!(a.data[1][1], 5.0);
    }

    #[test]
    fn test_mat_row() {
        let mat = Mat {
            rows: 3,
            cols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        };

        let row = Mat::row(&mat, 1);

        assert_eq!(row.rows, 1);
        assert_eq!(row.cols, 2);
        assert_eq!(row.data, vec![vec![3.0, 4.0]]);
    }

    #[test]
    fn test_mat_copy() {
        let src = Mat {
            rows: 2,
            cols: 2,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };

        let mut dst = Mat {
            rows: 2,
            cols: 2,
            data: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        };

        Mat::copy(&mut dst, &src);

        assert_eq!(dst.data[0][0], 1.0);
        assert_eq!(dst.data[0][1], 2.0);
        assert_eq!(dst.data[1][0], 3.0);
        assert_eq!(dst.data[1][1], 4.0);
    }

    #[test]
    fn test_nn_forward() {
        let arch = vec![2, 3, 2];
        let mut nn = NN::new(&arch);

        nn.weights[0].data = vec![vec![0.5, 0.3, 0.1], vec![0.2, 0.4, 0.6]];
        nn.biases[0].data = vec![vec![0.1, 0.2, 0.3]];

        nn.weights[1].data = vec![vec![0.5, 0.2], vec![0.1, 0.3], vec![0.4, 0.6]];
        nn.biases[1].data = vec![vec![0.4, 0.1]];

        nn.activations[0].data = vec![vec![0.6, 0.7]];

        NN::forward(&mut nn);

        assert_eq!(nn.activations[1].data[0][0], 0.631_812_45);
        assert_eq!(nn.activations[1].data[0][1], 0.659_260_4);
    }
}
