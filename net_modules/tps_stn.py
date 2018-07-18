import tensorflow as tf
import numpy as np
import zutils.tf_math_funcs as tmf
from net_modules.spatial_transformer import _interpolate as generic_interpolate


class TPS:

    @staticmethod
    def TPS(nx, ny, cp, p, fp_more=None):
        """Thin Plate Spline Spatial Transformer Layer
        TPS control points are arranged in a regular grid.
        U : float Tensor
            shape [num_batch, height, width, num_channels].
        nx : int
            The number of control points on x-axis
        ny : int
            The number of control points on y-axis
        cp : float Tensor
            control points. shape [num_batch, nx*ny, 2].
        p: float Tensor
            transform points. shape [num_batch, num_points, 2].
        """

        T, fp = TPS._solve_system(cp, nx, ny, fp_more=fp_more)
        x_s, y_s = TPS._transform_xy(T, fp, p[:, :, 0], p[:, :, 1])
        output = tf.concat(
            [tf.expand_dims(x_s, axis=-1), tf.expand_dims(y_s, axis=-1)],
            axis=2
        )

        return output

    @staticmethod
    def TPS_STN(U, nx, ny, cp, out_size, fp_more=None):
        """Thin Plate Spline Spatial Transformer Layer
        TPS control points are arranged in a regular grid.
        U : float Tensor
            shape [num_batch, height, width, num_channels].
        nx : int
            The number of control points on x-axis
        ny : int
            The number of control points on y-axis
        cp : float Tensor
            control points. shape [num_batch, nx*ny, 2].
        out_size: tuple of two ints
            The size of the output of the network (height, width)
        ----------
        Reference :
          https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
        """

        T, fp = TPS._solve_system(cp, nx, ny, fp_more=fp_more)
        output = TPS._transform_img(T, fp, U, out_size)
        return output

    @staticmethod
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    @staticmethod
    def _interpolate(im, x, y, out_size):
        output = generic_interpolate(im, x, y, out_size)
        return output

    @staticmethod
    def _meshgrid(height, width, fp):
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        return TPS._detailedgrid(x_t_flat, y_t_flat, fp)

    @staticmethod
    def _detailedgrid(x_t_flat, y_t_flat, fp):

        x_t_flat_b = tf.expand_dims(x_t_flat, axis=1)   # [1 or n, 1, h*w]  h*w==num_points
        y_t_flat_b = tf.expand_dims(y_t_flat, axis=1)   # [1 or n, 1, h*w]

        p_batch_size = tmf.get_shape(x_t_flat)[0]
        num_batch = tmf.get_shape(fp)[0]
        if p_batch_size == 1:
            x_t_flat_g = tf.tile(x_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
            y_t_flat_g = tf.tile(y_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
        else:
            x_t_flat_g = x_t_flat_b
            y_t_flat_g = y_t_flat_b
            assert num_batch == p_batch_size, "batch sizes do not match"

        px = tf.expand_dims(fp[:, :, 0], 2)  # [n, nx*ny, 1]
        py = tf.expand_dims(fp[:, :, 1], 2)  # [n, nx*ny, 1]
        d = tf.sqrt(tf.pow(x_t_flat_b - px, 2.) + tf.pow(y_t_flat_b - py, 2.))
        r = tf.pow(d, 2) * tf.log(d + 1e-6)  # [n, nx*ny, h*w]

        ones = tf.ones_like(x_t_flat_g)  # [n, 1, h*w]

        grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r], 1)  # [n, nx*ny+3, h*w]
        return grid

    @staticmethod
    def _transform_internal(T, grid):
        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        T_g = tf.matmul(T, grid)  # MARK
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])
        return x_s_flat, y_s_flat

    @staticmethod
    def _transform_xy(T, fp, x, y):

        assert len(tmf.get_shape(x)) == 2 and tmf.get_shape(x) == tmf.get_shape(y), \
            "x and y must be rank 2 and of the same size"

        grid = TPS._detailedgrid(x, y, fp)
        x_s_flat, y_s_flat = TPS._transform_internal(T, grid)
        x_s = tf.reshape(x_s_flat, tmf.get_shape(x))
        y_s = tf.reshape(y_s_flat, tmf.get_shape(y))
        return x_s, y_s

    @staticmethod
    def _transform_img(T, fp, input_dim, out_size):

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = out_size[0]
        out_width = out_size[1]
        grid = TPS._meshgrid(out_height, out_width, fp)  # [2, h*w]

        x_s_flat, y_s_flat = TPS._transform_internal(T, grid)

        num_batch = tf.shape(input_dim)[0]
        num_channels = tf.shape(input_dim)[3]

        input_transformed = TPS._interpolate(
            input_dim, x_s_flat, y_s_flat, out_size)

        output = tf.reshape(
            input_transformed,
            tf.stack([num_batch, out_height, out_width, num_channels]))
        return output

    @staticmethod
    def _spline_grid(nx, ny):
        gx = 2. / nx  # grid x size
        gy = 2. / ny  # grid y size
        cx = -1. + gx / 2.  # x coordinate
        cy = -1. + gy / 2.  # y coordinate

        p_ = np.empty([nx * ny, 3], dtype='float32')
        i = 0
        for _ in range(ny):
            for _ in range(nx):
                p_[i, :] = 1, cx, cy
                i += 1
                cx += gx
            cx = -1. + gx / 2
            cy += gy

        return p_

    @staticmethod
    def _solve_system_final(cp, fp, W_inv):
        cp_pad = tf.pad(cp + fp, [[0, 0], [0, 3], [0, 0]], "CONSTANT")
        T = tf.matmul(W_inv, cp_pad)
        T = tf.transpose(T, [0, 2, 1])
        return T

    @staticmethod
    def _solve_system_cp(cp, fp):

        fp = tf.stop_gradient(fp)
        # fp = [batch, num_keypoints, 2]
        num_batch = tmf.get_shape(fp)[0]
        p = tf.concat([tf.ones_like(fp[:, :, 0:1]), fp], axis=2)  # [batch, num_keypoints, 3]
        p_1 = tf.expand_dims(p, axis=2)  # [batch, num_keypoints, 1, 3]
        p_2 = tf.expand_dims(p, axis=1)  # [batch, 1, num_keypoints, 3]
        d = tf.sqrt(tf.reduce_sum((p_1 - p_2) ** 2, axis=3))  # [batch, num_keypoints, num_keypoints]
        r = d * d * tf.log(d * d + 1e-5)

        W = tf.concat([
            tf.concat([p, r], axis=2),
            tf.concat([tf.zeros([num_batch, 3, 3], dtype=p.dtype), tf.transpose(p, [0, 2, 1])], axis=2),
        ], axis=1)  # [batch, num_keypoints+3, num_keypoints+3]

        if False:
            W_inv = tf.matrix_inverse(W)
        else:
            # better stability for degenerated control points
            s, u, v = tf.svd(W)
            s_inv = tf.where(s > 1e-6, 1/s, tf.zeros_like(s))
            s_inv_m = tmf.diag_to_matrix(s_inv)
            W_inv = tf.matmul(tf.matmul(v, s_inv_m), tf.transpose(u, [0, 2, 1]))

        W_inv = tf.stop_gradient(W_inv)
        T = TPS._solve_system_final(cp, fp, W_inv)
        return T

    @staticmethod
    def _solve_system(cp, nx=None, ny=None, fp_more=None):

        if nx is None or ny is None or nx <= 0 or ny <= 0:
            assert fp_more is not None, "fp_more should be specified"
            fp = fp_more  # # [batch, num_keypoints, 2]
            T = TPS._solve_system_cp(cp, fp)

        else:
            p_ = TPS._spline_grid(nx, ny)
            num_batch = tmf.get_shape(cp)[0]
            fp = tf.constant(p_[:, 1:], dtype='float32')  # [nx*ny, 2]
            fp = tf.expand_dims(fp, 0)  # [1, nx*ny, 2]
            fp = tf.tile(fp, tf.stack([num_batch, 1, 1]))  # [n, nx*ny, 2]

            if fp_more is None:
                p_1 = p_.reshape([nx * ny, 1, 3])
                p_2 = p_.reshape([1, nx * ny, 3])
                d = np.sqrt(np.sum((p_1 - p_2) ** 2, 2))  # [nx*ny, nx*ny]
                r = d * d * np.log(d * d + 1e-5)
                W = np.zeros([nx * ny + 3, nx * ny + 3], dtype='float32')
                W[:nx * ny, 3:] = r
                W[:nx * ny, :3] = p_
                W[nx * ny:, 3:] = p_.T

                W_inv = np.linalg.inv(W)
                W_inv_t = tf.constant(W_inv, dtype='float32')  # [nx*ny+3, nx*ny+3]
                W_inv_t = tf.expand_dims(W_inv_t, 0)  # [1, nx*ny+3, nx*ny+3]
                W_inv_t = tf.tile(W_inv_t, tf.stack([num_batch, 1, 1]))

                T = TPS._solve_system_final(cp, fp, W_inv_t)

            else:

                fp = tf.concat([fp, fp_more], axis=1)
                T = TPS._solve_system_cp(cp, fp)

        return T, fp


TPS_STN = TPS.TPS_STN
TPS_TRANSFORM = TPS.TPS

