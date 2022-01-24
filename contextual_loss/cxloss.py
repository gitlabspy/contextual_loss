import tensorflow as tf


def patch_decomposition(T_features):
    patch_size = 1
    patches_as_depth_vectors = tf.image.extract_patches(
        images=T_features,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patches_NHWC = tf.reshape(
        patches_as_depth_vectors,
        shape=[-1, patch_size, patch_size, tf.shape(patches_as_depth_vectors)[3]],
        name="patches_PHWC",
    )
    patches_HWCN = tf.transpose(patches_NHWC, perm=[1, 2, 3, 0], name="patches_HWCP")
    return patches_HWCN


@tf.function
def contextual_loss(x, y, h=0.5):
    N = tf.shape(x)[0]
    y_mu = tf.reduce_mean(y, axis=[0, 1, 2], keepdims=True)
    x_centered = x - y_mu
    y_centered = y - y_mu
    x_centered = x_centered / tf.norm(
        x_centered, ord="euclidean", axis=-1, keepdims=True
    )
    y_centered = y_centered / tf.norm(
        y_centered, ord="euclidean", axis=-1, keepdims=True
    )
    # d = x_centered * y_centered
    cosine_dist = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in range(N):
        x_i = tf.gather(x_centered, i, axis=0)[tf.newaxis]
        y_i = tf.gather(y_centered, i, axis=0)[tf.newaxis]
        patches_HWCN_i = patch_decomposition(y_i)
        cosine_dist_i = tf.nn.conv2d(
            x_i, patches_HWCN_i, strides=[1, 1, 1, 1], padding="VALID"
        )
        cosine_dist = cosine_dist.write(i, cosine_dist_i)
    d = cosine_dist.concat()
    d = -(d - 1) / 2
    d_min = tf.reduce_min(d, axis=-1, keepdims=True)
    relative_dist = d / (d_min + 1e-5)

    ## It causes NaN
    # w = tf.math.exp((1.0 - relative_dist) / h)
    # cx_ij = w / tf.reduce_sum(w, axis=-1, keepdims=True)

    ## Stable version:
    cx_ij = tf.math.softmax((1 - relative_dist), axis=-1)

    k_max_NC = tf.math.reduce_max(cx_ij, axis=[1, 2])
    CS = tf.reduce_mean(k_max_NC, axis=[1])
    CX_as_loss = 1 - CS
    CX_loss = -tf.math.log(1 - CX_as_loss)
    CX_loss = tf.reduce_mean(CX_loss)
    return CX_loss


if __name__ == "__main__":
    a = tf.random.normal((2, 64, 64, 256))
    b = tf.random.uniform((2, 64, 64, 256), -1.0, 1.0)
    print(contextual_loss(a, b, 0.1))
    print(contextual_loss(a, a, 0.1))
    print(contextual_loss(b, b, 0.1))
