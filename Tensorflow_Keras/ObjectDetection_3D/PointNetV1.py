import tensorflow as tf

g_type2class={'Abnormal': 0}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Abnormal': 0}
g_type_mean_size = {'Abnormal': np.array([3.88311640418,1.62856739989,1.52563191462])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs

for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]
    
def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device("/cpu:0"):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer
    Returns:
    Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):

    with tf.variable_scope(scope) as sc:
        assert(data_format=='NHWC' or data_format=='NCHW')
        if data_format == 'NHWC':
            num_in_channels = inputs.get_shape()[-1].value
        elif data_format=='NCHW':
            num_in_channels = inputs.get_shape()[1].value
        kernel_shape = [kernel_size, num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        outputs = tf.nn.conv1d(inputs, kernel,
                               stride=stride,
                               padding=padding,
                               data_format=data_format)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn',
                                      data_format=data_format)

    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):

    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        assert(data_format=='NHWC' or data_format=='NCHW')
        if data_format == 'NHWC':
            num_in_channels = inputs.get_shape()[-1].value
        elif data_format=='NCHW':
            num_in_channels = inputs.get_shape()[1].value
        kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn',
                                        data_format=data_format)

        if activation_fn is not None: outputs = activation_fn(outputs)
        return outputs

def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w, num_output_channels, num_in_channels] # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1], padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels], tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn: outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn')

        if activation_fn is not None: outputs = activation_fn(outputs)

        return outputs

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):

    with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn: outputs = batch_norm_for_conv3d(outputs, is_training, bn_decay=bn_decay, scope='bn')

    if activation_fn is not None: outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
    inputs: 2-D tensor BxN
    num_outputs: int

    Returns:
    Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                 tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn: outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None: outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.
    Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints

    Returns:
    Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D avg pooling.
    Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints

    Returns:
    Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D max pooling.
    Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints

    Returns:
    Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.max_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D avg pooling.
    Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints

    Returns:
    Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.avg_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def batch_norm_template_unused(inputs, is_training, scope, moments_dims, bn_decay):
    """ NOTE: this is older version of the util func. it is deprecated.
    Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
    Return:
      normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = _variable_on_cpu(name='beta',shape=[num_channels],
                                initializer=tf.constant_initializer(0))
        gamma = _variable_on_cpu(name='gamma',shape=[num_channels],
                                initializer=tf.constant_initializer(1.0))
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
        # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = tf.cond(is_training,
                                   lambda: ema.apply([batch_mean, batch_var]),
                                   lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
    Return:
      normed:        batch-normalized maps
    """
    bn_decay = bn_decay if bn_decay is not None else 0.9
    return tf.contrib.layers.batch_norm(inputs, 
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope=scope,
                                      data_format=data_format)


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
    Return:
      normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, data_format):
    """ Batch normalization on 1D convolutional maps.

    Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
    Return:
      normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay, data_format)



  
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
    """ Batch normalization on 2D convolutional maps.

    Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
    Return:
      normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay, data_format)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
    Return:
      normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.
    Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints
    Returns:
    tensor variable
    """
    with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs

def get_instance_seg_v1_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points):
    ''' 3D instance segmentation PointNet v1 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        end_points: dict
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
        end_points: dict
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    net = tf.expand_dims(point_cloud, 2)

    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

    logits = tf_util.conv2d(net, 2, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    logits = tf.squeeze(logits, [2]) # BxNxC
    return logits, end_points
 

def get_3d_box_estimation_v1_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points):
    ''' 3D Box Estimation PointNet v1 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    ''' 
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
        padding='VALID', scope='maxpool2')
    net = tf.squeeze(net, axis=[1,2])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, scope='fc1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, scope='fc2', bn=True,
        is_training=is_training, bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
    return output, end_points


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None):
    ''' Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
    Output:
        end_points: dict (map from name strings to TF tensors)
    '''
    end_points = {}
    
    # 3D Instance Segmentation PointNet
    logits, end_points = get_instance_seg_v1_net(\
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points)
    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = \
        point_cloud_masking(point_cloud, logits, end_points)

    # T-Net and coordinate translation
    center_delta, end_points = get_center_regression_net(\
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points)
    stage1_center = center_delta + mask_xyz_mean # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = \
        object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output, end_points = get_3d_box_estimation_v1_net(\
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

    return end_points