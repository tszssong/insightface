import mxnet as mx

bn_momentum = 0.9

def BK(data):
    return mx.symbol.BlockGrad(data=data)

# - - - - - - - - - - - - - - - - - - - - - - -
# Fundamental Elements
def BN(data, fix_gamma=False, momentum=bn_momentum, name=None):
    bn     = mx.symbol.BatchNorm( data=data, fix_gamma=fix_gamma, momentum=bn_momentum, name=('%s__bn'%name))
    return bn

def AC(data, act_type='relu', name=None):
    act    = mx.symbol.Activation(data=data, act_type=act_type, name=('%s__%s' % (name, act_type)))
    return act

def BN_AC(data, momentum=bn_momentum, name=None):
    bn     = BN(data=data, name=name, fix_gamma=False, momentum=momentum)
    bn_ac  = AC(data=bn,   name=name)
    return bn_ac

def Conv(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, no_bias=True, w=None, b=None, attr=None, num_group=1):
    Convolution = mx.symbol.Convolution
    if w is None:
        conv     = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, attr=attr)
    else:
        if b is None:
            conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=no_bias, weight=w, attr=attr)
        else:
            conv = Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s__conv' %name), no_bias=False, bias=b, weight=w, attr=attr)
    return conv

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < CVPR >
def Conv_BN(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov    = Conv(   data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_bn = BN(     data=cov,    name=('%s__bn' % name))
    return cov_bn

def Conv_BN_AC(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    cov_bn = Conv_BN(data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_ba = AC(     data=cov_bn, name=('%s__ac' % name))
    return cov_ba

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < ECCV >
def BN_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    bn_cov = Conv(   data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return bn_cov

def AC_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    ac     = AC(     data=data,   name=('%s__ac' % name))
    ac_cov = Conv(   data=ac,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return ac_cov

def BN_AC_Conv(data, num_filter,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=('%s__bn' % name))
    ba_cov = AC_Conv(data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return ba_cov

def Residual_Unit(data, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    # main part
    conv_m1 = BN_AC_Conv(data=data,    num_filter=num_mid, kernel=(1, 1), pad=(0, 0), name=('%s_conv-m1' % name))
    conv_m2 = BN_AC_Conv(data=conv_m1, num_filter=num_mid, kernel=(3, 3), pad=(1, 1), name=('%s_conv-m2' % name), stride=stride, num_group=g)
    conv_m3 = BN_AC_Conv(data=conv_m2, num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv-m3' % name))
    # adapter
    if first_block:
        data  = BN_AC_Conv(data=data,  num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv-w1' % name), stride=stride)
    out   = mx.symbol.ElementWiseSum(*[data, conv_m3],                                name=('%s_sum' % name))
    return out

def Shuffle_Unit_1(data, num_in, name, stride=(1, 1), g=1):
    num_slice = num_in/2
    print(str(num_in)+' '+str(num_slice))
    slice1 = mx.symbol.slice_axis(data=data, axis=1, begin=0,         end=num_slice,       name=('%s_slice1' % name))
    slice2 = mx.symbol.slice_axis(data=data, axis=1, begin=num_slice,         end=num_in,       name=('%s_slice2' % name))
    
    # main part
    conv_m1 = Conv_BN_AC(data=slice2,  num_filter=num_slice, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m1' % name), num_group=1)
    conv_m2 = Conv_BN(data=conv_m1,    num_filter=num_slice, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_conv-m2' % name), num_group=num_slice)
    conv_m3 = Conv_BN_AC(data=conv_m2, num_filter=num_slice, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m3' % name), num_group=1)
    # adapter
    
    out   = mx.symbol.Concat(slice1, conv_m3, name=('%s_concat' % name))
    return out

def Shuffle_Unit_2(data, num_in, name, stride=(1, 1), g=1):
    # main part
    conv_m1 = Conv_BN_AC(data=data,    num_filter=num_in, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m1' % name), num_group=1)
    conv_m2 = Conv_BN(data=conv_m1,    num_filter=num_in, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_conv-m2' % name), num_group=num_in)
    conv_m3 = Conv_BN_AC(data=conv_m2, num_filter=num_in, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m3' % name), num_group=1)
    
    # adapter
    conv_m4 = Conv_BN(data=data,       num_filter=num_in, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_conv-m4' % name), num_group=num_in)    
    conv_m5 = Conv_BN_AC(data=conv_m4, num_filter=num_in, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m5' % name), num_group=1)

    out   = mx.symbol.Concat(conv_m3, conv_m5, name=('%s_concat' % name))
    return out

def Shuffle_Unit_1_L(data, num_in, name, stride=(1, 1), g=1):
    num_slice = num_in/2
    print(str(num_in)+' '+str(num_slice))
    slice1 = mx.symbol.slice_axis(data=data, axis=1, begin=0,         end=num_slice,       name=('%s_slice1' % name))
    slice2 = mx.symbol.slice_axis(data=data, axis=1, begin=num_slice,         end=num_in,       name=('%s_slice2' % name))

    # main part
    conv_m1 = Conv_BN_AC(data=slice2,  num_filter=num_slice, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m1' % name), num_group=1)
    conv_m2 = Conv_BN(data=conv_m1,    num_filter=num_slice, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_conv-m2' % name), num_group=num_slice)
    conv_m3 = Conv_BN(data=conv_m2, num_filter=num_slice, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m3' % name), num_group=1)
    # adapter

    concat   = mx.symbol.Concat(slice1, conv_m2, name=('%s_concat' % name))
    out = Conv_BN_AC(data=concat, num_filter=num_in, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_shuffle' % name), num_group=1)
    return out

def Shuffle_Unit_2_L(data, num_in, num_mid, name, stride=(1, 1), g=1):
    # main part
    conv_m1 = Conv_BN_AC(data=data,    num_filter=num_mid, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m1' % name), num_group=1)
    conv_m2 = Conv_BN(data=conv_m1,    num_filter=num_mid, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_conv-m2' % name), num_group=num_mid)
    conv_m3 = Conv_BN(data=conv_m2, num_filter=num_mid, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m3' % name), num_group=1)

    # adapter
    conv_m4 = Conv_BN(data=data,       num_filter=num_in, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_conv-m4' % name), num_group=num_in)
    conv_m5 = Conv_BN(data=conv_m4, num_filter=num_mid, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_conv-m5' % name), num_group=1)

    concat   = mx.symbol.Concat(conv_m2, conv_m5, name=('%s_concat' % name))
    num_out = num_mid * 2
    out = Conv_BN_AC(data=concat, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name=('%s_shuffle' % name), num_group=1)
    return out

def GloRe_Unit(data, settings, name):
    # e.g.:
    # - settings = (num_in, num_mid)

    num_in, num_mid = settings
    num_state = int(2 * num_mid)
    num_node  = int(1 * num_mid)

    # reduce sampling space if is required
    x_pooled = data

    # generate "state" for each node:
    # (N, num_in, H_sampled, W_sampled) --> (N, num_state, H_sampled, W_sampled)
    #                                   --> (N, num_state, H_sampled*W_sampled)
    x_state = BN_AC_Conv(data=x_pooled, num_filter=num_state,  kernel=(1, 1), pad=(0, 0), name=('%s_conv-state' % name))
    x_state_reshaped = mx.symbol.reshape(x_state, shape=(0, 0, -1),                       name=('%s_conv-state-reshape' % name))

    # prepare "projection" funcrion (pixel space -> interaction space):
    # (N, num_in, H_sampled, W_sampled) --> (N, num_node, H_sampled, W_sampled)
    #                                   --> (N, num_node, H_sampled*W_sampled)
    x_proj = BN_AC_Conv(data=x_pooled, num_filter=num_node, kernel=(1, 1), pad=(0, 0),   name=('%s_conv-proj' % name))
    x_proj_reshaped = mx.symbol.reshape(x_proj, shape=(0, 0, -1),                        name=('%s_conv-proj-reshape' % name))

    # prepare "reverse projection" function (interaction spave -> pixel spave)
    x_rproj_reshaped = x_proj_reshaped

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Projection: pixel space -> interaction space
    # (N, num_state, H_sampled*W_sampled) x (N, num_node, H_sampled*W_sampled)T --> (N, num_state, num_node)
    x_n_state = mx.symbol.batch_dot(lhs=x_state_reshaped, rhs=x_proj_reshaped, transpose_b=True,  name=('%s_proj' % name))

    # ------

    # [2] Interaction Network (relation)
    x_n_rel = x_n_state

    # -> propogate information
    #     - G * X: propogate state
    #     (N, num_state, num_node+1) -permute-> (N, num_node+1, num_state) -conv-> (N, num_node, num_state)
    x_n_rel  = mx.symbol.transpose(data=x_n_rel, axes=(0,2,1),                           name=('%s_GCN-G-permute1' % name))
    x_n_rel  = BN_AC_Conv(data=x_n_rel, num_filter=num_node,  kernel=1, pad=0, stride=1, name=('%s_GCN-G' % name))
    x_n_rel  = mx.symbol.transpose(data=x_n_rel, axes=(0,2,1),                           name=('%s_GCN-G-permute2' % name))
    # -> add residual to original value to stable optimization
    x_n_rel  = mx.symbol.ElementWiseSum(*[x_n_state, x_n_rel],                           name=('%s_GCN-G_sum' % name))

    #     - H * W: compute new state's residual
    #     (N, num_node, num_state) -permute-> (N, num_state, num_node) -conv-> (N, num_state, num_node)
    x_n_state_new = BN_AC_Conv(data=x_n_rel, num_filter=num_state, kernel=1, pad=0, stride=1, name=('%s_GCN-GHW' % name))

    # ------
    # Reverse Projection: instance space -> pixel space
    # (N, num_node, num_state) x (N, num_node, H*W) --> (N, num_state, H*W)
    #                                               --> (N, num_state, H, W)
    x_out =      mx.symbol.batch_dot(lhs=x_n_state_new, rhs=x_rproj_reshaped,         name=('%s_reverse-proj' % name))
    x_out =      mx.symbol.reshape_like(x_out, rhs=x_state,                           name=('%s_reverse-proj-reshape' % name))

    # expend dimension (2nd fc)
    x_out = BN_AC_Conv(data=x_out,    num_filter=num_in,   kernel=(1, 1), pad=(0, 0), name=('%s_fc-2' % name))

    out   = mx.symbol.ElementWiseSum(*[data, x_out],                                  name=('%s_sum' % name))

    return out


def squeeze_excitation_block(name, data, num_filter, ratio):
    squeeze = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name=name + '_squeeze')
    squeeze = mx.symbol.Flatten(data=squeeze, name=name + '_flatten')
    excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*ratio), name=name + '_excitation1')
    excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name + '_excitation1_relu')
    excitation = mx.symbol.FullyConnected(data=excitation, num_hidden=num_filter, name=name + '_excitation2')
    excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name + '_excitation2_sigmoid')
    scale = mx.symbol.broadcast_mul(data, mx.symbol.reshape(data=excitation, shape=(-1, num_filter, 1, 1)))
    return scale


def Act(data, act_type, name):
    #ignore param act_type, set it in this function
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    return body

def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=stride, pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut, act_type='relu', name=name + '_relu3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn2 = mx.symbol.broadcast_mul(bn2, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut, act_type='relu', name=name + '_relu3')

def residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit1')
    if bottle_neck:
        conv1 = Conv(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn3 + shortcut, act_type='relu', name=name + '_relu3')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn2 = mx.symbol.broadcast_mul(bn2, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return Act(data=bn2 + shortcut, act_type='relu', name=name + '_relu3')

def residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit2')
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = Conv(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=conv3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          conv3 = mx.symbol.broadcast_mul(conv3, body)
        if dim_match:
            shortcut = data
        else:
            shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = Conv(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=conv2, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          conv2 = mx.symbol.broadcast_mul(conv2, body)
        if dim_match:
            shortcut = data
        else:
            shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit3')
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = Conv(data=bn1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = Act(data=bn2, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act2 = Act(data=bn3, act_type='relu', name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn4 = mx.symbol.broadcast_mul(bn4, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn4 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = Act(data=bn2, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn3 + shortcut

def residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    """Return ResNeXt Unit symbol for building ResNeXt
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    assert(bottle_neck)
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    num_group = 32
    #print('in unit3')
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    conv1 = Conv(data=bn1, num_group=num_group, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act1 = Act(data=bn2, act_type='relu', name=name + '_relu1')
    conv2 = Conv(data=act1, num_group=num_group, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act2 = Act(data=bn3, act_type='relu', name=name + '_relu2')
    conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')

    if use_se:
      #se begin
      body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
      body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                name=name+"_se_conv1", workspace=workspace)
      body = Act(data=body, act_type='relu', name=name+'_se_relu1')
      body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                name=name+"_se_conv2", workspace=workspace)
      body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
      bn4 = mx.symbol.broadcast_mul(bn4, body)
      #se end

    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return bn4 + shortcut

def residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):

    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    #print('in unit3')
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = Conv(data=bn1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = Act(data=bn2, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act2 = Act(data=bn3, act_type='relu', name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')

        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn4, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn4 = mx.symbol.broadcast_mul(bn4, body)
          #se end
        if dim_match:
            shortcut = data
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            return bn4+shortcut
        else:
          return bn4

    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = Act(data=bn2, act_type='relu', name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        if use_se:
          #se begin
          body = mx.sym.Pooling(data=bn3, global_pool=True, kernel=(7, 7), pool_type='avg', name=name+'_se_pool1')
          body = Conv(data=body, num_filter=num_filter//16, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv1", workspace=workspace)
          body = Act(data=body, act_type='relu', name=name+'_se_relu1')
          body = Conv(data=body, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                    name=name+"_se_conv2", workspace=workspace)
          body = mx.symbol.Activation(data=body, act_type='sigmoid', name=name+"_se_sigmoid")
          bn3 = mx.symbol.broadcast_mul(bn3, body)
          #se end

        if dim_match:
            shortcut = data
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            return bn3+shortcut
        else:
          return bn3

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
  uv = kwargs.get('version_unit', 3)
  version_input = kwargs.get('version_input', 1)
  if uv==1:
    if version_input==0:
      return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==2:
    return residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  elif uv==4:
    return residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
  else:
    if version_input<=1:
      return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
    else:
      return residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)

def resnet(units, num_stages, filter_list, num_classes, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    version_se = kwargs.get('version_se', 1)
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    print(version_se, version_input, version_output, version_unit)
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = data-127.5
    data = data*0.0078125
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if version_input==0:
      body = Conv(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type='relu', name='relu0')
      body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
      body = data
      body = Conv(data=body, num_filter=filter_list[0], kernel=(3,3), stride=(1,1), pad=(1, 1),
                                no_bias=True, name="conv0", workspace=workspace)
      body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
      body = Act(data=body, act_type='relu', name='relu0')
      #body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
      if version_input==0:
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      else:
        body = residual_unit(body, filter_list[i+1], (2, 2), False,
          name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      for j in range(units[i]-1):
        body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i+1, j+2),
          bottle_neck=bottle_neck, **kwargs)

    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    return fc1

def get_symbol(num_classes, num_layers, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    if num_layers >= 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  **kwargs)
