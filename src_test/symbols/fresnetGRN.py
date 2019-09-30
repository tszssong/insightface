# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#import mxnet as mx
#import symbol_utils
#
#def Act(data, act_type, name):
#    #ignore param act_type, set it in this function
#    #body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
#    body = mx.symbol.Activation(data=data,act_type='relu')
#    #act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
#    return body
#
#def ConvBNAct(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
#    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
#    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
#    act = Act(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
#    return act
#
#def ConvBN(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
#    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
#    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
#    return bn
#
#def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
#    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
#    return conv


import mxnet as mx
import symbol_utils
from symbol_basic import *
#from symbol_NLBlock import *

k_sec  = {  2:  2, \
        3:  2, \
        4:  2, \
        5:  2  }

def get_before_pool(data):
    #data = mx.symbol.Variable(name="data")

    # conv1
    conv1_x = Conv(data=data,  num_filter=64,  kernel=(3,3), name='conv1', pad=(1,1), stride=(2,2))
    conv1_x = BN_AC(data=conv1_x, name='conv1')
    #conv1_x = mx.symbol.Pooling(data=conv1_x, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

    # conv2
    num_in  = 64
    num_mid = 64
    num_out = 256
    for i in range(1,k_sec[2]+1):
        conv2_x = Residual_Unit(data=(conv1_x if i==1 else conv2_x),
                num_in=(num_in if i==1 else num_out),
                num_mid=num_mid,
                num_out=num_out,
                name="conv2_B%02d"%i,
                first_block=(i==1), stride=((1,1) if (i==1) else (1,1)))
        if i in []:
            conv2_x = GloRe_Unit(data=conv2_x,
                    settings=(num_out, int(num_out/2), int(num_out/2)),
                    name="conv2_B%02d_extra"%i)#, stride=(2,2))

            # conv3
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[3]+1):
        conv3_x = Residual_Unit(data=(conv2_x if i==1 else conv3_x),
                num_in=(num_in if i==1 else num_out),
                num_mid=num_mid,
                num_out=num_out,
                name="conv3_B%02d"%i,
                first_block=(i==1), stride=((2,2) if (i==1) else (1,1)))
        if i in []:
            conv3_x = GloRe_Unit(data=conv3_x,
                    settings=(num_out, int(num_out/2), int(num_out/2)),
                    name="conv3_B%02d_extra"%i)#, stride=(1,1))

            # conv4
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[4]+1):
        conv4_x = Residual_Unit(data=(conv3_x if i==1 else conv4_x),
                num_in=(num_in if i==1 else num_out),
                num_mid=num_mid,
                num_out=num_out,
                name="conv4_B%02d"%i,
                first_block=(i==1), stride=((2,2) if (i==1) else (1,1)))
        if i in [1]:
            conv4_x = GloRe_Unit(data=conv4_x,
                    settings=(num_out, num_mid),
                    name="conv4_B%02d_extra"%i)#, stride=(1,1))

            # conv5
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[5]+1):
        conv5_x = Residual_Unit(data=(conv4_x if i==1 else conv5_x),
                num_in=(num_in if i==1 else num_out),
                num_mid=num_mid,
                num_out=num_out,
                name="conv5_B%02d"%i,
                first_block=(i==1), stride=((2,2) if (i==1) else (1,1)))
        if i in []:
            conv5_x = GloRe_Unit(data=conv5_x,
                    settings=(num_out, int(num_out/2), int(num_out/2)),
                    name="conv5_B%02d_extra"%i)#, stride=(1,1))

            # output
    conv5_x = BN_AC(conv5_x, name='tail')
    return conv5_x


def get_linear(num_classes = 1000):
    before_pool = get_before_pool()
    # - - - - -
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1,1), name="global-pool")
    flat5     = mx.symbol.Flatten(data=pool5, name='flatten')
    fc6       = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='classifier')
    return fc6


def get_symbol(num_classes = 1000):
    fc6       = get_linear(num_classes)
    softmax   = mx.symbol.SoftmaxOutput( data=fc6,  name='softmax')
    sys_out   = softmax
    return sys_out

def get_symbol(num_classes, **kwargs):
    data = mx.symbol.Variable(name="data") # 224
    data = data-127.5
    data = data*0.0078125
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    print(version_input, version_output, version_unit)
    #if version_input==0:
    body=get_before_pool(data=data)

    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    return fc1


