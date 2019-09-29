import math
import sys
import os
import mxnet as mx
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
# def sigmoid(x):
#     return 1./(1. + mx.symbol.exp(-x))
def Sigmoid(data, name=None):
    x = mx.symbol.abs(data, name='%s_abs'%name)
    x = -(x/45.0 - 1)*10
    y = mx.symbol.exp(x, name='%s_exp'%name)
    y = 1.0/(1.0+y)
    return y

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' %(name, suffix))
    return act
    
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)    
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv    

    
def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj
    
def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
    	shortcut=identity
    	conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
    	identity=conv+shortcut
    return identity
        
def SResidual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    shortcut=Linear(data=identity, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=stride, name='%s%s_conv_proj_s' %(name, suffix))
    conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name=name, suffix='')
    identity=mx.symbol.ElementWiseSum(*[conv, shortcut], name=('%s_sum' % name)) 
    return identity

def get_symbol():
    num_classes = config.emb_size
    print('in_network', config)
    fc_type = config.net_output
    print('start get_symbol_v6')
    data = mx.symbol.Variable(name="data")
    yaw = mx.symbol.Variable(name="yaw")
    data = data-127.5
    data = data*0.0078125
    conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
    #conv_2 = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw")
    conv_2 = Residual(conv_1, num_block=2, num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128, name="res_2")
    conv_23 = SResidual(conv_2, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_23")
    conv_3 = Residual(conv_23, num_block=4, num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256, name="res_3")
    conv_34 = SResidual(conv_3, num_out=256, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_34")
    conv_4 = Residual(conv_34, num_block=6, num_out=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=512, name="res_4")
    conv_45 = SResidual(conv_4, num_out=256, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1024, name="dconv_45")
    conv_5 = Residual(conv_45, num_block=3, num_out=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=512, name="res_5")
    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")

    fc1 = symbol_utils.get_fc1(conv_6_sep, num_classes, fc_type)
    if config.dream:
        print("use dream")
        bn_mom = config.bn_mom
        body = fc1
        fc_dr1 = mx.sym.FullyConnected(data=body, num_hidden = num_classes, name = "fc_dr1")
        fc_dr1 = mx.sym.BatchNorm(data=fc_dr1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name="bn_fc_dr1")
        fc_dr2 = mx.sym.FullyConnected(data=fc_dr1, num_hidden = num_classes, name = "fc_dr2")
        fc_dr2 = mx.sym.BatchNorm(data=fc_dr2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name="bn_fc_dr2")
        # coef_yaw = sigmoid(10.0*(mx.symbol.abs(yaw)/45.0-1))
        coef_yaw = Sigmoid(yaw, name="sigmoid_yaw")
        fc_dream = body + mx.sym.broadcast_mul(coef_yaw, fc_dr2, name="fc_dream")

        return fc_dream
    else:
        return fc1
