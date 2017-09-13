# -*- coding: utf-8 -*-
# 本檔案為正體中文教學文件，於部署環境使用請再三留意

""" 簡單定義了 LeNet-5 的 TensorFlow 模型 """

###############################################################################
#             使用到的 TensorFlow 函式，若有不懂，請見 '末尾附註'
###############################################################################

import tensorflow as tf


# 輸入資料匹配
IMAGE_SIZE         = 128      # 圖像矩陣大小
IMAGE_CHANNELS     = 3        # RGB 圖像具有三個灰階層

# 卷稽核大小
CONV_LAYER_1_SIZE = 3         # 第一層卷稽核大小
CONV_LAYER_2_SIZE = 3         # 第二層卷稽核大小

# 各層深度
CONV_LAYER_1_DEEP = 64        # 第一層卷稽核總數量
CONV_LAYER_2_DEEP = 128       # 第二層卷稽核總數量
FC_LAYER_SIZE     = 1024      # 全連接層的大小



# 不太會需要更動的參數 & 解釋變數
# -----------------------------------------------------------------------------

# 卷積/池化步長與填充方法 見末尾附註
CONV_MOVE_SIZE = 1            # 卷積移動步長
POOL_SIZE      = 2            # 池化大小, 也是池化移動步長
POOL_LAYER_NUMBER = 2         # 總共有多少個池化層, 為了算全連結層大小
PADDING_METHOD = 'SAME'       # 卷積後填充方法

# 初始化
INIT_NORMAL_STDDEV = 0.1      # 常態分布的標準差, 用以初始化權重

# 為了初始化全連接層權重, 必須得到全連接層輸入大小
# -----------------------------------------------------------------------------
# 由於我們使用 'SAME' 的填充方法, 所以卷積層不會讓維度變小
# 只有池化層會變小, 根據池化層的大小和次數, 就可以算出最終池化後的輸出大小
# 最終層的輸出大小, 也就是全連接層的輸入大小
BEFORE_FC_LAYER_SIZE = int(IMAGE_SIZE / pow(POOL_SIZE, POOL_LAYER_NUMBER))
# 再乘上最後一層的深度(第二層卷積層的總數),就是進全連接層之前的總餐數量 (nodes)
FC_LAYER_NODES       = CONV_LAYER_2_DEEP * pow(BEFORE_FC_LAYER_SIZE , 2)

# 比較無關緊要的  variable scope 名稱, 這在 RNN 會比較重要
# -----------------------------------------------------------------------------
INPUT_LAYER_NAME         = "input_layer"

CONV_LAYER_1_NAME        = "CNN_CONV_1"
CONV_LAYER_1_WEIGHT_NAME = "wc1"
CONV_LAYER_1_BIAS_NAME   = "bc1"
POOL_LAYER_1_NAME        = "CNN_POOL_1"

CONV_LAYER_2_NAME        = "CNN_CONV_2"
CONV_LAYER_2_WEIGHT_NAME = "wc2"
CONV_LAYER_2_BIAS_NAME   = "bc2"
POOL_LAYER_2_NAME        = "CNN_POOL_2"

FC_LAYER_NAME            = "FC_1"
FC_LAYER_WEIGHT_NAME     = "wd1"
FC_LAYER_BIAS_NAME       = "bd1"

OUTPUT_LAYER_NAME        = "FC_2"
OUTPUT_LAYER_WEIGHT_NAME = "wd2"
OUTPUT_LAYER_BIAS_NAME   = "bd2"

INIT_WEIGHT_NAME         = "CNN_WEIGHTS"
INIT_BIAS_NAME           = "CNN_WEIGHTS"

# =============================================================================
#                                                                       define
# =============================================================================



# -----------------------------------------------------------------------------
#  model(_input, _w, _b, _keepratio)
#
#   - LeNet-5 的 TensorFlow Model 定義
#
# inputs  :
#   - _input < tf placeholder  > : 輸入圖像的 placeholder
#                           大小為 [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_ChANNEL]
#   - _w < tf placeholder >      : 網路權重的 placeholder
#                           大小為 [BATCH_SIZE, LABEL_NUMBER]
#   - _b < tf placeholder >      : 網路偏權的 placeholder
#                           大小為 [BATCH_SIZE, LABEL_NUMBER]
#
# outputs :
#   - img < np.float32 > : 圖像的 RGB 矩陣數值格式
#
# -----------------------------------------------------------------------------


def model(_input, _w, _b, _keepratio):

# 輸入層
# -----------------------------------------------------------------------------
    with tf.variable_scope(INPUT_LAYER_NAME):
        _input_r = tf.reshape(_input, shape = [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

# 第一層卷積層
# -----------------------------------------------------------------------------

    with tf.variable_scope(CONV_LAYER_1_NAME):
        _conv1 = tf.nn.relu(tf.nn.bias_add(
                 tf.nn.conv2d(_input_r,
                              _w[CONV_LAYER_1_WEIGHT_NAME],
                              strides = [1, CONV_MOVE_SIZE, CONV_MOVE_SIZE, 1],
                              padding = PADDING_METHOD),
                              _b[CONV_LAYER_1_BIAS_NAME]))

# 第一層池化層
# -----------------------------------------------------------------------------
    with tf.variable_scope(POOL_LAYER_1_NAME):
        _pool1 = tf.nn.max_pool(_conv1,
                ksize   = [1, POOL_SIZE, POOL_SIZE, 1],
                strides = [1, POOL_SIZE, POOL_SIZE, 1],
                padding = PADDING_METHOD)
        _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)


# 第二層卷積層
# -----------------------------------------------------------------------------
    with tf.variable_scope(CONV_LAYER_2_NAME):
        _conv2 = tf.nn.relu(tf.nn.bias_add(
                 tf.nn.conv2d(_pool_dr1,
                             _w[CONV_LAYER_2_WEIGHT_NAME],
                             strides=[1, CONV_MOVE_SIZE, CONV_MOVE_SIZE, 1],
                             padding=PADDING_METHOD),
                             _b[CONV_LAYER_2_BIAS_NAME]))

# 第二層池化層
# -----------------------------------------------------------------------------
    with tf.variable_scope(POOL_LAYER_2_NAME):
        _pool2 = tf.nn.max_pool(_conv2,
                 ksize   = [1, POOL_SIZE, POOL_SIZE, 1],
                 strides = [1, POOL_SIZE, POOL_SIZE, 1],
                 padding=PADDING_METHOD)
        _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)


# 全連接層
# -----------------------------------------------------------------------------
    with tf.variable_scope(FC_LAYER_NAME):
        # 向量化
        _dense1 = tf.reshape(_pool_dr2,
                     [-1, _w[FC_LAYER_WEIGHT_NAME].get_shape().as_list()[0]])

        _fc1 = tf.nn.relu(tf.nn.bias_add(
                tf.matmul(_dense1,
                          _w[FC_LAYER_WEIGHT_NAME]),
                          _b[FC_LAYER_BIAS_NAME]))
        _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)


# 輸出層
# -----------------------------------------------------------------------------
    with tf.variable_scope(OUTPUT_LAYER_NAME):
        # Fc2
        _out = tf.add(tf.matmul(_fc_dr1,
                    _w[OUTPUT_LAYER_WEIGHT_NAME]),
                    _b[OUTPUT_LAYER_BIAS_NAME])


# 整體模型
# -----------------------------------------------------------------------------
    with tf.variable_scope(FC_LAYER_NAME): #重複 NAME 只是懶得再取 NAME
         out = {
                 'input_r': _input_r,
                 'conv1'  : _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
                 'conv2'  : _conv2, 'pool2': _pool2, 'pool_dr2' : _pool_dr2,
                 'dense1' : _dense1, 'fc1' : _fc1  , 'fc_dr1'   : _fc_dr1,
                 'out'    : _out }

    return out



# -----------------------------------------------------------------------------
# load_image(addr, sq_size=224)
#
#   - 從給定的資料夾路徑讀取圖像，調整大小為正方形 , 並轉換成 RGB 格式
#
# inputs  :
#   - addr    < string > : 圖像檔案路徑 　　　　　　　p.s. 建議使用 glob 得到路徑
#   - sq_size < int >    : 圖片校正至正方形大小 　p.s. 預設調整至 224 x 224
#
# outputs :
#   - img < np.float32 > : 圖像的 RGB 矩陣數值格式
#
# -----------------------------------------------------------------------------

def initialize(_n_output, _image_channels=IMAGE_CHANNELS):

# 初始化 weight (權重值)
# -----------------------------------------------------------------------------
    with tf.variable_scope(INIT_WEIGHT_NAME):
        weights  = {
                CONV_LAYER_1_WEIGHT_NAME: tf.Variable(
                        tf.random_normal([CONV_LAYER_1_SIZE,
                                          CONV_LAYER_1_SIZE,
                                          1,
                                          CONV_LAYER_1_DEEP],
                                         stddev = INIT_NORMAL_STDDEV )),
                CONV_LAYER_2_WEIGHT_NAME: tf.Variable(
                        tf.random_normal([CONV_LAYER_2_SIZE,
                                          CONV_LAYER_2_SIZE,
                                          CONV_LAYER_1_DEEP,
                                          CONV_LAYER_2_DEEP],
                                         stddev = INIT_NORMAL_STDDEV )),
                FC_LAYER_WEIGHT_NAME: tf.Variable(
                        tf.random_normal([FC_LAYER_NODES,
                                          FC_LAYER_SIZE],
                                         stddev = INIT_NORMAL_STDDEV )),
                OUTPUT_LAYER_WEIGHT_NAME: tf.Variable(
                        tf.random_normal([FC_LAYER_SIZE, _n_output],
                                         stddev = INIT_NORMAL_STDDEV ))
                }

# 初始化 bias (偏權值)
# -----------------------------------------------------------------------------
    with tf.variable_scope(INIT_BIAS_NAME):
        biases   = {
                CONV_LAYER_1_BIAS_NAME: tf.Variable(
                        tf.random_normal([CONV_LAYER_1_DEEP],
                                         stddev = INIT_NORMAL_STDDEV)),
                CONV_LAYER_2_BIAS_NAME: tf.Variable(
                        tf.random_normal([CONV_LAYER_2_DEEP],
                                         stddev = INIT_NORMAL_STDDEV)),
                FC_LAYER_BIAS_NAME: tf.Variable(
                        tf.random_normal([FC_LAYER_SIZE],
                                         stddev = INIT_NORMAL_STDDEV)),
                OUTPUT_LAYER_BIAS_NAME: tf.Variable(
                        tf.random_normal([_n_output],
                                         stddev = INIT_NORMAL_STDDEV))
                }

    return weights, biases

# =============================================================================
#                                                                       notes
# =============================================================================

#   strides : 卷積/池化移動的步長, 是對應 input 的形狀
# -----------------------------------------------------------------------------
#             我們知道 input 形狀是 [ 批量, x, y, 通道 ] ( RGB通道為 3)
#             對應的移動步長應該是  [   A,   B, C,  D  ]
#                         A     : 每次批量資料的移動幅度
#                         B, C  : 卷積/池化在 x, y 軸上的移動幅度
#                         D     : 每次通道的移動幅度
#             由於每次卷積/池化都只在一個資料、一個通道上做, 所以 A, D 為 1
#             B, C 則是我們可以設定的：
#                - 卷積層移動幅度預設為 CONV_MOVE_SIZE
#                - 池化層移動幅度預設為 POOL_SIZE


#   padding : 卷積/池化結束後的填充方法
# -----------------------------------------------------------------------------
#             padding 可分為 'VALID'(有效) 和 'SAME'(相同)
#             如果是 VALID , 卷積的輸出就會正常地縮小
#             如果是 SAME,   卷積的輸出就會和輸入一樣大小, 但是輸出會置中
#             然後在為了維持和輸入相同的大小下多出來的地方, 補上 0
#             預設為 PADDING_METHOD
#
#             請留意這會影響全連接層輸入的大小, 見 BEFORE_FC_LAYER_SIZE
