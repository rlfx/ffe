# -*- coding: utf-8 -*-
# 本檔案為正體中文教學文件，於部署環境使用請再三留意

""" 從目前路徑 data 資料夾讀取圖像，轉換成 tfrecord 以便在 tensorflow 中使用 """

###############################################################################
#             使用到的 TensorFlow 函式，若有不懂，請見 '末尾附註'
###############################################################################

import sys

import numpy as np
import tensorflow as tf
import glob
import cv2

from random import shuffle

# parameters
# -----------------------------------------------------------------------------

IMAGE_SIZE          = 128                   # 圖像調整為正方形的矩陣大小
ZERO_LABEL          = 'down'                # 標籤為 0 的圖像檔案名稱

LABEL_NAME          = 'label'               # 樣本中標籤的索引
IMAGE_NAME          = 'image'               # 樣本中圖像的索引

TRAIN_DATA_RATIO    = 0.9                   # 訓練集資料比例
VAL_DATA_RATIO      = 0.0                   # 驗證集資料比例
TEST_DATA_RATIO     = 0.1                   # 測試集資料比例

SHUFFLE_DATA        = False                 # 是否要隨機打亂資料順序
DISPLAY_STEP        = 1                     # 每處理多少圖像顯示訊息

DATA_PATH        = 'data/bitcoin/*.jpg'     # 所有資料路徑

TRAIN_FILENAME   = 'train.tfrecords'        # 訓練集資料 TFRecord 名稱
VAL_FILENAME     = 'val.tfrecords'          # 驗證集資料 TFRecord 名稱
TEST_FILENAME    = 'test.tfrecords'         # 測試集資料 TFRecord 名稱

# =============================================================================
#                                                                       define
# =============================================================================

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
def load_image(addr, sq_size=IMAGE_SIZE):
    img = cv2.imread(addr)
    img = cv2.resize(img, (sq_size, sq_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    return img



# -----------------------------------------------------------------------------
# _int64_feature(value)
#
#   - 將給定的 label 存成 TensorFlow 的 Ｆｅａｔｕｒｅ 格式
#
# inputs  :
#   - value < 1 x N > : label 值, 例如 [0 0 0 1 1 1 0 0 0 1 0]
#
# outputs :
#   - feature < tf object > : TensorFlow int64 格式的資料
#                             相當於 C++ 中的 long long
#
# -----------------------------------------------------------------------------
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# -----------------------------------------------------------------------------
# _bytes_feature(value)
#
#   - 將給定的　ｂｙｔｅｓｔｒｅａｍ　存成 tensorFlow 的 Ｆｅａｔｕｒｅ 格式
#
# inputs  :
#   - value < bytestream > :  二進制的表示
#
# outputs :
#   - feature < tf object > : TensorFlow bytestream 格式的資料
#
# -----------------------------------------------------------------------------
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# -----------------------------------------------------------------------------
# create_tfrecord(file_name)
#
#   - 建立給定資料集的 TFRecords 檔案
#
# inputs  :
#   - tfrecord_filename < string >       : 二進制的表示
#   - data_addrs        < 1 x N string > : N 個圖像位址
#   - data_labels       < 1 x N int >    : N 個圖像的對應標籤值
#   - dispstep          < int >          : 每處理多少圖像需要顯示進度
#
# outputs :
#   - none
#
# -----------------------------------------------------------------------------
def create_tfrecord(tfrecord_filename, data_addrs, data_labels, dispstep=1):
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    for i in range(len(data_addrs)):

        # 每 dispstep 告知目前儲存進 TFRecords 的圖像資料有多少
        if not i % dispstep:
            print('write to {} : {}/{}'.format( tfrecord_filename, i+1, len(data_addrs)))
            sys.stdout.flush()

        # 讀取圖像
        image = load_image(addrs[i])
        label = data_labels[i]

        # 建立樣本資料 'example'
        feature  = {LABEL_NAME: _int64_feature(label),
                    IMAGE_NAME: _bytes_feature(tf.compat.as_bytes(image. tobytes()))}
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)

        # 序列寫入資料集 example 到 TFRecord 檔案中
        writer.write(example.SerializeToString())

    # 關閉檔案
    writer.close()
    sys.stdout.flush()



# =============================================================================
#                                                                       script
# =============================================================================


# 從檔名是否包含 'dwon' 與否, 來將資料打上 0 或 1 的標籤
# -----------------------------------------------------------------------------

# 讀取訓練資料路徑位址
addrs  = glob.glob(DATA_PATH)


# 0 : 下一個 bar close 下跌
# 1 : 下一個 bar close 上漲
labels = [0 if ZERO_LABEL in addr else 1 for addr in addrs]



# 打亂資料順序 ( 由於我們是時間序列資料, 所以切訓練與測試還是先不要打亂 )
# -----------------------------------------------------------------------------
if SHUFFLE_DATA:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)



# 將資料根據比例分為訓練集, 驗證集與測試集
# -----------------------------------------------------------------------------

train_addrs  = addrs[0:int(TRAIN_DATA_RATIO*len(addrs))]
train_labels = labels[0:int(TRAIN_DATA_RATIO*len(labels))]
create_tfrecord(TRAIN_FILENAME, train_addrs, train_labels, dispstep=DISPLAY_STEP)

test_start_ratio = TRAIN_DATA_RATIO + VAL_DATA_RATIO       # 測試集開始比例位置
test_end_ratio   = test_start_ratio + TEST_DATA_RATIO      # 測試集結束比例位置

val_addrs    = addrs[int(TRAIN_DATA_RATIO*len(addrs)):int(test_start_ratio*len(addrs))]
val_labels   = labels[int(TRAIN_DATA_RATIO*len(addrs)):int(test_start_ratio*len(addrs))]
create_tfrecord(VAL_FILENAME, val_addrs, val_labels, dispstep=DISPLAY_STEP)

test_addrs   = addrs[int(test_start_ratio*len(addrs)):int(test_end_ratio*len(addrs))]
test_labels  = labels[int(test_start_ratio*len(labels)):int(test_end_ratio*len(addrs))]
create_tfrecord(TEST_FILENAME, test_addrs, test_labels, dispstep=DISPLAY_STEP)



print('TFRecords created suceefully.')
# =============================================================================
#                                                                         note
# =============================================================================

#   tf.python_io.TFRecordWriter
#     Tensorflow 提供的高效率的寫入檔案的函數, 透過 TFRecordWriter 建立 writer
#     來將資料寫進檔案, 主要是應對大量資料讀寫時會浪費記憶體而設計的, 另外也是針對
#     TFRecord 類型資料提供寫入檔案的方法, 對應的還有讀取格式

#   tf.compat.as_bytes
#     將 String 轉換成 bytestream 同時使用 UTF-8 做編碼, 使用這個函數前需要先用
#      tobytes 將圖片的 RGB 矩陣透過 numpy 轉換成 bytes

#   tf.train.Feature
#     這邊的特徵(feature) 指的是已經用指定格式表示的字典結構資料,
#     這些字典結構之後會幫助我們在 TensorFlow 執行時去對應到相對的要求輸入
#     支援整數, 福點數和二進制, 在這邊我們使用的是二進制的方式構建 feature

#  tf.train.Example
#     這邊是將特徵 (feature) 包裝成樣本 (example) 多包含了 protocol buffer
#     簡單來說就是對會使用到的資料, 建立對應的接口函數與說明資料的內容格式
#     主要目的是提供要使用 TFRecord 檔案的時候, 能更快速的進入資料並使用內容
#     protocal buffer 基本上是一種比起 JSON, XML 更好的格式, 也受到編碼保護

#  SerializeToString
#     將 bytestream 的資料寫入檔案中, 成為被編碼過後的 TFRecord 的資料
#     隨意打開可能會是一堆類似 3330 0900 0000 0000 30cb 2a4a 這種東西
