# -*- coding: utf-8 -*-
# 本檔案為正體中文教學文件，於部署環境使用請再三留意

""" 從 tfrecords 讀取資料，並透過 CNN 'LeNet5' 做訓練 """

###############################################################################
#             使用到的 TensorFlow 函式，若有不懂，請見 '末尾附註'
###############################################################################

import tensorflow as tf
import numpy as np

import LeNet5 as lenet

MIN_AFTER_DEQUEUE = 1000
NUM_THREADS       = 4
SAFE_MARGIN       = 3

DISPLAY_STEP      = 1
SAVE_WEIGHTS_STEP = 10

MODEL_KEEP_INPUT  = 0.8                          # Dropout 的維持比例

IMAGE_CHANNELS  = 1                              # RGB 圖像則會有三個灰階層
LEARNING_RATE   = 0.001

BATCH_SIZE      = 20
MAX_STEP        = 1000                           # 最大迭代次數


# 請確認底下參數與 LeNet5 相同　（ 預設相同 )
# -----------------------------------------------------------------------------

CONV_LAYER_1_WEIGHT_NAME = "wc1"
CONV_LAYER_1_BIAS_NAME   = "bc1" 
CONV_LAYER_1_DEEP        = 64                    # 第一層卷稽核總數量

# 請確認底下參數與 create_tfrecords 相同　（ 預設相同 )
# -----------------------------------------------------------------------------
IMAGE_SIZE          = 128                        # 圖像調整為正方形的矩陣大小

LABEL_NAME          = 'label'                    # 樣本中標籤的索引
IMAGE_NAME          = 'image'                    # 樣本中圖像的索引
LABEL_NUMBER        = 2                          # 標籤總數

SUMMARY_PATH        = 'logs/summary'             # Summary log 位址
WEIGHT_BIAS_PATH    = 'logs/weights'             # 輸出權重位址


# =============================================================================
#                                                                       define
# =============================================================================

# -----------------------------------------------------------------------------
# read_tfrecords(filename_queue)
#
#   - 從文件序列中讀取 TFRecords 資料, 於 input_popeline 中使用
#
# inputs  :
#   - filename_queue < tf FIFOQueue object > : TensorFlow 的檔案序列
#
# outputs :
#   - image < tf Tensor float > : 圖像的 RGB 矩陣表示
#   - label < tf Tensor float > : 標籤的浮點數表示, 已標準化到 [0, 1] 區間
#
# see also : create_tfrecords.py
#
# -----------------------------------------------------------------------------
def read_tfrecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature  = {LABEL_NAME : tf.FixedLenFeature([], tf.int64),
                IMAGE_NAME : tf.FixedLenFeature([], tf.string),
               }
    features = tf.parse_single_example(serialized_example,features=feature)
    image      = tf.decode_raw(features[IMAGE_NAME], tf.float32)
    image      = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    label    = tf.cast(features[LABEL_NAME], tf.int32)
    label    = tf.sparse_to_dense(label, [LABEL_NUMBER], 1, 0)
    return image, label


# -----------------------------------------------------------------------------
# input_pipeline(filenames, batch_size, num_epochs=None)
#
#   - 建立輸入資料的 pipeline ( 詳見末尾附註 )
#
# inputs  :
#   - filenames  < 1 x N string > : 訓練資料的檔案名稱列表
#   - batch_size < int >          : batch (批量) 的大小
#   - num_epochs < int >          : epoch (迭代) 的次數, 預設為 None
#
# outputs :
#   - image_batch < tf Tensor float > : 批量圖像的 RGB 矩陣表示
#   - label_batch < tf Tensor float > : 批量圖像的標籤值, 已標準化到 [0, 1] 區間
#
# -----------------------------------------------------------------------------

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
    image, label = read_tfrecords(filename_queue)
    capacity = MIN_AFTER_DEQUEUE + (NUM_THREADS + SAFE_MARGIN) * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size, capacity=capacity, num_threads = NUM_THREADS,
        min_after_dequeue=MIN_AFTER_DEQUEUE)
    return image_batch, label_batch

# =============================================================================
#                                                                       script
# =============================================================================

# 讀取 TFRecords 訓練集資料
# -----------------------------------------------------------------------------
img_batch, label_batch = input_pipeline(["train.tfrecords"], BATCH_SIZE)



# 輸入輸出設定 (輸入輸出要以 placeholder 建立)
# -----------------------------------------------------------------------------

# 輸入設定
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS],
                   name="CNN_INPUT_x")

# 輸出設定
y = tf.placeholder(tf.float32, [None, LABEL_NUMBER],
                   name="CNN_TARGET_y")

# Dropout比例設定
keepratio = tf.placeholder(tf.float32, name="CNN_DROPOUT_keepratio")




# 參數設定 ( 參數可以透過 sess.run 獲得 )
# -----------------------------------------------------------------------------

# 權重與偏權值設定
weights, biases = lenet.initialize(LABEL_NUMBER, IMAGE_CHANNELS)

# 預測值設定
pred = lenet.model(x, weights, biases, keepratio)['out']

# 損失函數設定
cost_method = 'cross entropy'
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# 優化器設定
optm = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# 精確度設定 (accuracy)
corr = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) # 準確率, 根據 label 會不同
accr = tf.reduce_mean(tf.cast(corr, tf.float32))
accr_name = 'accuracy'

# 初始化設定
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())



# 開始初始化
# -----------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(init)
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
# 設置 Summary
# -----------------------------------------------------------------------------
    tf.summary.scalar(cost_method, cost)
    tf.summary.scalar(accr_name  , accr)
    merged         = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(SUMMARY_PATH, graph=sess.graph)

# 訓練開始
# -----------------------------------------------------------------------------
    print ("Start!")
    step = 0
    while not coord.should_stop() and step < MAX_STEP:
        step += 1
        imgs, labels = sess.run([img_batch, label_batch])
        sess.run([merged, optm], feed_dict = {x: imgs, y: labels, keepratio: MODEL_KEEP_INPUT})
        
        if step % DISPLAY_STEP == 0:
            #输出当前batch的精度。预测时keep的取值均为 1
            acc = sess.run(accr, feed_dict = {x: imgs, y: labels, keepratio: 1.})
            print('%s accuracy is %.4f' % (step, acc))
            
        if step % SAVE_WEIGHTS_STEP == 0:
            W_val, b_val = sess.run([weights, biases])
            for weight_i in range(CONV_LAYER_1_DEEP):
                np.savetxt(WEIGHT_BIAS_PATH + "/Weight_"+str(weight_i+1)+".csv",
                           W_val.get(CONV_LAYER_1_WEIGHT_NAME)[:,:,0,weight_i],
                           delimiter=",")                
            np.savetxt(WEIGHT_BIAS_PATH + "/bias_ALL.csv",
                       b_val.get(CONV_LAYER_1_BIAS_NAME),
                       delimiter=",")

    # 停止所有執行緒
    coord.request_stop()

    # 等待所有執行緒順利停止
    coord.join(threads)

    # 等待結束，關閉 Session
    sess.close()

print ("Optimization Finished.")

# =============================================================================
#                                                                         note
# =============================================================================

#  input_pipeline
#     這是一個負責把資料做亂序, 並且將資料分批準備好的函數
#
#     min_after_dequeue (最小出列長度) 代表在亂序的時候, 會使用到的緩衝區有多少
#     數值越大代表亂序效果越好, 但是啟動時間也會更長
#     本 code 預設為 MIN_AFTER_DEQUEUE
#     可以把他想成廚房切菜砧板的大小, 切菜最多就是切砧板大小的量, 切完就要放到盆裡
#
#     capacity (緩衝容量) 要比 min_after_dequeue 大
#     多出來的空間就是可以提前先載入預備資料
#     可以想成廚房桌檯的大小，一定要比砧板大小更大, 多的空間可以先放待會要切的食材
#     建議值 :
#        capacity = min_after_dequeue + ( num_threads + safe_margin ) * batch_size
#     其中
#        num_threads = 線程數     , 本 code 預設為 NUM_THREADS
#        safe_margin = 安全係數   , 本 code 預設為 SAFE_MARGIN
