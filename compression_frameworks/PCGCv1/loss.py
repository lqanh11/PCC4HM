# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2021.9.6

#import tensorflow as tf
import torch
import numpy as np

def get_bce_loss(pred, label):
  """ (Weighted) Binary cross entropy loss.
  Input:GPU
      pred: [batch size, 1, vsize, vsize, vsize] float32
      label: must be 0 or 1, [batch size, 1, vsize, vsize, vsize] float32
  output: 
      empty loss & full loss
  """

  # occupancy = pred
  occupancy = torch.clamp(torch.sigmoid(pred), min=1e-7, max=1.0 - 1e-7)

  mask_neg = torch.lt(label,0.5) #[batch size, vsize, vsize, vsize, 1]
  mask_pos = torch.gt(label,0.5)
  occupancy_neg = torch.masked_select(occupancy,mask_neg)
  occupancy_pos = torch.masked_select(occupancy,mask_pos)
  empty_loss = torch.mean(-torch.log(1.0 - occupancy_neg))
  full_loss = torch.mean(-torch.log(occupancy_pos))

  return empty_loss, full_loss #1个值的tensor

def get_confusion_matrix(pred, label, th=0.):
  """confusion matrix: 
      1   0
    1 TP  FN
    0 FP  TN(option)
  input:
    pred, label: float32 CPU [batch size, 1, vsize, vsize, vsize]
  output: 
    TP(true position), FP(false position), FN(false negative);
    float32 [batch size, vsize, vsize, vsize];
  """
  pred = torch.squeeze(pred, 1)
  label = torch.squeeze(label, 1)

  pred = torch.gt(pred,th).float()
  label = torch.gt(label,th).float()

  TP = pred * label
  FP = pred * (1. - label)
  FN = (1. - pred) * label
  # TN = (1 - pred) * (1 - label)

  return TP, FP, FN

def get_classify_metrics(pred, label, th=0.):
  """Metrics for classification.
  input:
      pred, label; type : float32 tensor CPU;  shape: [batch size, 1, vsize, vsize, vsize]
  output:
      precision rate; recall rate; IoU;
  """

  TP, FP, FN = get_confusion_matrix(pred, label, th=th)
  TP = torch.sum(TP)
  FP = torch.sum(FP)
  FN = torch.sum(FN)

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  IoU = TP / (TP + FP + FN)

  return precision, recall, IoU 


if __name__=='__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  np.random.seed(108)
  data = np.random.rand(2, 64, 64, 64, 1)* 10 - 5
  data = data.astype("float32")
  label = np.random.rand(2, 64, 64, 64, 1)
  label[label>=0.97] = 1
  label[label<0.97] = 0
  label = label.astype("float32")

  data = torch.from_numpy(data).to(device)
  label = torch.from_numpy(label).to(device)
  
  loss1, loss2 = get_bce_loss(data, label)
  print("loss1:",loss1)
  print("loss2:",loss2)
  '''np.random.seed(108)
  data = np.random.rand(2, 64, 64, 64, 1)* 10 - 5
  data = data.astype("float32")
  label = np.random.rand(2, 64, 64, 64, 1)
  label[label>=0.97] = 1
  label[label<0.97] = 0
  label = label.astype("float32")

  data = tf.Variable(data)
  label = tf.constant(label)

  if tf.executing_eagerly():
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
    for i in range(1000):
      with tf.GradientTape() as tape:
        loss1, loss2 = get_bce_loss(data, label)
        loss = loss1 + 3*loss2

        gradients = tape.gradient(loss, data)
        optimizer.apply_gradients([(gradients, data)])

        if i%100==0:
          print(i,loss.numpy())
          precision, recall, IoU = get_classify_metrics(data, label, 0.)
          print(precision.numpy(), recall.numpy(), IoU.numpy())

  if not tf.executing_eagerly():
    with tf.Session('') as sess:
      loss1, loss2 = get_bce_loss(data, label)
      loss = loss1 + 3*loss2
      train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
      sess.run(tf.global_variables_initializer())

      for i in range(1000):
        trainloss, _ = sess.run([loss,train])
        if i%100==0:
          print(i,trainloss)

          precision, recall, IoU = get_classify_metrics(data, label, 0.)
          print(sess.run([precision, recall, IoU]))
  '''
