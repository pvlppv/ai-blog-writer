import tensorflow as tf
import numpy as np
import os
import time
import datetime

model = tf.saved_model.load('blog_model')

states = None
next_char = tf.constant([input("Seed: ")])
result = [next_char]

for n in range(100):
  next_char, states = model.generate_one_step(next_char, states=states)
  result.append(next_char)

print(tf.strings.join(result)[0].numpy().decode("utf-8"))
