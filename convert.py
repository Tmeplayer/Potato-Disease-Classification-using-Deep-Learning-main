import tensorflow as tf

model = tf.keras.models.load_model(r'D:\Study\programing\Ai\a\Potato-Disease-Classification-using-Deep-Learning-main\model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model = converter.convert()

with open('model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)



interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


