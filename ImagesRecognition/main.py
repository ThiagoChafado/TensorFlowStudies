import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.datasets import mnist
#print(physical_devices = tf.config.list_physical_devices('GPU'))
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1,28*28).astype("float32") / 255.0
#Tensorflow automatically converts numpy arrays to a tensor

#Sequential API (Very convenient, not very flexible)

"""model = keras.Sequential([
    keras.Input(shape=(28*28)),
    layers.Dense(512,activation="relu"),
    layers.Dense(128,activation="relu"),
    layers.Dense(10),
])"""

"""#Or can add one layer at time
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(10))"""

#Functional API (A bit more flexible)
inputs = keras.Input(shape=(784,))
x = layers.Dense(512,activation="relu",name="first_layer")(inputs)
x = layers.Dense(256,activation="relu",name="second_layer")(x)
outputs = layers.Dense(10,activation="softmax")(x)
model = keras.Model(inputs = inputs,outputs = outputs)

print(model.summary())


model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),#For softmax not defined,set True
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train,y_train, batch_size=1000,epochs=20)
model.evaluate(x_test,y_test,batch_size=1)
