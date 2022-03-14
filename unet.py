import tensorflow as tf


# Contraction
input = tf.keras.layers.Input(shape=(572, 572, 3))

c0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(input)
c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c0)
c2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c1)

c3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c2)
c4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c3)
c5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c4)

c6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c5)
c7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c6)
c8 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c7)

c9 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c8)
c10 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c9)
c11 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c10)

c12 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c11)
c13 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c12)

# Expansion

c14 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), activation="relu")(c13)
# c10 from 64x64 to 58x58
crop1 = tf.keras.layers.Cropping2D(cropping=(4, 4))(c10)
concat1 = tf.keras.layers.concatenate([c14, crop1])
c15 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(concat1)
c16 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c15)

c17 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), activation="relu")(c16)
# c7 from 136x136 to 104x104
crop2 = tf.keras.layers.Cropping2D(cropping=(16, 16))(c7)
concat2 = tf.keras.layers.concatenate([c17, crop2])
c18 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(concat2)
c19 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c18)

c20 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), activation="relu")(c19)
# c4 from 280x280 to 200x200
crop3 = tf.keras.layers.Cropping2D(cropping=(40, 40))(c4)
concat3 = tf.keras.layers.concatenate([c20, crop3])
c21 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(concat3)
c22 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c21)

c23 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), activation="relu")(c22)
# c10 from 568x568 to 392x392
crop4 = tf.keras.layers.Cropping2D(cropping=(88, 88))(c1)
concat4 = tf.keras.layers.concatenate([c23, crop4])
c24 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(concat4)
c25 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="valid", strides=1, activation="relu")(c24)

output = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), padding="valid", strides=1)(c25)
unet = tf.keras.models.Model(inputs=[input], outputs=[output], name="unet")
unet.summary()
