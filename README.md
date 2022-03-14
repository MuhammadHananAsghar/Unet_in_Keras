# Unet
![image](https://user-images.githubusercontent.com/44013285/158085915-a9bd3ff5-3613-44e1-a3db-24a45a293eec.png)

```python
input_image = tf.image.resize_with_pad(datapoint['image'], 572, 572)
input_mask = tf.image.resize_with_pad(datapoint['segmentation_mask'], 388, 388)

# This is based on our dataset. The output channels are 3, think of it as each pixel will be classified
# into three classes, but I have written 4 here, as I do padding with 0, so we end up have four classes.
outputs = layers.Conv2D(4, kernel_size=1)(c21)
```
