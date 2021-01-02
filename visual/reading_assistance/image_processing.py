from PIL import Image

image = Image.open('5.png')
image_size = image.size
print(image_size)
new_image = image.resize((int(image_size[0]-1000), int(image_size[1]-1000)))
new_image.save('test.png')
print(new_image.size)