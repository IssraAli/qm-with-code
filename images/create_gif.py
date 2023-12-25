import numpy as np
import imageio
import os



with imageio.get_writer('../../pib-stat.gif', mode = 'I', duration = 100) as writer:
    # photolist = os.listdir('./images/pib')
    # photolist.sort()
    for number in range(0, 214, 2):
        image = imageio.imread(str(number) + '.png')
        writer.append_data(image)
# images = []
# photolist = os.listdir('./images/pib')
# print(photolist)
# for filename in photolist:
#     images.append(img.imread(filename))
# img.mimsave('./pib-gauss.gif', images)
