import numpy as np

def convolution(oldimage, kernel):
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if (len(oldimage.shape) == 3):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2,kernel_w // 2), (0, 0)), mode = 'constant',constant_values = 0).astype(np.float32)
    elif (len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=(
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2,kernel_w // 2)), mode = 'constant', constant_values = 0).astype(np.float32)
    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape)

    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w

    if (h == 0):
        return image_conv[h:, w:w_end]
    if (w == 0):
        return image_conv[h:h_end, w:]
    return image_conv[h:h_end, w:w_end]