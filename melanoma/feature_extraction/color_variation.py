import numpy as np

def get_RGB(img):
    r_lesion = []
    g_lesion = []
    b_lesion = []

    r_skin = []
    g_skin = []
    b_skin = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.array_equal(img[i][j], np.asarray([255,255,255])):
                r_lesion.append(img[i][j][0])
                g_lesion.append(img[i][j][1])
                b_lesion.append(img[i][j][2])
            else:
                r_skin.append(img[i][j][0])
                g_skin.append(img[i][j][1])
                b_skin.append(img[i][j][2])

    return r_lesion, g_lesion, b_lesion, r_skin, g_skin, b_skin
