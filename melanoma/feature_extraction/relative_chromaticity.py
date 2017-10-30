from color_variation import *

def get_relative_chromaticity(img):
    r_lesion, g_lesion, b_lesion, r_skin, g_skin, b_skin = get_RGB(img)

    rel_r = ((np.mean(r_lesion)) / (np.mean(r_lesion) + np.mean(g_lesion) + np.mean(b_lesion))) \
    - (np.mean(r_skin) / (np.mean(r_skin) + np.mean(g_skin) + np.mean(b_skin)))

    rel_g = ((np.mean(g_lesion)) / (np.mean(r_lesion) + np.mean(g_lesion) + np.mean(b_lesion))) \
    - (np.mean(g_skin) / (np.mean(r_skin) + np.mean(g_skin) + np.mean(b_skin)))

    rel_b = ((np.mean(b_lesion)) / (np.mean(r_lesion) + np.mean(g_lesion) + np.mean(b_lesion))) \
    - (np.mean(b_skin) / (np.mean(r_skin) + np.mean(g_skin) + np.mean(b_skin)))

    return rel_r, rel_g, rel_b
