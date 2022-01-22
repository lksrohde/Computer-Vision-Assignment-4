import numpy as np


def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """
    cost_ssd = 0

    m = patch1.shape[0]
    for x in range(m):
        for y in range(m):
            cost_ssd += (patch1[x][y] - patch2[x][y]) ** 2
    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """

    m = patch1.shape[0]
    wl = np.reshape(patch1, (m**2, 1))
    wr = np.reshape(patch2, (m**2, 1))
    wl_mean = np.full((m ** 2, 1),np.mean(wl))
    wr_mean = np.full((m ** 2, 1), np.mean(wr))

    left_sub = wl - wl_mean
    right_sub = wr - wr_mean
    numerator = np.transpose(left_sub) @ (right_sub)
    denominator = np.linalg.norm(left_sub) * np.linalg.norm(right_sub)
    cost_nc = (numerator/denominator)[0][0]

    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape 

    m = patch1.shape[0]
    cost_val = (cost_ssd(patch1, patch2) / (m**2)) + alpha * cost_nc(patch1, patch2)

    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image
    
    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'
        
    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """

    assert np.isscalar(window_size)
    assert window_size % 2 == 1
    return np.pad(input_img, window_size, padding_mode)


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2 
    assert padded_img_r.ndim == 2 
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    disparity = np.zeros(padded_img_l.shape)
    H = padded_img_l.shape[0]
    W = padded_img_l.shape[1]

    for x in range(W):
        window_width = x
        if x < (window_size / 2) + 1:
            window_width = 0
        elif x > W - (window_size / 2) - 1:
            window_width = W - window_size
        for y in range(H):
            window_height = y
            if y < (window_size / 2) + 1:
                window_height = 0
            elif x > H - (window_size / 2) - 1:
                window_height = H - window_size
            best_disp = 0
            cost = np.iinfo(np.int).max
            used_disp = min(max_disp, window_width + 1)
            for i in range(used_disp):
                patch1 = padded_img_l[window_width:window_width + window_size,
                         window_height: window_height + window_size]
                patch2 = padded_img_r[window_width - i: window_width + window_size - i,
                         window_height: window_height + window_size]
                curr_cost = cost_function(patch1, patch2, alpha)
                if (curr_cost < cost):
                    best_disp = i
                    cost = curr_cost
            disparity[y][x] = best_disp

    assert disparity.ndim == 2
    return disparity

def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:
    
    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape

    N = disparity_gt.shape[0] * disparity_gt.shape[1]

    d = disparity_gt - disparity_res
    aepe = d.sum() / N

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #
    alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])
    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer 
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
        
        Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [?] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [?].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """

        return (1, -1, -1, -1)
