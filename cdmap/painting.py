"""
This module is largely responsible for impainting the distance map,
a.k.a. removing possible undefined values inside the image.
"""

import scipy.sparse
import numpy

class NanInpainter():
    """
    This module is a Python-port of the inpaint_nans in MATLAB from
    John d'Errico:
    https://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint_nans
    We only ported the "mean" method based on Laplace's equation.
    """
    def __init__(self,method="default"):
        self.method = method

    def __call__(self,distance_map):
        if self.method is None:
            return distance_map
        if self.method == "default":
            distance_map = NanInpainter.inpaint_nans(img=distance_map)
            return distance_map
        raise AssertionError("Incorrect inpainting method defined, abort calling")

    @staticmethod
    def inpaint_nans(img):
        """
        Input image and return inpainted image.
        """
        assert isinstance(img, numpy.ndarray), "Expected ndarray, got " + str(type(img))
        assert len(img.shape) == 2, "Expected 2D image, got " + str(img.shape)
        dim_x = img.shape[0]
        dim_y = img.shape[1]
        dist_points = img.flatten()
        nan_list = [ind for ind in range(len(dist_points)) if numpy.isnan(dist_points[ind])]
        known_list = [ind for ind in range(len(dist_points)) if not numpy.isnan(dist_points[ind])]
        talks_to = numpy.array([[-1,0],[0,-1],[1,0],[0,1]])
        neighbors_list = NanInpainter._identify_neighbors(nan_list,dim_x,dim_y,talks_to)
        if len(nan_list) > 0:
            B = NanInpainter._solve_partials_for_nans(dist_points,
                                                      neighbors_list,
                                                      nan_list,
                                                      known_list,
                                                      dim_x,
                                                      dim_y)
            img_inpainted = numpy.reshape(B,(dim_x,dim_y))
            return img_inpainted
        return img

    @staticmethod
    # Inpainting
    def _identify_neighbors(nan_list,dim_x,dim_y,talks_to):
        """
        Identify the neighbors for the pixels
        ----
        Returns a list.
        """
        # get neighbors
        len_pre_nn = len(talks_to) * len(nan_list)
        neighbors_list = numpy.full(len_pre_nn,numpy.nan)
        ind = 0
        if nan_list != []:
            for cur_nan in nan_list:
                pos_x = int(cur_nan / dim_y)
                pos_y = cur_nan % dim_x
                cur_pos = [pos_x,pos_y]
                for cur_talk in talks_to:
                    neighbor_candidate = cur_pos + cur_talk
                    # Only neighbors within the image boundaries are allowed:
                    if (not any(numpy.array(neighbor_candidate) < 0) and
                        not any(neighbor_candidate >= [dim_x, dim_y])):
                        neighbor_id = neighbor_candidate[1] * dim_x + neighbor_candidate[0]
                        neighbors_list[ind] = neighbor_id
                        ind = ind + 1
        else:
            neighbors_list = []
        neighbors_list = [int(x) for x in neighbors_list if str(x) != 'nan']
        return neighbors_list

    @staticmethod
    def _solve_partials_for_nans(A,neighbors_list,nan_list,known_list,dim_x,dim_y):
        """
        Solves Laplace's equation and expects the list of relevant variables.
        ----
        Returns the missing pixels.
        """
        all_list = neighbors_list + nan_list
        # first consider rows
        L = [x for x in all_list if (x % dim_x != 0) and (x % dim_x != dim_x - 1)]
        cur_x = numpy.repeat(L,3)
        cur_y = numpy.repeat(L,3) + numpy.tile([-1,0,1],len(L))
        cur_v = numpy.tile([1,-2,1],len(L))
        values_row = scipy.sparse.csc_matrix((cur_v,(cur_x,cur_y)),shape=(dim_x**2,dim_y**2))
        # now consider columns
        L = [x for x in all_list if (int(x / dim_x) != 0) and (int(x / dim_x) != dim_x - 1)]
        cur_x = numpy.repeat(L,3)
        cur_y = numpy.repeat(L,3) + numpy.tile([-dim_x,0,dim_x],len(L))
        cur_v = numpy.tile([1,-2,1],len(L))
        values_col = scipy.sparse.csc_matrix((cur_v,(cur_x,cur_y)),shape=(dim_x**2,dim_y**2))
        # solve partial differential equation
        fda = values_row + values_col
        rhs = -fda[:,known_list] * A[known_list]
        k = [ind for x,ind in zip(fda[:,nan_list],range(fda[:,nan_list].shape[0]))
             if numpy.sum(x) != 0]
        B = A
        not_found = fda[:,nan_list].toarray()[k]
        out = numpy.linalg.lstsq(not_found,rhs[k],rcond=None)
        B[nan_list] = out[0]
        return B

class ImageFromDistancesCreator():
    """
    Take raw distance map as input and create image from inpaint_nans Scaling
    can be <linear> (default), <minmax>, <perpixel>.  <linear> and <perpixel>
    require the full distance map array to compute the distances from the
    beginning
    """
    def __init__(self,
            scaling_method="linear",
            inpainter=NanInpainter(),
            correct_image_outliers=True):
        self.scaling_method = scaling_method
        self.inpainter = inpainter
        self.correct_image_outliers = correct_image_outliers
        self._target_min = 0
        self._target_max = 1

    def __call__(self,dist_map_list=None):
        """
        Return the images from the distances in a grid. As <per_pixel> and <linear> require
        multiple samples, we require the full dataset at once.
        ----
        Return a list of the distance maps.
        """
        if self.inpainter is not None:
            dist_map_list = [self.inpainter(img) for img in dist_map_list]
        dist_map_list = self._rescale_image_array(dist_map_list,scaling_method=self.scaling_method)
        if self.correct_image_outliers:
            dist_map_list = [
                    ImageFromDistancesCreator.set_image_values_outside_boundary(
                        dist_map,
                        target_min=self._target_min,
                        target_max=self._target_max) for dist_map in dist_map_list]
        return dist_map_list

    def _rescale_image_array(self,dist_map_list,scaling_method="linear"):
        """
        Perform the actual re-scaling without image wrapper.
        ----
        Returns the re-scaled images in the image domain.
        """
        full_arr = numpy.array(dist_map_list)
        assert not numpy.isnan(full_arr).any(), "Array contains NaNs"
        # loop over list instead of using a 3D array
        assert len(dist_map_list) > 0, ("Length of input array is smaller than one: " +
                                        str(len(dist_map_list)))
        if scaling_method == "linear":
            # Normalization using 6 std creates [-0.5,0.5], then +0.5 results in [0,1]
            res_mean = numpy.mean(full_arr)
            res_std = numpy.sqrt(numpy.var(full_arr))
            dm_rescaled = [(img - res_mean)/(6 * res_std) + 0.5 for img in dist_map_list]
        elif scaling_method == "minmax_individual":
            # Does this suffer from shallow copies?
            dm_rescaled = [(img - img.min())/(img.max() - img.min()) for img in dist_map_list]
        elif scaling_method == "perpixel":
            res_mean = numpy.mean(full_arr,axis=0)
            res_std = numpy.sqrt(numpy.var(full_arr,axis=0))
            # / equals numpy.true_divide() equals element wise division
            dm_rescaled = [(img - res_mean)/(6 * res_std) + 0.5 for img in dist_map_list]
        else:
            raise ValueError("Incorrect scaling method for image creation specified")
        return dm_rescaled

    @staticmethod
    def set_image_values_outside_boundary(img_unbound,target_min=0,target_max=1):
        """
        Sets image values if they lie around the desired borders. Expects an image.
        ----
        Returns the image capped on the min and max values.
        """
        too_large = img_unbound > target_max
        too_small = img_unbound < target_min
        img = img_unbound
        img[too_large] = target_max
        img[too_small] = target_min
        return img
