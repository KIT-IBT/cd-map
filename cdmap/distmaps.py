"""
This module creates the craniosynostosis distance maps.
"""
import warnings
import numpy
import vtk
import vtk.util.numpy_support

class MeshLoader():
    """
    Loads a mesh from the 496-dataset (currently, only ply is supported). If a
    text file is provided, it expects the 10 landmarks, otherwise it assumes
    dense correspondence and uses the shape model based landmarks.
    """
    def __init__(self,
            path_to_file,
            path_to_lms=None,
            datatype=None,
            lms_center_ids = None):
        assert datatype is not None, "No datatype specified"
        if lms_center_ids is None:
            lms_center_ids = [8,9,5]
        self.polydata = MeshLoader.load_ply(path_to_file)
        if datatype == "subject":
            self.all_landmarks = MeshLoader.load_landmarks(path_to_lms)
            self.center_def_lms = self.all_landmarks[lms_center_ids]
        elif datatype == "instance":
            lm_ids = MeshLoader.define_shape_model_landmarks()
            vtk_points = vtk.util.numpy_support.vtk_to_numpy(
                    self.polydata.GetPoints().GetData())
            self.all_landmarks = vtk_points[lm_ids]
            defined_landmarks = [8,9,5]
            self.center_def_lms = self.all_landmarks[defined_landmarks]
        else:
            raise TypeError("Incorrect datatype specified, was " + str(datatype))
    def __call__(self):
        """
        Returns polydata and the three important landmarks to define the coordinate system.
        """
        return self.polydata, self.center_def_lms

    def get_points_and_cells(self):
        """
        Returns both points and cells in a tuple.
        """
        vtk_points = vtk.util.numpy_support.vtk_to_numpy(self.polydata.GetPoints().GetData())
        cell_list = vtk.util.numpy_support.vtk_to_numpy(self.polydata.GetPolys().GetData())
        vtk_cells = numpy.reshape(cell_list,[-1,4])[:,1:]
        return vtk_points,vtk_cells

    @staticmethod
    def load_ply(path_to_ply):
        """
        Uses the vtk ply reader to read ply data.
        """
        reader_ply = vtk.vtkPLYReader()
        reader_ply.SetFileName(path_to_ply)
        reader_ply.Update()
        polydata = reader_ply.GetOutput()
        # If there are no points in 'vtk_poly_data' something went wrong
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError("No point data could be loaded from '" + path_to_ply)
        return polydata

    @staticmethod
    def define_shape_model_landmarks():
        """
        Defines the 10 landmarks on the shape model data,
        as defined on our publicly available statistical shape model.
        Indices start at zero.
        https://zenodo.org/record/5638148
        """
        # From shape model v2 as of 2022-04
        indices = [3737, 2463, 4899, 9368, 4199, 4240, 9053, 8935, 2400, 11115]
        return indices

    @staticmethod
    def load_landmarks(path_to_txt):
        """
        Load landmarks from text file, expects 10x3 file as csv
        """
        all_landmarks = numpy.genfromtxt(path_to_txt,delimiter=',')
        return all_landmarks

class MeshIntersector():
    """
    Uses vtk polydata and the landmarks to create the center point and axes.
    When called, expects a list of a list of tuples (a list of rays with two
    elements each also in a list).
    Each of the two ray element is start and end point.
    Example of rays in coordinate axes from origin:
    [[(0,0,0),(1,0,0)],[(0,0,0),(0,1,0)],[(0,0,0),(0,0,1)]]
    """
    def __init__(self,
            polydata,
            center_def_lms):
        """
        Requires polydata and center defining landmarks.
        """
        self.polydata = polydata
        self.obb_tree = None
        self.center_def_lms = center_def_lms
        # If landmarks are not specified, assuming shape model data

    @staticmethod
    def normalize_to_unit_vector(input_vec):
        """
        Normalizes vector to length one in Euclidean space.
        """
        sanitized_vec = numpy.array([x for x in input_vec if str(x) != 'nan'])
        output_vec = sanitized_vec / numpy.sqrt(
                sum(sanitized_vec ** 2,0))
        return output_vec

    @staticmethod
    def define_axes_from_landmarks(landmarks):
        """
        Defines the three axes and the corresponding rotation matrix
        from the three landmarks.
        """
        pt_center = numpy.mean(landmarks[0:2],0)
        # transform coordinates
        x_axis = MeshIntersector.normalize_to_unit_vector(landmarks[2] - pt_center)
        y_axis = MeshIntersector.normalize_to_unit_vector(landmarks[0] - landmarks[1])
        z_axis = MeshIntersector.normalize_to_unit_vector(numpy.cross(x_axis,y_axis))
        # assemble rotation matrix
        R = numpy.transpose([x_axis,y_axis,z_axis])
        return pt_center,R

    def create_tree(self,polydata):
        """
        Creates oriented bounding boxes tree to prepare fast ray intersection.
        ----
        Returns an obb_tree
        """
        obb_tree = vtk.vtkOBBTree()
        obb_tree.SetDataSet(polydata)
        obb_tree.BuildLocator()
        return obb_tree

    def intersect_mesh(self,tuple_list):
        """
        Expects a list of a list of tuples (a list of rays with two
        elements each also in a list) for ray intersection with the polydata in self.
        Each of the two ray element is start and end point.
        Example of rays in coordinate axes from origin:
        [[(0,0,0),(1,0,0)],[(0,0,0),(0,1,0)],[(0,0,0),(0,0,1)]]
        ----
        Returns a list of list of tuples as an output (a list of directions,
        for each direction a list of intersection points as a 3D tuple).
        """
        assert isinstance(tuple_list,list), ("Expecting a list (of tuples), type is " +
                                          str(type(tuple_list)))
        # Check if we have a tree, and if not, then create one on the fly
        if self.obb_tree is None:
            self.obb_tree = self.create_tree(self.polydata)
        full_point_list = []
        for cur_tuple in tuple_list:
            assert isinstance(cur_tuple,list) and len(cur_tuple) == 2, ("Bad check " +
                    "in the beginning, expected a list of lists "
                    "containing two 3-tuples, got " + str(type(cur_tuple)) +
                    " " + str(len(cur_tuple)))
            pt_center = cur_tuple[0]
            pt_target = cur_tuple[1]
            intersection_array = vtk.vtkPoints()
            self.obb_tree.IntersectWithLine(pt_center, pt_target, intersection_array, None)
            pts_vtk_intersec_data = intersection_array.GetData()
            num_pts_vtk_intersec = pts_vtk_intersec_data.GetNumberOfTuples()
            single_point_list = []
            if num_pts_vtk_intersec > 0:
                for idx in range(num_pts_vtk_intersec):
                    _tup = pts_vtk_intersec_data.GetTuple3(idx)
                    single_point_list.append(_tup)
            full_point_list.append(single_point_list)
        return full_point_list

    @staticmethod
    def reduce_list_to_min_dist_tuple(tuple_hit_list,tuple_start):
        """
        Expects a list of tuples (a list of lenths as 3D tuples)
        and reduces it to the one with the smallest distance.
        ----
        Returns a tuple list of distances.
        """
        assert isinstance(tuple_hit_list,list), ("Expected inner list of hit points, got " +
                                              str(type(tuple_hit_list)))
        assert isinstance(tuple_start,tuple), "Expected start tuple, got " + str(type(tuple_start))
        for cur_tuple in tuple_hit_list:
            assert isinstance(cur_tuple,tuple) , ("Expected type tuple of hit points, got " +
                                               str(type(cur_tuple)))
            assert len(cur_tuple) == 3, ("Expected tulples of lengths 3 of hit points, got " +
                                         str(len(cur_tuple)))
        # If only one element, nothing to do
        if len(tuple_hit_list) == 0:
            return []
        if len(tuple_hit_list) == 1:
            return tuple_hit_list
        dist_list = []
        for ele in tuple_hit_list:
            dist_list.append(numpy.linalg.norm(numpy.array(ele) - numpy.array(tuple_start)))
        min_index = numpy.argmin(dist_list)
        return [tuple_hit_list[min_index]]

    @staticmethod
    def reduce_distance_list(list_of_tuple_list,tuples_list_rays):
        """
        Wrapper for reduce_list_to_min_dist_tuple() for our dataset.
        ----
        Returns a tuple list of target points.
        """
        assert isinstance(list_of_tuple_list,list), ("Expected outer list, got " +
                                                  str(type(list_of_tuple_list)))
        assert isinstance(tuples_list_rays,list), ("Expected tuples start list, got "
                                                + str(type(list_of_tuple_list)))
        # tuples list rays contains the point at position 0 and the end point at position 1
        reduced_list = [MeshIntersector.reduce_list_to_min_dist_tuple(
            cur_list,start_point[0]) for cur_list,start_point in zip(
                list_of_tuple_list,tuples_list_rays)]
        return reduced_list

    def define_tip(self,up_vector=numpy.array([0,0,500])):
        """
        Points a ray upward and computes tip as the intersection point with
        the surface mesh.
        ----
        Returns a numpy array of size 3.
        """
        pt_center,R = MeshIntersector.define_axes_from_landmarks(self.center_def_lms)
        up_vec_rotated = R @ up_vector
        intersection_vector = [[tuple(pt_center),tuple(up_vec_rotated)]]
        pt_tip = MeshIntersector.intersect_mesh(self,intersection_vector)
        pt_tip = pt_tip[0]
        if isinstance(pt_tip,list) and len(pt_tip) > 1:
            warnings.warn("Got more than one intersection, taking the smaller one", UserWarning)
            pt_tip = MeshIntersector.reduce_list_to_min_dist_tuple(pt_tip,pt_center)
        return numpy.array(pt_tip[0])

    def __call__(self,tuple_list):
        """
        Returns hit list, not distances, be careful!
        ----
        Returns a list of hit points closest to the origin.
        """
        full_hit_list = self.intersect_mesh(tuple_list)
        single_hit_list = MeshIntersector.reduce_distance_list(full_hit_list,tuple_list)
        return single_hit_list

class RayCreator():
    """
    Provides rays for different mapping types. Currently supported: Spherical (halfsphere),
    Arch-spherical (leftright_halfsphere), and Cylindrical (cylinder).
    Requires RayIntersector to compute the rays. For quadratical maps, use num_rays, for
    non-quadratical use x_arr and y_arr to overwrite.
    """
    def __init__(self,
            method,
            intersector,
            num_rays_x=28,
            num_rays_y=28,
            ray_length_mm=200):
        self.method = method
        self.intersector = intersector
        self.ray_length_mm = ray_length_mm
        # call methods to get tuples depending on the input
        self.x_arr,self.y_arr = self.initialize_arrays(method,num_rays_x,num_rays_y)

    def initialize_arrays(self,method,num_rays_x,num_rays_y):
        """
        Initialize the image arrays from the number of rays requested and the method chosen.
        """
        assert method is not None, "Method missing, valid are: 'half_sphere' and 'cylinder'"
        if method == "halfsphere":
            x_arr,y_arr = RayCreator.get_half_sphere_array(num_rays_x,num_rays_y)
        elif method == "leftright_halfsphere":
            x_arr,y_arr = RayCreator.get_left_right_halfsphere_array(num_rays_x,num_rays_y)
        elif method == "arcsin_halfsphere":
            x_arr,y_arr = RayCreator.get_arcsin_half_sphere_array(num_rays_x,num_rays_y)
        elif method == "cylinder":
            x_arr,y_arr = RayCreator.get_cylinder_array(num_rays_x,num_rays_y)
        elif method == "arcsin_cylinder":
            x_arr,y_arr = RayCreator.get_arcsin_cylinder_array(num_rays_x,num_rays_y)
        elif method == "noears_halfsphere":
            x_arr,y_arr = RayCreator.get_half_sphere_array(num_rays_x,num_rays_y)
        else:
            raise TypeError("method type incorrect, given ", method)
        return x_arr,y_arr

    # Half sphere array: x = rotation angle phi and y = rotation array theta

    @staticmethod
    def get_half_sphere_array(num_rays_x,num_rays_y):
        """
        Creates halfsphere array (phi: 0..2pi, theta: 0..pi/2)
        ----
        Returns the two angle directions.
        """
        x_arr = numpy.linspace(0, 2*numpy.pi, num_rays_x+1)[0:num_rays_x]
        y_arr = numpy.linspace(0.5*numpy.pi, 0, num_rays_y+1)[1:num_rays_y+1]
        return x_arr, y_arr

    # left-right next generation half_sphere compatible array
    @staticmethod
    def get_left_right_halfsphere_array(num_rays_x,num_rays_y):
        """
        Creates leftright halfsphere array (phi: 0..pi, theta: 0..pi)
        ----
        Returns the two angle directions.
        """
        x_arr = numpy.linspace(0, numpy.pi, num_rays_x+1)[0:num_rays_x]
        y_arr = numpy.linspace(0, numpy.pi, num_rays_y+1)[0:num_rays_y]
        return x_arr, y_arr

    # Half sphere array: x = rotation angle phi and y = rotation array theta
    @staticmethod
    def get_arcsin_half_sphere_array(num_rays_x,num_rays_y):
        """
        Creates leftright halfsphere array (phi: 0..2pi, theta: 0..pi/2), same
        as halfsphere array.
        ----
        Returns the two angle directions.
        """
        x_arr = numpy.linspace(0, 2*numpy.pi, num_rays_x+1)[0:num_rays_x]
        y_arr = numpy.arcsin(numpy.linspace(1, 0, num_rays_y+1)[1:num_rays_y+1])
        return x_arr, y_arr


    # Cylinder array: x = rotation angle phi and y = Cartesian y
    @staticmethod
    def get_cylinder_array(num_rays_x,num_rays_y,tip=None):
        """
        Creates cylinder array (phi: 0..2pi, h: 0..h_tip)
        ----
        Returns the two directions.
        """
        if tip is None:
            tip = 1
        x_arr = numpy.linspace(0, 2*numpy.pi, num_rays_x+1)[0:num_rays_x]
        y_arr = numpy.linspace(tip, 0, num_rays_y+1)[1:num_rays_y+1]
        return x_arr,y_arr

    # Arcsin cylinder array: x = rotation angle phi and y = Cartesian y
    @staticmethod
    def get_arcsin_cylinder_array(num_rays_x,num_rays_y,tip=None):
        """
        Creates arcin-cylinder array (phi: 0..2pi, h: 0..h_tip), same as cylinder
        ----
        Returns the two directions.
        """
        if tip is None:
            tip = 1
        x_arr = numpy.linspace(0, 2*numpy.pi, num_rays_x+1)[0:num_rays_x]
        y_arr = numpy.arcsin(numpy.linspace(tip, 0, num_rays_y+1)[1:num_rays_y+1] * numpy.sin(1))
        return x_arr,y_arr

    def create_tuples_from_half_sphere(self):
        """
        Creates and returns tuple list from self.x_arr and self.y_arr.
        """
        pt_center,R = MeshIntersector.define_axes_from_landmarks(self.intersector.center_def_lms)
        tuple_list = RayCreator.map_transform_spherical(self.x_arr,
                                                        self.y_arr,
                                                        pt_center,
                                                        self.ray_length_mm,
                                                        R)
        return tuple_list

    def create_tuples_from_left_right_half_sphere(self):
        """
        Creates and returns tuple list from self.x_arr and self.y_arr.
        """
        pt_center,R = MeshIntersector.define_axes_from_landmarks(self.intersector.center_def_lms)
        tuple_list = RayCreator.map_transform_left_right_spherical(self.x_arr,
                                                                   self.y_arr,
                                                                   pt_center,
                                                                   self.ray_length_mm,
                                                                   R)
        return tuple_list

    def create_no_ears_tuples_from_half_sphere(self):
        """
        Creates and returns tuple list from self.x_arr and self.y_arr.
        Cuts off 25% of the lower ear part which is roughly the image without ears.
        """
        center_old,R = MeshIntersector.define_axes_from_landmarks(self.intersector.center_def_lms)
        pt_tip = self.intersector.define_tip()
        # Move new center point a little bit to the top, maybe 25%?
        pt_center_new = center_old + (pt_tip - center_old) * 0.25
        tuple_list = RayCreator.map_transform_spherical(self.x_arr,
                                                        self.y_arr,
                                                        pt_center_new,
                                                        self.ray_length_mm,
                                                        R)
        return tuple_list

    def create_tuples_from_cylinder(self):
        """
        Creates and returns tuple list from self.x_arr, self.y_arr and the tip point.
        """
        tuple_list = [[(None,None,None),(None,None,None)]] * len(self.x_arr) * len(self.y_arr)
        pt_center,R = MeshIntersector.define_axes_from_landmarks(self.intersector.center_def_lms)
        pt_tip = self.intersector.define_tip()
        tuple_list = RayCreator.map_transform_cylindrical(self.x_arr,
                                                          self.y_arr,
                                                          pt_center,
                                                          pt_tip,
                                                          self.ray_length_mm,
                                                          R)
        return tuple_list

    def create_tuples(self):
        """
        Wrapper for the different tuple creation functions. Currently supports
        spherical, arch-spherical, cylindrical, the spherical with the arcsin variant,
        which provides a more regular spacing but leaves out a large portion at
        the tip of the head.
        """
        assert self.method is not None, "Method missing, valid are: 'halfsphere' and 'cylinder'"
        if self.method == "halfsphere":
            tuples_in = self.create_tuples_from_half_sphere()
        elif self.method == "leftright_halfsphere":
            tuples_in = self.create_tuples_from_left_right_half_sphere()
        elif self.method == "cylinder":
            tuples_in = self.create_tuples_from_cylinder()
        elif self.method == "arcsin_cylinder":
            tuples_in = self.create_tuples_from_cylinder()
        elif self.method == "arcsin_halfsphere":
            tuples_in = self.create_tuples_from_half_sphere()
        elif self.method == "noears_halfsphere":
            tuples_in = self.create_no_ears_tuples_from_half_sphere()
        else:
            raise ValueError("Incorrect method provided, is " + str(self.method))
        return tuples_in

    def create_tuples_from_direction_vectors(self,target_points):
        """
        Creates and returns tuple list from direction vectors, transformed using the landmarks.
        """
        pt_center,R = MeshIntersector.define_axes_from_landmarks(self.intersector.center_def_lms)
        tuple_list = [[(None,None,None),(None,None,None)]] * len(target_points)
        for point,i in zip(target_points,range(len(target_points))):
            dir_vec = R @ point
            pt_end = pt_center + self.ray_length_mm * dir_vec
            tuple_list[i] = [tuple(pt_center),tuple(pt_end)]
        return tuple_list

    def create_tuples_from_file(self,path_to_txt):
        """
        Reads a csv file and returns tuple list from the direction vectors of the file.
        """
        target_points = numpy.genfromtxt(path_to_txt,delimiter=',')
        tuple_list = self.create_tuples_from_direction_vectors(target_points)
        return tuple_list

    @staticmethod
    def map_transform_left_right_spherical(x_arr,
                                           y_arr,
                                           pt_center=None,
                                           ray_length_mm=1,
                                           R=numpy.eye(3)):
        """
        Creates and returns tuple list from x and y array using the arch-spherical method.
        """
        if pt_center is None:
            pt_center = [0,0,0]
        tuple_list = [[(None,None,None),(None,None,None)]] * len(x_arr) * len(y_arr)
        for x,x_i in zip(x_arr,range(len(x_arr))):
            for y,y_i in zip(y_arr,range(len(y_arr))):
                dir_vec_sph = numpy.array([numpy.sin(x) * numpy.cos(y),
                                           numpy.cos(x),
                                           numpy.sin(x) * numpy.sin(y)])
                dir_vec = R @ dir_vec_sph
                pt_end = pt_center + ray_length_mm * dir_vec
                tuple_list[y_i*len(y_arr) + x_i] = [tuple(pt_center),tuple(pt_end)]
        return tuple_list

    @staticmethod
    def map_transform_spherical(x_arr,
                                y_arr,
                                pt_center=None,
                                ray_length_mm=1,
                                R=numpy.eye(3)):
        """
        Creates and returns tuple list from x and y array using the spherical method.
        """
        if pt_center is None:
            pt_center = [0,0,0]
        tuple_list = [[(None,None,None),(None,None,None)]] * len(x_arr) * len(y_arr)
        for x,x_i in zip(x_arr,range(len(x_arr))):
            for y,y_i in zip(y_arr,range(len(y_arr))):
                dir_vec_sph = numpy.array([numpy.cos(x) * numpy.cos(y),
                                           numpy.sin(x) * numpy.cos(y),
                                           numpy.sin(y)])
                dir_vec = R @ dir_vec_sph
                pt_end = pt_center + ray_length_mm * dir_vec
                tuple_list[y_i*len(y_arr) + x_i] = [tuple(pt_center),tuple(pt_end)]
        return tuple_list

    @staticmethod
    def map_transform_cylindrical(x_arr,
                                  y_arr,
                                  pt_center=None,
                                  pt_tip=None,
                                  ray_length_mm=1,
                                  R=numpy.eye(3)):
        """
        Creates and returns tuple list from x and y array using the cylindrical method.
        """
        if pt_center is None:
            pt_center = [0,0,0]
        if pt_tip is None:
            pt_tip = [0,0,1]
        tuple_list = [[(None,None,None),(None,None,None)]] * len(x_arr) * len(y_arr)
        for x,x_i in zip(x_arr,range(len(x_arr))):
            for y,y_i in zip(y_arr,range(len(y_arr))):
                pt_start = pt_center + (y * (pt_tip - pt_center))
                dir_vec_cyl = numpy.array([numpy.cos(x) * 1, numpy.sin(x), 0])
                dir_vec = R @ dir_vec_cyl
                pt_end = pt_start + ray_length_mm * dir_vec
                tuple_list[y_i*len(y_arr) + x_i] = [tuple(pt_start),tuple(pt_end)]
        return tuple_list

    def compute_target_points(self,tuples_in):
        """
        Uses the MeshIntersector to create the target points (not distances).
        """
        point_list_with_min_dist = self.intersector(tuples_in)
        return point_list_with_min_dist

    def distances_from_target_points(self,point_list,tuples_in,num_dimensions=2):
        """
        Uses the MeshIntersector to extract distances from the target points.
        """
        dists = numpy.array([None] * len(point_list))
        for i,point,tuple_in in zip(range(len(point_list)),point_list,tuples_in):
            if point != []:
                dists[i] = numpy.linalg.norm(numpy.array(point) - numpy.array(tuple_in[0]))
        if num_dimensions==2:
            dists = dists.reshape(len(self.x_arr),len(self.y_arr))
        return dists.astype(float)

    def __call__(self):
        """
        Performs the full pipeline: Uses the MeshIntersector to create tuples,
        compute target points and distances.
        ----
        Returns the distances.
        """
        tuples_in = self.create_tuples()
        point_list_with_min_dist = self.compute_target_points(tuples_in)
        dists = self.distances_from_target_points(point_list_with_min_dist,tuples_in)
        return dists

class AttributionTransformer():
    """
    This class contains only static methods to transform attributions
    onto the 3D surface scans.
    """
    def __init__(self,method):
        self.method = method

    def __call__(self, mesh, image, tip_steps = 224):
        """
        Extract attribution from mesh and image.
        ----
        Returns color values of attribution.
        """
        polydata,lms = mesh()
        points,_ = mesh.get_points_and_cells()
        pt_center,R = MeshIntersector.define_axes_from_landmarks(lms)
        # If cylindrical mapping, we also need the tip
        if self.method == "cylinder":
            intersector = MeshIntersector(polydata,lms)
            pt_tip = intersector.define_tip()
        else:
            pt_tip = None
        points_transformed = AttributionTransformer.mesh_points_to_distance_map(self.method,
                                                                                points,
                                                                                pt_center,
                                                                                R,
                                                                                pt_tip=pt_tip)
        color_vals = AttributionTransformer.color_values_from_transformed_points(self.method,
                                                                                 points_transformed,
                                                                                 image)
        return color_vals

    @staticmethod
    def color_values_from_transformed_points(method,points_transformed,image):
        """
        Use points tranformed from the 3D domain to the 2D domain and
        interpolate color values in the 2D domain using an image.
        """
        image = numpy.sum(image,2)
        # image_length = len(image)
        img_len_x = image.shape[0]-1
        img_len_y = image.shape[1]-1
        if method == "halfsphere":
            img_coords = numpy.array(points_transformed)[:,:2] / [2 * numpy.pi, 0.5 * numpy.pi]
            img_coords_rev = [[ic[1],ic[0]] for ic in img_coords]
        elif method == "leftright_halfsphere":
            img_coords = numpy.array(points_transformed)[:,:2] / [numpy.pi, numpy.pi]
            img_coords_rev = [[ic[0],ic[1]] for ic in img_coords]
        elif method == "cylinder":
            # Now scale to image:
            img_coords = numpy.array(points_transformed)[:,[0,2]] / [2 * numpy.pi, 1]
            img_coords_rev = [[ic[1],ic[0]] for ic in img_coords]
        img_coords_cor_size = numpy.array(img_coords_rev) * [img_len_x,img_len_y]
        # We assume that we have attribution values on the whole head
        # If ray distribution however was different we need to re-scale it
        color_values = [AttributionTransformer.linear_interpolation_on_image_grid(
            image,
            point,
            offset_boundary=False) for point in img_coords_cor_size]
        return color_values

    @staticmethod
    def reverse_normalize_image(img):
        """
        Returns scaled image from [0..255] to [1..0]
        """
        return 1 - img / 255

    @staticmethod
    def euclidean_to_dm_spherical(pts):
        """
        Expects already centered points. Transforms the 3D points
        into phi and theta values using the spherical transformation
        for later interpolation.
        ----
        Returns list of coordinates.
        """
        dists = [numpy.linalg.norm(point,2) for point in pts]
        new_frame_coords = dists
        for i,_ in enumerate(new_frame_coords):
            r = dists[i]
            point = pts[i]
            theta = numpy.arcsin(point[2]/r)
            phi = numpy.arctan2(point[1],point[0])
            if phi < 0:
                phi = phi + 2* numpy.pi
            eucl = [phi, theta, r]
            new_frame_coords[i] = eucl
        return new_frame_coords

    @staticmethod
    def euclidean_to_dm_leftright_halfsphere(pts):
        """
        Expects already centered points. Transforms the 3D points
        into phi and theta values using the arch-spherical transformation
        for later interpolation.
        ----
        Returns list of coordinates.
        """
        # Assumes already centered points
        dists = [numpy.linalg.norm(point,2) for point in pts]
        new_frame_coords = dists
        for i,_ in enumerate(new_frame_coords):
            r = dists[i]
            point = pts[i]
            # Original
            theta = numpy.arccos(point[1]/r)
            phi = numpy.arctan2(point[2],point[0])
            if phi < 0:
                phi = phi + 2* numpy.pi
            eucl = [phi, theta, r]
            new_frame_coords[i] = eucl
        return new_frame_coords

    @staticmethod
    def euclidean_to_dm_cylinder(pts):
        """
        Expects already centered points. z-direction is expected to be scaled
        from 0 to 1. Transforms the 3D points into phi and theta values using
        the cylindrical transformation for later interpolation.
        ----
        Returns list of coordinates.
        """
        # Assumes already centered points
        new_frame_coords = [[0,0,0] for point in pts]
        rhos = [numpy.linalg.norm(point[:2]) for point in pts]
        for i,_ in enumerate(pts):
            point = pts[i]
            z = pts[i][2]
            rho = rhos[i]
            phi = numpy.arctan2(point[1],point[0])
            if phi < 0:
                phi = phi + 2* numpy.pi
            eucl = [phi, rho, z]
            new_frame_coords[i] = eucl
        return new_frame_coords

    @staticmethod
    def mesh_points_to_distance_map(method,points,pt_center,R,pt_tip=None):
        """
        This is a wrapper, to start off the transformation depending on the method
        ----
        Returns transformed list of points.
        """
        if method == "halfsphere":
            pts_diff = points - pt_center
            pts_rot = [numpy.transpose(R) @ point for point in pts_diff]
            pts_fin = AttributionTransformer.euclidean_to_dm_spherical(pts_rot)
        elif method == "leftright_halfsphere":
            pts_diff = points - pt_center
            pts_rot = [numpy.transpose(R) @ point for point in pts_diff]
            pts_fin = AttributionTransformer.euclidean_to_dm_leftright_halfsphere(pts_rot)
        elif method == "cylinder":
            pts_diff = points - pt_center
            # tip_scaling = pt_tip - pt_center
            pt_tip_scaled = [numpy.transpose(R) @ (pt_tip - pt_center)]
            length_tip = numpy.linalg.norm(pt_tip_scaled,2)
            # Make list of points
            pts_rot = [numpy.transpose(R) @ point for point in pts_diff]
            pts_rot = [point / [1,1,length_tip] for point in pts_rot]
            pts_fin = AttributionTransformer.euclidean_to_dm_cylinder(pts_rot)
        return pts_fin

    @staticmethod
    def linear_interpolation_on_image_grid(img,point,offset_boundary=False):
        """
        Expects image and point on that image. Interpolate the color value
        of the image on the position of the point using bilinear interpolation.
        This implementation works on a regular grid, but is substantially faster than
        the standard 2d interpolation libraries of matlab or scipy.
        ----
        Returns interpolated color value of that point. If is not inside
        the image, returns zero.
        """
        # Idea: Add offset to extrapolate parts of image of boundary +-1
        lower = numpy.array(numpy.floor(point),dtype=int)
        upper = numpy.array(numpy.ceil(point),dtype=int)
        if (offset_boundary is True) and ( (any(lower == -1)) or (any(upper == len(img)))):
            pt_rel = point - lower
            pt_rel_rec = 1 - pt_rel
            # Catch corner cases:
            if lower[0] == -1 and lower[1] == -1:
                return img[upper[0],upper[1]]
            if lower[0] == -1 and upper[1] == len(img):
                return img[upper[0],lower[1]]
            if upper[0] == len(img) and lower[1] == -1:
                return img[lower[0],upper[1]]
            if upper[0] == len(img) and upper[1] == len(img):
                return img[lower[0],lower[1]]
            # Catch edge cases:
            if lower[0] == -1:
                c = img[upper[0],lower[1]] * pt_rel[0] * pt_rel_rec[1]
                d = img[upper[0],upper[1]] * pt_rel[0] * pt_rel[1]
                pt_int = c + d
            elif lower[1] == -1:
                b = img[lower[0],upper[1]] * pt_rel_rec[0] * pt_rel[1]
                d = img[upper[0],upper[1]] * pt_rel[0] * pt_rel[1]
                pt_int = b + d
            elif upper[0] == len(img):
                a = img[lower[0],lower[1]] * pt_rel_rec[0] * pt_rel_rec[1]
                b = img[lower[0],upper[1]] * pt_rel_rec[0] * pt_rel[1]
                pt_int = a + b
            elif upper[1] == len(img):
                a = img[lower[0],lower[1]] * pt_rel_rec[0] * pt_rel_rec[1]
                c = img[upper[0],lower[1]] * pt_rel[0] * pt_rel_rec[1]
                pt_int = a + c
            return pt_int
        # Else: start normal routine
        if any(lower < 0) or any(upper > len(img) - 1):
            pt_int = 0
        else:
            img = numpy.array(img)
            pt_rel = point - lower
            pt_rel_rec = 1 - pt_rel
            a = img[lower[0],lower[1]] * pt_rel_rec[0] * pt_rel_rec[1]
            b = img[lower[0],upper[1]] * pt_rel_rec[0] * pt_rel[1]
            c = img[upper[0],lower[1]] * pt_rel[0] * pt_rel_rec[1]
            d = img[upper[0],upper[1]] * pt_rel[0] * pt_rel[1]
            pt_int = a + b + c + d
        return pt_int
