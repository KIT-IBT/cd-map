import argparse
import distmaps
import painting
import numpy
import sys
import warnings
import skimage.io

"""Create asymmetric hit points
            Required inputs:
            input_path_ply=/path/to/ply.ply
            If subject:
            --input_path_landmarks=/path/to/lms.txt
            Outputs:
            --output_path_hit_points=/path/to/file.csv
            --output_path_distances=/path/to/file.npy
            --output_path_image=/path/to/file.png
            --ray_file=/path/to/icosphere.csv
            Optional:
            --num_rays
            --method
            __doc__
"""

def parseargs():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_path_ply', type=str, help='input path for ply file')
    parser.add_argument('--input_path_landmarks', '-l', type=str, default=None , help='input path for landmark file (if shape model instance, not required)')
    parser.add_argument('--ray_file', type=str, default=None, help='ray file for icosphere')
    parser.add_argument('--num_rays_x', type=int, default=None, help='number of rays in x direction, overwrites num_rays')
    parser.add_argument('--num_rays_y', type=int, default=None, help='number of rays in y direction, overwrites num_rays')
    parser.add_argument('--num_rays', type=int, default=49, help='number of rays equals image length')
    parser.add_argument('--output_path_hit_points', '-p', type=str, default=None , help='output hit points')
    parser.add_argument('--output_path_distances', '-d', type=str, default=None , help='output distances from map')
    parser.add_argument('--output_path_image', '-m', type=str, default=None , help='output image (png)')
    parser.add_argument('--method', type=str, default='halfsphere', help='method, default half sphere')
    # parser.add_argument('--scaling', type=str, default='linear', help='scaling, default linear')
    parser.add_argument('--verbose', '-v', type=int, default=1, help='verbosity level')
    args = parser.parse_args()
    return args
args = parseargs()
# print(args)

if args.input_path_landmarks == None:
    ply_datatype = "instance"
else:
    ply_datatype = "subject"

if args.output_path_distances == None and args.output_path_image == None and args.output_path_hit_points == None:
    warnings.warn("No output file specified.")

# Load mesh and create rays
loader = distmaps.MeshLoader(path_to_file=args.input_path_ply,path_to_lms=args.input_path_landmarks,datatype=ply_datatype)
intersector = distmaps.MeshIntersector(loader.polydata,loader.center_defining_landmarks)

if args.num_rays_x is None:
    args.num_rays_x = args.num_rays

if args.num_rays_y is None:
    args.num_rays_y = args.num_rays

creator = distmaps.RayCreator(method=args.method,intersector=intersector,num_rays_x=args.num_rays_x, num_rays_y=args.num_rays_y)
# cur_dists = creator()
# Creator only returns distances, we want points as well
if args.ray_file is not None:
    tuples_in = creator.createTuplesFromFile(args.ray_file)
else:
    tuples_in = creator.create_tuples()

hit_points_list = creator.compute_target_points(tuples_in)
numpy.set_printoptions(threshold=sys.maxsize)

if args.output_path_hit_points != None:
    # Replace empty values with zeros
    hit_points_nonan = [ [(0,0,0)] if val == [] else val for val in hit_points_list]
    hit_points = numpy.asarray(hit_points_nonan,dtype=object)[:,0,:]
    hit_points_nonan = hit_points
    numpy.savetxt(args.output_path_hit_points,hit_points_nonan,delimiter=',')

if args.output_path_distances is not None or args.output_path_image is not None:
    if args.ray_file is None:
        distances = creator.distances_from_target_points(hit_points_list,tuples_in)
        nan_inpainter = painting.NanInpainter()
        distances = nan_inpainter(distances)
    else:
        distances = creator.distances_from_target_points(hit_points_list,tuples_in,num_dimensions=1)

    if args.output_path_distances != None:
        numpy.save(args.output_path_distances,distances)

    if args.output_path_image != None:
        painter = painting.ImageFromDistancesCreator()
        img = painter([distances])[0]
        img_uint8 = numpy.uint8(img * 255)
        skimage.io.imsave(args.output_path_image,img_uint8)

