import trace1 as G
import open3d as o3d
import numpy as np

def get_surfaces_from_point_cloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    planes = pcd.detect_planar_patches(normal_variance_threshold_deg=60, coplanarity_deg=75, outlier_ratio=0.75, min_plane_edge_length=0.0, min_num_points=0, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)) #si potrebbe fare fine tuning
    surfaces = []
    edges = []
    for i in range (8):
        for j in range (i+1,8):
            edges.append([i,j])
    for plane in planes:
        vertices = np.asarray(plane.get_box_points())
        edges = sorted(edges, key= lambda edge: np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
        midpoints = [np.mean([vertices[edge[0]],vertices[edge[1]]], axis=0) for edge in edges[:4]]
        p0 = midpoints.pop(0)
        midpoints = sorted(midpoints, key= lambda point: np.linalg.norm(point - p0))
        p1 = midpoints[0]
        p2 = midpoints[1]
        points = [p0, p1, p2]
        surfaces.append(G.Surface(points))

    return surfaces

