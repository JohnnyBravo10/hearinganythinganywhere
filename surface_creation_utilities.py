import trace1 as G
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay

#method to create rectangular surfaces
def get_surfaces_from_point_cloud(pcd_path):
    #loading point cloud and detecting planar patches (boxes with small z dimension)
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
        #identifying the four shortest edges
        midpoints = [np.mean([vertices[edge[0]],vertices[edge[1]]], axis=0) for edge in edges[:4]]
        p0 = midpoints.pop(0)
        midpoints = sorted(midpoints, key= lambda point: np.linalg.norm(point - p0))
        p1 = midpoints[0]
        p2 = midpoints[1]
        points = [p0, p1, p2]
        #the surface is a section of the planar patch
        surfaces.append(G.Surface(np.array(points)))

    return surfaces


#method to optimize surfaces to possibly have also triangular shape or generic parallelogram shape
def get_surfaces_from_point_cloud_with_optimization(pcd_path, cut_impurity = 0.05, steps_per_side = 11):
    #loading point cloud and detecting planar patches (boxes with small z dimension)
    pcd = o3d.io.read_point_cloud(pcd_path)
    planes = pcd.detect_planar_patches(normal_variance_threshold_deg=60, coplanarity_deg=75, outlier_ratio=0.75, min_plane_edge_length=0.0, min_num_points=0, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)) #si potrebbe fare fine tuning
    surfaces = []
    edges = []
    for i in range (8):
        for j in range (i+1,8):
            edges.append([i,j])
    for plane in planes:
        print(plane)
        vertices = np.asarray(plane.get_box_points())
        #internal_points = pcd.points[plane.get_point_indices_within_bounding_box(pcd.points)]
        target = len(plane.get_point_indices_within_bounding_box(pcd.points))
        edges = sorted(edges, key= lambda edge: np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))

        midpoints = [np.mean([vertices[edge[0]],vertices[edge[1]]], axis=0) for edge in edges[:4]]
        p0 = midpoints.pop(0)
        midpoints = sorted(midpoints, key= lambda point: np.linalg.norm(point - p0))
        p1 = midpoints[0]
        p2 = midpoints[1]
        points = [p0, p1, p2]
        best_surface = [True, points, np.linalg.norm(p1 - p0) * np.linalg.norm(p2 - p0)]

        short_edges = edges[:4]

        #separating vertices in face up and face down
        face_up_vertices = []
        face_down_vertices = []
        for short_edge in short_edges:
            v1 = vertices[short_edge[0]]
            v2 = vertices[short_edge[1]]

            if v1[2] != v2[2]:
                if v1[2] < v2[2]:
                    face_down_vertices.append(v1)
                    face_up_vertices.append(v2)
                else:
                    face_down_vertices.append(v2)
                    face_up_vertices.append(v1)

            else:
                if v1[1] != v2[1]:
                    if v1[1] < v2[1]:
                        face_down_vertices.append(v1)
                        face_up_vertices.append(v2)
                    else:
                        face_down_vertices.append(v2)
                        face_up_vertices.append(v1)

                else:
                    if v1[0] != v2[0]:
                        if v1[0] < v2[0]:
                            face_down_vertices.append(v1)
                            face_up_vertices.append(v2)
                        else:
                            face_down_vertices.append(v2)
                            face_up_vertices.append(v1)
        
        face_up_vertices = sorted(face_up_vertices, key= lambda point: np.linalg.norm(point - face_down_vertices[0]))
        face_down_vertices = sorted(face_down_vertices, key= lambda point: np.linalg.norm(point - face_up_vertices[0]))

        cuts_to_try = np.linspace(0, 1, steps_per_side)

        for cut in cuts_to_try:

            #trying to cut part of the rectangle to use a generic parallelogram
            new_area = cut *  np.linalg.norm(face_down_vertices[0] - face_down_vertices[2]) * np.linalg.norm(face_down_vertices[1] - face_down_vertices[3])

            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2], face_down_vertices[1], face_up_vertices[1]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[3] + (1-cut)*face_down_vertices[1],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[1], face_down_vertices[2], face_up_vertices[2]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:  
                print("optimization found 1")#######################
                best_surface = [True, [np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0),  np.mean([cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2], cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2]], axis=0), np.mean([cut*face_down_vertices[3] + (1-cut)*face_down_vertices[1],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[1]], axis=0)], new_area]
            
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[2], face_up_vertices[2], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[0], face_down_vertices[3], face_up_vertices[3]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[1], face_up_vertices[1], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3], face_down_vertices[0], face_up_vertices[0]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:
                print("optimization found 2")#######################
                best_surface = [True, [np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0),  np.mean([cut*face_down_vertices[2] + (1-cut)*face_down_vertices[0], cut*face_up_vertices[2] + (1-cut)*face_up_vertices[0]], axis=0), np.mean([cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3]], axis=0)], new_area]

            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[1],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[1], face_down_vertices[2], face_up_vertices[2]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[3] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[2], face_down_vertices[1], face_up_vertices[1]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:
                print("optimization found 3")#######################
                best_surface = [True, [np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0),  np.mean([cut*face_down_vertices[0] + (1-cut)*face_down_vertices[1], cut*face_up_vertices[0] + (1-cut)*face_up_vertices[1]], axis=0), np.mean([cut*face_down_vertices[3] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[2]], axis=0)], new_area]

            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[1], face_up_vertices[1], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0], face_down_vertices[3], face_up_vertices[3]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[2], face_up_vertices[2], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3], face_down_vertices[0], face_up_vertices[0]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:
                print("optimization found 4")#######################
                best_surface = [True, [np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0),  np.mean([cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0], cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0]], axis=0), np.mean([cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3]], axis=0)], new_area]

            #trying to cut part of the rectangle to use triangle
            new_area = np.linalg.norm(face_down_vertices[0] - face_down_vertices[2]) * np.linalg.norm(face_down_vertices[1] - face_down_vertices[3]) * 0.5

            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2], face_down_vertices[1], face_up_vertices[1]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2], face_down_vertices[2], face_up_vertices[2]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:    
                print("optimization found 5")#######################
                best_surface = [False, [np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0),  np.mean([ cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2]], axis=0), np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0)], new_area]

            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3], face_down_vertices[1], face_up_vertices[1]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3], face_down_vertices[2], face_up_vertices[2]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:    
                print("optimization found 6")#######################
                best_surface = [False, [np.mean([face_down_vertices[0], face_up_vertices[0]], axis=0),  np.mean([ cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3]], axis=0), np.mean([face_down_vertices[2], face_up_vertices[2]], axis=0)], new_area]
                
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0], face_down_vertices[2], face_up_vertices[2]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0], face_down_vertices[1], face_up_vertices[1]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:    
                print("optimization found 7")#######################
                best_surface = [False, [np.mean([face_down_vertices[2], face_up_vertices[2]], axis=0),  np.mean([cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0]], axis=0), np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0)], new_area]

            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3], face_down_vertices[2], face_up_vertices[2]])) + noise)
            noise = np.random.normal(scale=1e-8, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3], face_down_vertices[1], face_up_vertices[1]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < cut_impurity * target) and new_area < best_surface[2]:    
                print("optimization found 8")#######################
                best_surface = [False, [np.mean([face_down_vertices[0], face_up_vertices[0]], axis=0),  np.mean([cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3]], axis=0), np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0)], new_area]

        surfaces.append(G.Surface(np.array(best_surface[1]), best_surface[0]))

    return surfaces