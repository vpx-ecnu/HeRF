import math
import heapq
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

import time

INDEX_MODE = "xyzxyz"


def distance_point_to_box(point, box_min, box_max):
    x, y, z = point
    x_min, y_min, z_min = box_min
    x_max, y_max, z_max = box_max

    # distance along axis
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    dz = max(z_min - z, 0, z - z_max)

    # inside
    if dx == 0 and dy == 0 and dz == 0:
        return 0.0

    # outside
    return dx * dx + dy * dy + dz * dz


def get_index_xyz(index, mode, device):
    if mode == "xxyyzz":
        l = int((len(index) + 2) / 3)
        x = index[:l]
        y = index[l : 2 * l]
        z = index[2 * l :]
    elif mode == "xyzxyz":
        x = index[::3]
        y = index[1::3]
        z = index[2::3]
    else:
        raise ValueError("mode must be xxyyzz or xyzxyz")
    # print('x, y, z', x, y, z)
    return torch.tensor((int(x, 2), int(y, 2), int(z, 2)), device=device)


class KDNode:
    def __init__(
        self,
        axis=None,
        axis_mid=None,
        axis_left=None,
        axis_right=None,
        left=None,
        right=None,
        layer=None,
        index=None,
        is_leaf=False,
        confidence=None,
    ):
        self.axis = axis
        self.axis_mid = axis_mid
        self.axis_left = axis_left
        self.axis_right = axis_right
        self.left = left
        self.right = right
        self.layer = layer
        self.index = get_index_xyz(index, INDEX_MODE, axis_mid.device)
        self.is_leaf = is_leaf
        self.confidence = confidence


class KDVoxNode:
    def __init__(
        self,
        axis=None,
        left=None,
        right=None,
        layer=None,
        box_min=None,
        box_max=None,
        index=None,
        is_leave=False,
        confidence=None,
    ):
        self.axis = axis
        self.left = left
        self.right = right
        self.layer = layer
        self.box_min = box_min
        self.box_max = box_max
        self.index = get_index_xyz(index, INDEX_MODE, box_max.device)
        self.is_leave = is_leave
        self.dist = torch.inf
        self.confidence = confidence

    def __lt__(self, other):
        return self.dist < other.dist

    def __le__(self, other):
        return self.dist <= other.dist

    def __gt__(self, other):
        return self.dist > other.dist

    def __ge__(self, other):
        return self.dist >= other.dist


class KDCache:
    def __init__(self, kdtree, scene_bbox, cache_grid_num):
        self.device = kdtree.device
        self.in_grid_thre = 0.0
        self.scene_bbox = scene_bbox.to(self.device)
        self.cache_grid_num = cache_grid_num
        self.cache_grid_size = (
            self.scene_bbox[1] - self.scene_bbox[0]
        ) / cache_grid_num
        grid_coord = self.build_grid_coordinate()
        self.cache = self.build(grid_coord, kdtree)

    def build_grid_coordinate(self):
        x = torch.arange(
            self.scene_bbox[0][0],
            self.scene_bbox[1][0],
            self.cache_grid_size[0],
            device=self.device,
        )
        y = torch.arange(
            self.scene_bbox[0][1],
            self.scene_bbox[1][1],
            self.cache_grid_size[1],
            device=self.device,
        )
        z = torch.arange(
            self.scene_bbox[0][2],
            self.scene_bbox[1][2],
            self.cache_grid_size[2],
            device=self.device,
        )
        Z, Y, X = torch.meshgrid(x, y, z)
        coord = torch.cat([X[..., None], Y[..., None], Z[..., None]], dim=-1)
        return coord + self.cache_grid_size / 2.0

    def build(self, points, kdtree):
        H, W, D = points.shape[:3]
        points = points.view(H * W * D, 3)

        index_size = kdtree.max_depth / 3.
        index_size = math.pow(2, index_size) - 1.
        index_empty = -torch.ones((1, 3), device=self.device) * index_size * 2.0
        index_list = []
        for i in tqdm(range(points.shape[0])):
            point = points[i]
            index = kdtree.get_kd_index(point, index_empty)
            index_list.append(index.view(3, 1))

        index_list = torch.stack(index_list, dim=-1).squeeze()
        index_list = index_list.reshape(3, H, W, D) / index_size
        return index_list.to(torch.float16)


class KDTree:
    def __init__(
        self, points, confidence, max_depth=24, max_sub_points=1, code_mode="xyzxyz"
    ):
        INDEX_MODE = code_mode
        self.device = points.device
        self.code_mode = code_mode
        self.max_depth = max_depth
        self.max_sub_points = max_sub_points
        self.confidence_thre = 0.5
        self.knn_outlier_thre = 0.05  # for prune
        self.root = self.build(points, confidence, 0)

    def get_longest_axis(self, points):
        bbox_len = (
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        )
        return torch.max(torch.tensor(bbox_len), dim=0)[1]

    def build(self, points, confs, depth, pre_index=None):
        is_leaf = False
        if len(points) == 0 and depth > self.max_depth:
            return None
        if len(points) <= self.max_sub_points or depth == self.max_depth:
            is_leaf = True

        if pre_index is None:
            pre_index = "0" * self.max_depth

        axis = self.get_longest_axis(points)
        sort_index = points[:, axis].argsort()
        points = points[sort_index]

        median_idx = len(points) // 2
        axis_mid = points[median_idx][axis]
        axis_left = points[0][axis]
        axis_right = points[-1][axis]
        left_pre_index = pre_index
        right_pre_index = pre_index[:depth] + "1" + pre_index[depth + 1 :]

        if is_leaf:
            left_node = None
            right_node = None
            # conf = torch.mean(confs)
            conf = 0.5
        else:
            # confs = confs[sort_index]
            left_node = self.build(
                points[:median_idx], confs[:median_idx], depth + 1, left_pre_index
            )
            right_node = self.build(
                points[median_idx:], confs[median_idx:], depth + 1, right_pre_index
            )
            conf = (left_node.confidence + right_node.confidence) / 2

        return KDNode(
            axis=axis,
            axis_mid=axis_mid,
            axis_left=axis_left,
            axis_right=axis_right,
            left=left_node,
            right=right_node,
            layer=depth,
            index=pre_index,
            is_leaf=is_leaf,
            confidence=conf,
        )

    def get_kd_index(self, point, empty_index):
        def _find_node(node):
            if node is None:
                return empty_index

            if node.is_leaf:
                return node.index

            if point[node.axis] < node.axis_left:  # outside
                return empty_index
            elif point[node.axis] < node.axis_mid:  # left
                return _find_node(node.left)
            elif point[node.axis] < node.axis_right:  # right
                return _find_node(node.right)
            else:  # outside
                return empty_index

        return _find_node(self.root)


class KDVox:
    def __init__(
        self,
        points,
        confidence,
        max_depth=24,
        max_sub_points=1,
        code_mode="xyzxyz",
        box_margin_thre=0.01,
        k_search=8,
        layer_search=10,
    ):
        INDEX_MODE = code_mode
        self.device = points.device
        self.code_mode = code_mode
        self.max_depth = max_depth
        self.max_sub_points = max_sub_points
        self.box_margin_thre = box_margin_thre
        self.k_search = k_search
        self.layer_search = max(layer_search, 0)
        self.confidence_thre = 0.5
        self.knn_outlier_thre = 0.05  # for prune
        self.root = self.build(points, confidence, 0)

    def get_box(self, points):
        threshold = self.box_margin_thre
        if len(points) == 1:
            return points[0] - threshold, points[0] + threshold
        min_x = torch.min(points[:, 0])
        max_x = torch.max(points[:, 0])
        min_y = torch.min(points[:, 1])
        max_y = torch.max(points[:, 1])
        min_z = torch.min(points[:, 2])
        max_z = torch.max(points[:, 2])
        return torch.tensor([min_x, min_y, min_z], device=points.device), torch.tensor(
            [max_x, max_y, max_z], device=points.device
        )

    def build(self, points, confidence, depth, pre_index=None):
        is_leave = False
        if len(points) == 0 or depth > self.max_depth:
            return None
        if len(points) <= self.max_sub_points or depth == self.max_depth:
            is_leave = True

        if pre_index is None:
            pre_index = "0" * self.max_depth

        axis = depth % len(points[0])
        points = points[points[:, axis].argsort()]
        median_idx = len(points) // 2
        box_min, box_max = self.get_box(points)
        # pre_index[:depth] + '0' + pre_index[depth+1:]
        left_pre_index = pre_index
        right_pre_index = pre_index[:depth] + "1" + pre_index[depth + 1 :]

        if is_leave:
            left_node = None
            right_node = None
            conf = torch.mean(confidence)
        else:
            left_node = self.build(
                points[:median_idx], confidence[:median_idx], depth + 1, left_pre_index
            )
            right_node = self.build(
                points[median_idx:], confidence[median_idx:], depth + 1, right_pre_index
            )
            conf = (left_node.confidence + right_node.confidence) / 2
        return KDVoxNode(
            axis=axis,
            left=left_node,
            right=right_node,
            layer=depth,
            box_min=box_min,
            box_max=box_max,
            index=pre_index,
            is_leave=is_leave,
            confidence=conf,
        )

    def update_box(self, node, point):
        if point[0] > node.box_max[0]:
            node.box_max[0] = point[0]
        if point[0] < node.box_min[0]:
            node.box_min[0] = point[0]
        if point[1] > node.box_max[1]:
            node.box_max[1] = point[1]
        if point[1] < node.box_min[1]:
            node.box_min[1] = point[1]
        if point[2] > node.box_max[2]:
            node.box_max[2] = point[2]
        if point[2] < node.box_min[2]:
            node.box_min[2] = point[2]

    def add(self, points):
        def _add(node, point):
            box_min, box_max = self.get_box([point])
            dist_to_node = distance_point_to_box(point, node.box_min, node.box_max)
            dist_max = 3 * math.pow(self.box_margin_thre, 2)
            if node.is_leave and dist_to_node <= dist_max:
                if dist_to_node == 0:
                    return False
                else:
                    self.update_box(node, box_min)
                    self.update_box(node, box_max)
                    return True

            depth = node.layer

            if depth < self.max_depth:
                axis = (node.axis + 1) % 3
                left_pre_index = node.index[:depth] + "0" + node.index[depth + 1 :]
                right_pre_index = node.index[:depth] + "1" + node.index[depth + 1 :]

                if node.left is None and node.right is None:
                    node.left = KDVoxNode(
                        axis=axis,
                        left=None,
                        right=None,
                        layer=node.layer + 1,
                        box_min=node.box_min.copy(),
                        box_max=node.box_max.copy(),
                        index=left_pre_index,
                        is_leave=True,
                        confidence=node.confidence,
                    )
                    node.right = KDVoxNode(
                        axis=axis,
                        left=None,
                        right=None,
                        layer=node.layer + 1,
                        box_min=box_min,
                        box_max=box_max,
                        index=right_pre_index,
                        is_leave=True,
                        confidence=0.5,
                    )
                    node.is_leave = False
                    self.update_box(node, box_min)
                    self.update_box(node, box_max)
                    return True

                elif node.left is not None and node.right is None:
                    node.right = KDVoxNode(
                        axis=axis,
                        left=None,
                        right=None,
                        layer=node.layer + 1,
                        box_min=box_min,
                        box_max=box_max,
                        index=right_pre_index,
                        is_leave=True,
                        confidence=0.5,
                    )
                    node.is_leave = False
                    self.update_box(node, box_min)
                    self.update_box(node, box_max)
                    return True

                elif node.left is None and node.right is not None:
                    node.left = KDVoxNode(
                        axis=axis,
                        left=None,
                        right=None,
                        layer=node.layer + 1,
                        box_min=box_min,
                        box_max=box_max,
                        index=left_pre_index,
                        is_leave=True,
                        confidence=0.5,
                    )
                    node.is_leave = False
                    self.update_box(node, box_min)
                    self.update_box(node, box_max)
                    return True

            else:
                self.update_box(node, point)
                return True

            dist_left = distance_point_to_box(
                point, node.left.box_min, node.left.box_max
            )
            dist_right = distance_point_to_box(
                point, node.right.box_min, node.right.box_max
            )

            if dist_left < dist_right:
                if _add(node.left, point):
                    self.update_box(node, box_min)
                    self.update_box(node, box_max)
                    return True
                else:
                    return False
            else:
                if _add(node.right, point):
                    self.update_box(node, box_min)
                    self.update_box(node, box_max)
                    return True
                else:
                    return False

            raise ValueError("should not be here")

        sorted_points = sorted(
            points,
            key=lambda point: self.search_knn(point, k=1, dist_thre=torch.inf)[0][1],
            reverse=True,
        )
        for point in sorted_points:
            _add(self.root, point)

    def grow(self, points, grow_layer=None, grow_n_node_every_layer=1):
        def _get_node_of_layer(root, layer):
            if root.layer == layer:
                return [root]
            else:
                return _get_node_of_layer(root.left, layer) + _get_node_of_layer(
                    root.right, layer
                )

        if grow_layer is None:
            grow_layer = max(self.max_depth - 5, 0)
        node_grow_layer = _get_node_of_layer(self.root, grow_layer)

        add_count = 0
        for node in node_grow_layer:
            point_in_node = torch.tensor(
                [
                    distance_point_to_box(point, node.box_min, node.box_max) == 0
                    for point in points
                ]
            )
            self.add(points[point_in_node][:grow_n_node_every_layer])
            add_count += 1
            points = points[~point_in_node]

        self.add(points)
        add_count += len(points)
        print("grow {} points on layer {}".format(add_count, grow_layer))

    def prune(self, node, prune_mode=["confidence"], print_info=False):
        def _del_or_not(n):
            for m in prune_mode:
                assert m in [
                    "confidence",
                    "normal",
                    "knn_outlier",
                    "repeat",
                ], "prune mode should be one of confidence, normal, knn_outlier, repeat"

            if "confidence" in prune_mode and n.confidence < self.confidence_thre:
                return True
            if "normal" in prune_mode:
                raise NotImplementedError
            if "knn_outlier" in prune_mode:
                mid_point = (n.box_min + n.box_max) / 2
                nearest_dist = self.search_knn(mid_point, k=2, dist_thre=torch.inf)[1][
                    1
                ]
                if nearest_dist > self.knn_outlier_thre:
                    return True
            if "repeat" in prune_mode:
                mid_point = (n.box_min + n.box_max) / 2
                nearest_dist = self.search_knn(mid_point, k=2, dist_thre=torch.inf)[1][
                    1
                ]
                if nearest_dist == 0:
                    return True
            return False

        if node is None:
            return None

        node.left = self.prune(node.left, prune_mode, print_info)
        node.right = self.prune(node.right, prune_mode, print_info)
        node.is_leaf = (not node.left) and (not node.right)

        if _del_or_not(node) and node.is_leave:
            if print_info:
                print("delete node: ", node.index)
            return None

        return node

    def search_knn(self, point, root=None, k=1, dist_thre=0.5, layer=None):
        nearest_points = []
        visited = set()
        # dist_thre_inv = 1.0 / dist_thre

        def _find_nearest(node):
            if node is None:
                return
            visited.add(node)

            if node.is_leave or node.layer == layer:
                dist = distance_point_to_box(point, node.box_min, node.box_max)
                node.dist = dist
                if len(nearest_points) < k and dist <= dist_thre:
                    heapq.heappush(nearest_points, (-dist, node, node.index))
                elif len(nearest_points) == k and dist < -nearest_points[0][0]:
                    heapq.heappushpop(nearest_points, (-dist, node, node.index))
                return

            if node.right is not None:
                dist_right = distance_point_to_box(
                    point, node.right.box_min, node.right.box_max
                )
            else:
                dist_right = torch.inf
            if node.left is not None:
                dist_left = distance_point_to_box(
                    point, node.left.box_min, node.left.box_max
                )
            else:
                dist_left = torch.inf

            if dist_left < dist_right:
                if node.left not in visited:
                    _find_nearest(node.left)
                if node.right not in visited:
                    if len(nearest_points) < k and dist_right <= dist_thre:
                        _find_nearest(node.right)
                    elif (
                        len(nearest_points) == k and dist_right < -nearest_points[0][0]
                    ):
                        _find_nearest(node.right)
            else:
                if node.right not in visited:
                    _find_nearest(node.right)
                if node.left not in visited:
                    if len(nearest_points) < k and dist_left <= dist_thre:
                        _find_nearest(node.left)
                    elif len(nearest_points) == k and dist_left < -nearest_points[0][0]:
                        _find_nearest(node.left)

        _find_nearest(self.root if root is None else root)

        knn_list = heapq.nlargest(k, nearest_points)
        # in_thre_list = list(filter(lambda x: x[0] > dist_thre_inv, knn_list))
        if len(knn_list) == 0:
            return [torch.inf], None, None

        dist_list, node_list, index_list = zip(*knn_list)
        return dist_list, node_list, index_list

    def print(self, root, layer=None):
        if root is None:
            return
        if layer is None:
            if root.is_leave:
                print(
                    "axis, layer, box_min, box_max, index",
                    root.axis,
                    root.layer,
                    root.box_min,
                    root.box_max,
                    root.index,
                    self.code_mode,
                )
        else:
            if root.layer == layer:
                print(
                    "axis, layer, box_min, box_max, index",
                    root.axis,
                    root.layer,
                    root.box_min,
                    root.box_max,
                    root.index,
                    self.code_mode,
                )
        self.print(root.left, layer)
        self.print(root.right, layer)

    def draw_box(self, node, line_set_list, color=[0, 0, 0], layer=None):
        if node is None:
            return line_set_list
        if (layer is None and node.is_leave) or node.layer == layer:
            min_pos = node.box_min
            max_pos = node.box_max

            vertices = np.array(
                [
                    [min_pos[0], min_pos[1], min_pos[2]],  # 0
                    [min_pos[0], max_pos[1], min_pos[2]],  # 1
                    [max_pos[0], max_pos[1], min_pos[2]],  # 2
                    [max_pos[0], min_pos[1], min_pos[2]],  # 3
                    [min_pos[0], min_pos[1], max_pos[2]],  # 4
                    [min_pos[0], max_pos[1], max_pos[2]],  # 5
                    [max_pos[0], max_pos[1], max_pos[2]],  # 6
                    [max_pos[0], min_pos[1], max_pos[2]],
                ]
            )  # 7

            lines = np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],  # bottom
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],  # top
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ]
            )  # vertical

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(
                np.array([color for i in range(len(lines))])
            )
            line_set_list.append(line_set)
            return line_set_list
        else:
            line_set_list = self.draw_box(node.left, line_set_list, color, layer)
            line_set_list = self.draw_box(node.right, line_set_list, color, layer)
            return line_set_list

    def draw_tree(self, root, random_points=None, layer=None):
        geometry_list = []
        geometry_list = self.draw_box(root, geometry_list, layer=layer)

        if random_points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(random_points)
            geometry_list.append(pcd)

        return geometry_list


# ply_path = './../data/bunny.ply'

# ============= KD-TREE build =================
# n_points = 50
# random_points = np.random.rand(n_points, 3)
# random_confidence = np.random.rand(n_points, 1)
# max_layer = min(24, int(math.log2(n_points))+1)
# max_layer = math.ceil(max_layer/3) * 3
# start_time = time.time()
# kd_tree = KDTree(random_points, random_confidence, max_depth=max_layer, max_sub_points=2, code_mode='xxyyzz')
# end_time = time.time() - start_time
# print(f'build kd_tree of {n_points} points and {max_layer} layers consuming {end_time:.2f}s')
# # === vis ===
# geometry_list = kd_tree.draw_tree(kd_tree.root, random_points)
# o3d.visualization.draw_geometries(geometry_list)
# kd_tree.print(kd_tree.root)


# ================ KD-TREE prune ==============
# kd_tree.prune(kd_tree.root, ['knn_outlier', 'confidence'], True)
# # === vis ===
# geometry_list = kd_tree.draw_tree(kd_tree.root, random_points)
# o3d.visualization.draw_geometries(geometry_list)


# ================ KD-TREE add ================
# n_add = 50
# random_points_add = np.random.rand(n_add, 3)
# kd_tree.add(random_points_add)
# print('add random_points', random_points_add)
# random_points = np.concatenate((random_points, random_points_add), axis=0)
# === vis ===
# kd_tree.print(kd_tree.root)
# geometry_list = kd_tree.draw_tree(kd_tree.root, random_points)
# o3d.visualization.draw_geometries(geometry_list)


# ================ KD-TREE grow ================
# def generate_priority_points_queue(points, density, nearest_dist, pixel_loss):
#     dens_w = 0.2
#     dist_w = 0.2
#     loss_w = 0.6
#     priority = dens_w * density + dist_w * nearest_dist + loss_w * pixel_loss
#     idx = np.argsort(-priority[:, 0])
#     return points[idx]
#
# n_grow = 50
# random_points_add = np.random.rand(n_grow, 3)
# random_density = np.random.rand(n_grow, 1)
# random_dist = np.random.rand(n_grow, 1)
# random_loss = np.random.rand(n_grow, 1)
# points_to_grow = generate_priority_points_queue(random_points_add, random_density, random_dist, random_loss)
# kd_tree.grow(points_to_grow)
# # === vis ===
# geometry_list = kd_tree.draw_tree(kd_tree.root, random_points)
# o3d.visualization.draw_geometries(geometry_list)


# ================ KD-TREE prune ================
# kd_tree.prune(kd_tree.root, ['repeat'], True)
# # === vis ===
# geometry_list = kd_tree.draw_tree(kd_tree.root, random_points)
# o3d.visualization.draw_geometries(geometry_list)


# ================ KD-TREE search ===============
# n_search = 1
# search_points = np.random.rand(n_search, 3)
# start_time = time.time()
# knn = kd_tree.search_knn(search_points[0], k=8, dist_thre=2.0)
# end_time = time.time() - start_time
# print(f'search {n_search} points in kd_tree consuming {end_time:.2f}s')
# print('search_knn', len(knn), knn)
# # === vis ===
# if n_points <= 1000:
#     geometry_list = []
#     geometry_list = kd_tree.draw_tree(kd_tree.root, random_points)
#     for n in knn:
#         n[0].box_min -= 0.01
#         n[0].box_max += 0.01
#         geometry_list = kd_tree.draw_box(n[0], geometry_list, [1,0,0])
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(search_points)
#     pcd.colors = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
#
#     geometry_list.append(pcd)
#     o3d.visualization.draw_geometries(geometry_list)


# ================ KD-TREE vis ===============
# import pickle
# kd_tree_path = '/mnt/code/TensoIR_mac/scripts/kdtree.pkl'
# points_path = '/mnt/code/TensoIR_mac/scripts/step-0200-0.ply'
#
# with open(kd_tree_path, 'rb') as f:
#     kd_tree = pickle.load(f)
#
# pcd = o3d.io.read_point_cloud(points_path)
#
# # kd_tree.print(kd_tree.root)
#
# # geometry_list = kd_tree.draw_tree(kd_tree.root, pcd, layer=10)
# # o3d.visualization.draw_geometries(geometry_list)
#
# geometry_list = []
# knn = kd_tree.search_knn(pcd.points[0].numpy(), k=8, dist_thre=2.0)
# for n in knn:
#     n[0].box_min -= 0.01
#     n[0].box_max += 0.01
#     geometry_list = kd_tree.draw_box(n[0], geometry_list, [1,0,0])
# geometry_list.append(pcd)
# o3d.visualization.draw_geometries(geometry_list)


# ================ KD-CACHE vis ===============
import pickle, io

kd_cache_path = "/Users/minisal/DATA/KD-Tensor/debug/kdcache_grid=100_bbox=-1.5_1.5.pkl"
kd_points_path = "/Users/minisal/DATA/KD-Tensor/debug/layer=21_res=128.pkl"
pcd_path = "/Users/minisal/DATA/KD-Tensor/debug/step-0200-0.ply"


class CPU_pickle_load(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


VIS = False

if VIS:
    with open(kd_points_path, "rb") as f:
        kd_points = CPU_pickle_load(f).load()

    color_1 = kd_points["xyz_sampled"].numpy() / 2.0 + 0.5
    color_2 = kd_points["index"].numpy() / 2.0 + 0.5
    coord = kd_points["xyz_sampled"].numpy()
    # coord = np.stack([coord[:, 2], coord[:, 1], coord[:, 0]], axis=1).reshape(-1, 3)

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(coord)
    pcd_1.colors = o3d.utility.Vector3dVector(color_1)

    kd_valid = color_2[:, 0] != -2.0
    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(coord[kd_valid])
    pcd_2.colors = o3d.utility.Vector3dVector(color_2[kd_valid])

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.paint_uniform_color([0, 0, 0])

    o3d.visualization.draw_geometries([pcd_2, pcd])
