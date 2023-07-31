# Modified DAVIS interactive robot https://github.com/albertomontesg/davis-interactive
# I tamed their robot to work as a slave for me


import time
import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.special import comb
from skimage.filters import rank
from skimage.morphology import dilation, disk, erosion, medial_axis
from sklearn.neighbors import radius_neighbors_graph
import cv2


def bezier_curve(points, nb_points=1000):
    """ Given a list of points compute a bezier curve from it.

    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the
            number of points and the second dimension representing the
            (x, y) coordinates.
        nb_points: Integer. Number of points to sample from the bezier curve.
            This value must be larger than the number of points given in
            `points`. Maximum value 10000.

    # Returns
        ndarray: Array of shape (1000, 2) with the bezier curve of the
            given path of points.

    """
    nb_points = min(nb_points, 1000)

    points = np.asarray(points, dtype=np.float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            '`points` should be two dimensional and have shape: (N, 2)')

    n_points = len(points)
    if n_points > nb_points:
        # We are downsampling points
        return points

    t = np.linspace(0., 1., nb_points).reshape(1, -1)

    # Compute the Bernstein polynomial of n, i as a function of t
    i = np.arange(n_points).reshape(-1, 1)
    n = n_points - 1
    polynomial_array = comb(n, i) * (t**(n - i)) * (1 - t)**i

    bezier_curve_points = polynomial_array.T.dot(points)

    return bezier_curve_points


def bresenham(points):
    """ Apply Bresenham algorithm for a list points.

    More info: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm

    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the number
            if points and the second coordinate representing the (x, y)
            coordinates.

    # Returns
        ndarray: Array of points after having applied the bresenham algorithm.
    """

    points = np.asarray(points, dtype=np.int)

    def line(x0, y0, x1, y1):
        """ Bresenham line algorithm.
        """
        d_x = x1 - x0
        d_y = y1 - y0

        x_sign = 1 if d_x > 0 else -1
        y_sign = 1 if d_y > 0 else -1

        d_x = np.abs(d_x)
        d_y = np.abs(d_y)

        if d_x > d_y:
            xx, xy, yx, yy = x_sign, 0, 0, y_sign
        else:
            d_x, d_y = d_y, d_x
            xx, xy, yx, yy = 0, y_sign, x_sign, 0

        D = 2 * d_y - d_x
        y = 0

        line = np.empty((d_x + 1, 2), dtype=points.dtype)
        for x in range(d_x + 1):
            line[x] = [x0 + x * xx + y * yx, y0 + x * xy + y * yy]
            if D >= 0:
                y += 1
                D -= 2 * d_x
            D += 2 * d_y

        return line

    nb_points = len(points)
    if nb_points < 2:
        return points

    new_points = []

    for i in range(nb_points - 1):
        p = points[i:i + 2].ravel().tolist()
        new_points.append(line(*p))

    new_points = np.concatenate(new_points, axis=0)

    return new_points


class TamedRobot(object):

    def __init__(self,
                 kernel_size=.15,
                 max_kernel_radius=16,
                 min_nb_nodes=4,
                 nb_points=1000):

        self.kernel_size = kernel_size
        self.max_kernel_radius = max_kernel_radius
        self.min_nb_nodes = min_nb_nodes
        self.nb_points = nb_points

    def _generate_scribble_mask(self, mask):
        """ Generate the skeleton from a mask
        Given an error mask, the medial axis is computed to obtain the
        skeleton of the objects. In order to obtain smoother skeleton and
        remove small objects, an erosion and dilation operations are performed.
        The kernel size used is proportional the squared of the area.

        # Arguments
            mask: Numpy Array. Error mask

        Returns:
            skel: Numpy Array. Skeleton mask
        """
        mask = np.asarray(mask, dtype=np.uint8)
        side = np.sqrt(np.sum(mask > 0))

        mask_ = mask
        # kernel_size = int(self.kernel_size * side)
        kernel_radius = self.kernel_size * side * .5
        kernel_radius = min(kernel_radius, self.max_kernel_radius)

        compute = True
        while kernel_radius > 1. and compute:
            kernel = disk(kernel_radius)
            mask_ = rank.minimum(mask.copy(), kernel)
            mask_ = rank.maximum(mask_, kernel)
            compute = False
            if mask_.astype(np.bool).sum() == 0:
                compute = True
                prev_kernel_radius = kernel_radius
                kernel_radius *= .9

        mask_ = np.pad(
            mask_, ((1, 1), (1, 1)), mode='constant', constant_values=False)
        skel = medial_axis(mask_.astype(np.bool))
        skel = skel[1:-1, 1:-1]
        return skel

    def _mask2graph(self, skeleton_mask):
        """ Transforms a skeleton mask into a graph

        Args:
            skeleton_mask (ndarray): Skeleton mask

        Returns:
            tuple(nx.Graph, ndarray): Returns a tuple where the first element
                is a Graph and the second element is an array of xy coordinates
                indicating the coordinates for each Graph node.

                If an empty mask is given, None is returned.
        """
        mask = np.asarray(skeleton_mask, dtype=np.bool)
        if np.sum(mask) == 0:
            return None

        h, w = mask.shape
        x, y = np.arange(w), np.arange(h)
        X, Y = np.meshgrid(x, y)

        X, Y = X.ravel(), Y.ravel()
        M = mask.ravel()

        X, Y = X[M], Y[M]
        points = np.c_[X, Y]
        G = radius_neighbors_graph(points, np.sqrt(2), mode='distance')
        T = nx.from_scipy_sparse_matrix(G)

        return T, points

    def _acyclics_subgraphs(self, G):
        """ Divide a graph into connected components subgraphs
        Divide a graph into connected components subgraphs and remove its
        cycles removing the edge with higher weight inside the cycle. Also
        prune the graphs by number of nodes in case the graph has not enought
        nodes.

        Args:
            G (nx.Graph): Graph

        Returns:
            list(nx.Graph): Returns a list of graphs which are subgraphs of G
                with cycles removed.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        S = []  # List of subgraphs of G

        for c in nx.connected_components(G):
            g = G.subgraph(c).copy()

            # Remove all cycles that we may find
            has_cycles = True
            while has_cycles:
                try:
                    cycle = nx.find_cycle(g)
                    weights = np.asarray([G[u][v]['weight'] for u, v in cycle])
                    idx = weights.argmax()
                    # Remove the edge with highest weight at cycle
                    g.remove_edge(*cycle[idx])
                except nx.NetworkXNoCycle:
                    has_cycles = False

            S.append(g)

        return S

    def _longest_path_in_tree(self, G):
        """ Given a tree graph, compute the longest path and return it
        Given an undirected tree graph, compute the longest path and return it.

        The approach use two shortest path transversals (shortest path in a
        tree is the same as longest path). This could be improve but would
        require implement it:
        https://cs.stackexchange.com/questions/11263/longest-path-in-an-undirected-tree-with-only-one-traversal

        Args:
            G (nx.Graph): Graph which should be an undirected tree graph

        Returns:
            list(int): Returns a list of indexes of the nodes belonging to the
                longest path.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        if not nx.is_tree(G):
            raise ValueError('Graph G must be a tree (graph without cycles)')

        # Compute the furthest node to the random node v
        v = list(G.nodes())[0]
        distance = nx.single_source_shortest_path_length(G, v)
        vp = max(distance.items(), key=lambda x: x[1])[0]
        # From this furthest point v' find again the longest path from it
        distance = nx.single_source_shortest_path(G, vp)
        longest_path = max(distance.values(), key=len)
        # Return the longest path

        return list(longest_path)

    def interact(self, error_mask):
        # start_time = time.time()
        out_mask = np.zeros_like(error_mask)

        # Downsample for faster processing
        error_mask = (cv2.resize(error_mask, dsize=None, fx=(0.5), fy=(0.5), interpolation=cv2.INTER_AREA)>0.5).astype(np.uint8)

        # Generate scribbles
        skel_mask = self._generate_scribble_mask(error_mask)
        # skel_time = time.time() - start_time
        # print('skel', skel_time)

        graph_out = self._mask2graph(skel_mask)
        if graph_out is None:
            return out_mask
        G, P = graph_out
        # mask2graph_time = time.time() - start_time - skel_time
        # print('m2g time', mask2graph_time)

        # t_start = time.time()
        S = self._acyclics_subgraphs(G)
        # t = (time.time() - t_start)
        # print('asub time', t)

        # t_start = time.time()
        longest_paths_idx = [self._longest_path_in_tree(s) for s in S]
        longest_paths = [P[idx] for idx in longest_paths_idx]
        # t = (time.time() - t_start) 
        # print('longest time', t)

        # t_start = time.time()
        scribbles_paths = [
            bezier_curve(p, self.nb_points) for p in longest_paths
        ]
        # t = (time.time() - t_start) 
        # print('asub time', t)

        # t_start = time.time()
        for path in scribbles_paths:
            # Re-upsample the line
            path = bresenham(path*2)
            out_mask[path[:, 1], path[:, 0]] = 1
        # t = (time.time() - t_start) 
        # print('final time', t)

        return out_mask

