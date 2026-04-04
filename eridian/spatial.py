#!/usr/bin/env python3
"""
Spatial Hashing Module for Eridian
Efficient spatial partitioning and memory management for large point clouds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import threading
from collections import defaultdict

from .logging import get_logger, get_monitor, safe_execute, timer_context


class SpatialHash:
    """
    Spatial hash for efficient spatial queries and duplicate detection.
    Divides space into a grid of cells for O(1) nearest neighbor lookups.
    """
    
    def __init__(self, cell_size: float = 0.1):
        """
        Initialize spatial hash.
        
        Args:
            cell_size: Size of each grid cell in world units (meters)
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        
        # Hash grid: (cell_x, cell_y, cell_z) -> list of point indices
        self._grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        
        # Point data
        self._positions: List[np.ndarray] = []
        self._metadata: List[dict] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.debug(f"Spatial hash initialized (cell_size={cell_size})")
    
    def _get_cell_key(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Get grid cell key for a position."""
        x, y, z = position
        return (
            int(x * self.inv_cell_size),
            int(y * self.inv_cell_size),
            int(z * self.inv_cell_size)
        )
    
    def _get_neighbor_keys(self, position: np.ndarray, radius: float = 0.0) -> Set[Tuple[int, int, int]]:
        """Get cell keys within a radius of a position."""
        if radius <= 0:
            return {self._get_cell_key(position)}
        
        cx, cy, cz = self._get_cell_key(position)
        cells = int(np.ceil(radius * self.inv_cell_size))
        
        keys = set()
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                for dz in range(-cells, cells + 1):
                    keys.add((cx + dx, cy + dy, cz + dz))
        
        return keys
    
    def insert(self, position: np.ndarray, metadata: Optional[dict] = None) -> int:
        """
        Insert a point into the spatial hash.
        
        Args:
            position: 3D position (x, y, z)
            metadata: Optional metadata dictionary for the point
            
        Returns:
            Index of the inserted point
        """
        with self._lock:
            idx = len(self._positions)
            key = self._get_cell_key(position)
            
            self._positions.append(position.copy())
            self._metadata.append(metadata or {})
            self._grid[key].append(idx)
            
            self.monitor.increment_counter("points_inserted")
            return idx
    
    def insert_batch(self, positions: np.ndarray, 
                    metadata_list: Optional[List[dict]] = None) -> List[int]:
        """
        Insert multiple points efficiently.
        
        Args:
            positions: Array of positions (N, 3)
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of inserted indices
        """
        with self._lock:
            indices = []
            start_idx = len(self._positions)
            
            for i, pos in enumerate(positions):
                key = self._get_cell_key(pos)
                idx = start_idx + i
                self._grid[key].append(idx)
                indices.append(idx)
            
            self._positions.extend(positions)
            
            if metadata_list:
                self._metadata.extend(metadata_list)
            else:
                self._metadata.extend([{}] * len(positions))
            
            self.monitor.increment_counter("points_inserted", len(positions))
            return indices
    
    def query_radius(self, position: np.ndarray, radius: float) -> List[int]:
        """
        Query all points within a radius of a position.
        
        Args:
            position: Query position (x, y, z)
            radius: Search radius in world units
            
        Returns:
            List of point indices within the radius
        """
        with self._lock:
            keys = self._get_neighbor_keys(position, radius)
            radius_sq = radius * radius
            
            results = []
            for key in keys:
                for idx in self._grid.get(key, []):
                    pos = self._positions[idx]
                    dist_sq = np.sum((pos - position) ** 2)
                    if dist_sq <= radius_sq:
                        results.append(idx)
            
            return results
    
    def find_duplicates(self, position: np.ndarray, threshold: float = 0.02) -> List[int]:
        """
        Find duplicate points within a threshold distance.
        
        Args:
            position: Query position
            threshold: Distance threshold for duplicate detection
            
        Returns:
            List of duplicate point indices
        """
        duplicates = self.query_radius(position, threshold)
        return [idx for idx in duplicates if idx != len(self._positions) - 1]
    
    def get_point(self, idx: int) -> Tuple[np.ndarray, dict]:
        """
        Get a point and its metadata by index.
        
        Args:
            idx: Point index
            
        Returns:
            Tuple of (position, metadata)
        """
        with self._lock:
            if 0 <= idx < len(self._positions):
                return self._positions[idx].copy(), self._metadata[idx].copy()
            else:
                raise IndexError(f"Point index {idx} out of range")
    
    def get_all_positions(self) -> np.ndarray:
        """Get all positions as a numpy array."""
        with self._lock:
            if not self._positions:
                return np.zeros((0, 3))
            return np.array(self._positions)
    
    def get_all_metadata(self) -> List[dict]:
        """Get all metadata."""
        with self._lock:
            return [m.copy() for m in self._metadata]
    
    def remove_points(self, indices: List[int]):
        """
        Remove points by indices (Note: this is O(n) operation).
        
        Args:
            indices: List of indices to remove
        """
        with self._lock:
            if not indices:
                return
            
            # Create set of indices to remove
            to_remove = set(indices)
            
            # Filter positions and metadata
            new_positions = []
            new_metadata = []
            index_map = {}  # Maps old indices to new indices
            
            for i, (pos, meta) in enumerate(zip(self._positions, self._metadata)):
                if i not in to_remove:
                    new_idx = len(new_positions)
                    new_positions.append(pos)
                    new_metadata.append(meta)
                    index_map[i] = new_idx
            
            # Rebuild grid
            self._grid.clear()
            for idx, pos in enumerate(new_positions):
                key = self._get_cell_key(pos)
                self._grid[key].append(idx)
            
            self._positions = new_positions
            self._metadata = new_metadata
            
            self.monitor.increment_counter("points_removed", len(indices))
    
    def clear(self):
        """Clear all points from the spatial hash."""
        with self._lock:
            self._grid.clear()
            self._positions.clear()
            self._metadata.clear()
            self.logger.debug("Spatial hash cleared")
    
    def get_stats(self) -> dict:
        """Get statistics about the spatial hash."""
        with self._lock:
            occupied_cells = len(self._grid)
            total_cells = occupied_cells  # Approximation
            
            return {
                'num_points': len(self._positions),
                'occupied_cells': occupied_cells,
                'avg_points_per_cell': len(self._positions) / max(occupied_cells, 1),
                'cell_size': self.cell_size,
            }


class OctreeNode:
    """Node in an octree for hierarchical spatial partitioning."""
    
    def __init__(self, center: np.ndarray, size: float, depth: int = 0, max_depth: int = 8):
        self.center = center
        self.size = size
        self.depth = depth
        self.max_depth = max_depth
        
        # Point storage
        self.positions: List[np.ndarray] = []
        self.colors: Optional[List[np.ndarray]] = None
        self.indices: List[int] = []
        
        # Children
        self.children: Optional[List['OctreeNode']] = None
        
        # Statistics
        self.point_count = 0
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.children is None
    
    def subdivides(self) -> bool:
        """Check if this node has been subdivided."""
        return self.children is not None
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of this node."""
        half_size = self.size / 2.0
        min_bound = self.center - half_size
        max_bound = self.center + half_size
        return min_bound, max_bound


class Octree:
    """
    Octree for hierarchical spatial partitioning and LOD management.
    Enables efficient querying and level-of-detail rendering.
    """
    
    def __init__(self, center: np.ndarray = np.zeros(3), 
                 size: float = 10.0,
                 max_points_per_node: int = 1000,
                 max_depth: int = 8):
        """
        Initialize octree.
        
        Args:
            center: Center of the root node
            size: Size of the root node (cube)
            max_points_per_node: Maximum points before subdivision
            max_depth: Maximum tree depth
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        
        self.root = OctreeNode(center, size, 0, max_depth)
        self.max_points_per_node = max_points_per_node
        self.max_depth = max_depth
        
        self.total_points = 0
        self._lock = threading.RLock()
        
        self.logger.debug(f"Octree initialized (size={size}, max_depth={max_depth})")
    
    def insert(self, position: np.ndarray, color: Optional[np.ndarray] = None,
               index: Optional[int] = None) -> bool:
        """
        Insert a point into the octree.
        
        Args:
            position: 3D position
            color: Optional color
            index: Optional point index
            
        Returns:
            True if insertion successful
        """
        with self._lock:
            return self._insert_recursive(self.root, position, color, index)
    
    def _insert_recursive(self, node: OctreeNode, position: np.ndarray,
                         color: Optional[np.ndarray], index: Optional[int]) -> bool:
        """Recursively insert a point."""
        # Check if point is within node bounds
        min_bound, max_bound = node.get_bounds()
        if not (np.all(position >= min_bound) and np.all(position <= max_bound)):
            return False
        
        # If leaf node and has space, add point
        if node.is_leaf():
            if node.point_count < self.max_points_per_node or node.depth >= self.max_depth:
                node.positions.append(position.copy())
                if color is not None:
                    if node.colors is None:
                        node.colors = []
                    node.colors.append(color.copy())
                node.indices.append(index if index is not None len(node.positions))
                node.point_count += 1
                self.total_points += 1
                self.monitor.increment_counter("points_inserted")
                return True
            
            # Subdivide if full
            self._subdivide(node)
        
        # Insert into appropriate child
        for child in node.children:
            if self._insert_recursive(child, position, color, index):
                return True
        
        return False
    
    def _subdivide(self, node: OctreeNode):
        """Subdivide a node into 8 children."""
        if node.depth >= self.max_depth:
            return
        
        half_size = node.size / 2.0
        quarter_size = node.size / 4.0
        
        # Create 8 children
        node.children = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    child_center = node.center + np.array([dx, dy, dz]) * quarter_size
                    child = OctreeNode(child_center, half_size, node.depth + 1, self.max_depth)
                    node.children.append(child)
        
        # Redistribute existing points
        for i, pos in enumerate(node.positions):
            color = node.colors[i] if node.colors else None
            for child in node.children:
                if self._insert_recursive(child, pos, color, node.indices[i]):
                    break
        
        node.positions.clear()
        if node.colors:
            node.colors.clear()
        node.indices.clear()
    
    def query_radius(self, position: np.ndarray, radius: float) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Query all points within a radius.
        
        Args:
            position: Query position
            radius: Search radius
            
        Returns:
            List of (position, color) tuples
        """
        with self._lock:
            results = []
            self._query_radius_recursive(self.root, position, radius, results)
            return results
    
    def _query_radius_recursive(self, node: OctreeNode, position: np.ndarray,
                               radius: float, results: list):
        """Recursively query points within radius."""
        # Check if node bounding sphere intersects query sphere
        min_bound, max_bound = node.get_bounds()
        closest_point = np.clip(position, min_bound, max_bound)
        dist_sq = np.sum((closest_point - position) ** 2)
        
        if dist_sq > radius * radius:
            return
        
        # If leaf node, check points
        if node.is_leaf():
            radius_sq = radius * radius
            for i, pos in enumerate(node.positions):
                dist_sq = np.sum((pos - position) ** 2)
                if dist_sq <= radius_sq:
                    color = node.colors[i] if node.colors else None
                    results.append((pos.copy(), color.copy() if color is not None else None))
            return
        
        # Otherwise, query children
        for child in node.children:
            self._query_radius_recursive(child, position, radius, results)
    
    def get_all_points(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get all points in the octree.
        
        Returns:
            Tuple of (positions, colors) or (positions, None)
        """
        with self._lock:
            positions = []
            colors = []
            
            self._collect_recursive(self.root, positions, colors)
            
            if not positions:
                return np.zeros((0, 3)), None
            
            pos_array = np.array(positions)
            col_array = np.array(colors) if colors else None
            
            return pos_array, col_array
    
    def _collect_recursive(self, node: OctreeNode, positions: list, colors: list):
        """Recursively collect all points."""
        if node.is_leaf():
            positions.extend(node.positions)
            if node.colors:
                colors.extend(node.colors)
        else:
            for child in node.children:
                self._collect_recursive(child, positions, colors)
    
    def clear(self):
        """Clear all points from the octree."""
        with self._lock:
            self.root = OctreeNode(self.root.center, self.root.size, 0, self.max_depth)
            self.total_points = 0
            self.logger.debug("Octree cleared")
    
    def get_stats(self) -> dict:
        """Get octree statistics."""
        with self._lock:
            node_count = self._count_nodes(self.root)
            max_depth_reached = self._get_max_depth(self.root)
            
            return {
                'total_points': self.total_points,
                'node_count': node_count,
                'max_depth': max_depth_reached,
                'avg_points_per_leaf': self.total_points / max(node_count, 1),
            }
    
    def _count_nodes(self, node: OctreeNode) -> int:
        """Count nodes recursively."""
        if node.is_leaf():
            return 1
        return 1 + sum(self._count_nodes(child) for child in node.children)
    
    def _get_max_depth(self, node: OctreeNode) -> int:
        """Get maximum depth recursively."""
        if node.is_leaf():
            return node.depth
        return max(self._get_max_depth(child) for child in node.children)


if __name__ == "__main__":
    # Test spatial hashing and octree
    from .logging import setup_logging
    
    logger = setup_logging(level="INFO")
    
    # Test spatial hash
    print("Testing SpatialHash...")
    sh = SpatialHash(cell_size=0.5)
    
    # Insert random points
    positions = np.random.randn(100, 3) * 5.0
    sh.insert_batch(positions)
    
    # Query
    query_pos = np.array([0.0, 0.0, 0.0])
    neighbors = sh.query_radius(query_pos, 2.0)
    print(f"Found {len(neighbors)} neighbors within 2.0m")
    print(f"Stats: {sh.get_stats()}")
    
    # Test octree
    print("\nTesting Octree...")
    octree = Octree(size=10.0, max_points_per_node=10)
    
    # Insert points
    for i, pos in enumerate(positions[:50]):
        color = np.random.rand(3)
        octree.insert(pos, color, index=i)
    
    # Query
    results = octree.query_radius(query_pos, 2.0)
    print(f"Found {len(results)} neighbors within 2.0m")
    print(f"Stats: {octree.get_stats()}")
    
    # Get all points
    all_pos, all_col = octree.get_all_points()
    print(f"Retrieved {len(all_pos)} points")
    
    print("Spatial hashing module test completed ✓")