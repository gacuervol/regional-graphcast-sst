# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for creating icosahedral meshes."""

import itertools
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.spatial import transform

# debug
import jax
def mokey():
  return var1

# Funciones anadidas
from . import model_utils_newvars
from itertools import product
import re
from typing import Iterator
import xarray


def get_custommesh_nodes_coords(lat: xarray.DataArray, lon: xarray.DataArray, divisions: int) -> Iterator[tuple[float, float]]:
    """
    Genera las coordenadas de los nodos de una malla personalizada basada en datos de latitud y longitud.

    Args:
        lat (xr.DataArray): Un DataArray de xarray que contiene las coordenadas de latitud.
        lon (xr.DataArray): Un DataArray de xarray que contiene las coordenadas de longitud.
        divisions (int): El número de divisiones deseadas para la malla en la dirección de la latitud.

    Returns:
        Iterator[Tuple[float, float]]: Un iterador que produce tuplas de coordenadas (latitud, longitud) para cada nodo en la malla generada.

    Example:
        >>> import xarray as xr
        >>> lat = xr.DataArray([0, 1, 2], dims="lat")
        >>> lon = xr.DataArray([0, 1, 2, 3], dims="lon")
        >>> divisions = 2
        >>> coords = get_custommesh_nodes_coords(lat, lon, divisions)
        >>> list(coords)
        [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0)]
    """
    # m_divisions = divisions + 1
    # d = (lat.data[-1] - lat.data[0]) / m_divisions
    # n = int(np.floor((lon.data[-1] - lon.data[0]) / d))
    # lon_mesh = np.linspace(lon[0].data, lon[0].data + (n * d), n)
    # lat_mesh = np.linspace(lat[0].data, lat[0].data + (m_divisions * d), m_divisions)
    divisions = divisions + 1

    lon_mesh = np.linspace(lon[0].data, lon[-1].data, divisions)#lon[0].data + (n * d), n)
    lat_mesh = np.linspace(lat[0].data, lat[-1].data, divisions) #lat[0].data + (m_divisions * d), m_divisions)

    return iter(product(lat_mesh, lon_mesh))


def get_custommesh_nodes_xyz(lat: xarray.DataArray, lon: xarray.DataArray, divisions: int) -> Iterator[tuple]:
    """
    Genera las coordenadas espaciales (x, y, z) de los nodos de una malla personalizada basada en datos de latitud y longitud.

    Args:
        lat (xr.DataArray): Un DataArray de xarray que contiene las coordenadas de latitud.
        lon (xr.DataArray): Un DataArray de xarray que contiene las coordenadas de longitud.
        divisions (int): El número de divisiones deseadas para la malla en la dirección de la latitud.

    Returns:
        Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]: Un iterador que produce tuplas de coordenadas (x, y, z) 
        para cada nodo en la malla generada en el espacio cartesiano.

    Example:
        >>> import xarray as xr
        >>> import model_utils
        >>> lat = xr.DataArray([0, 1, 2], dims="lat")
        >>> lon = xr.DataArray([0, 1, 2, 3], dims="lon")
        >>> divisions = 2
        >>> coords_xyz = get_custommesh_nodes_xyz(lat, lon, divisions)
        >>> for x, y, z in coords_xyz:
        ...     print(f"x: {x}, y: {y}, z: {z}")
    """
    gen_custommesh_coords = get_custommesh_nodes_coords(lat, lon, divisions)
    
    def generate_xyz(mesh_nodes_coords):
        for lat, lon in mesh_nodes_coords:
            phi_list, theta_list = model_utils_newvars.lat_lon_deg_to_spherical(np.array(lat), 
                                                                        np.array(lon))
            x_list, y_list, z_list = model_utils_newvars.spherical_to_cartesian(phi_list, theta_list)
            yield x_list, y_list, z_list
    
    return generate_xyz(gen_custommesh_coords)


def generate_triangles(lat: xarray.DataArray, lon: xarray.DataArray, divisions: int):
    """
    Genera las caras de los triángulos de una malla personalizada basada en datos de latitud y longitud.

    Args:
        lat (xr.DataArray): Un DataArray de xarray que contiene las coordenadas de latitud.
        lon (xr.DataArray): Un DataArray de xarray que contiene las coordenadas de longitud.
        divisions (int): El número de divisiones deseadas para la malla en la dirección de la latitud.

    Returns:
        Iterator[Tuple[int, int, int]]: Un iterador que produce tuplas de tres enteros, 
        representando los índices de los nodos que forman cada cara de los triángulos en la malla.

    Example:
        >>> import xarray as xr
        >>> lat = xr.DataArray([0, 1, 2], dims="lat")
        >>> lon = xr.DataArray([0, 1, 2, 3], dims="lon")
        >>> divisions = 2
        >>> faces = generate_triangles(lat, lon, divisions)
        >>> list(faces)
        [(0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4), (3, 4, 6), (4, 7, 6), (4, 5, 7), (5, 8, 7)]
    """
    gen_custommesh_xyz = get_custommesh_nodes_xyz(lat, lon, divisions)
    nodes = list(gen_custommesh_xyz)
    n = len(nodes)
    m_divisions = divisions + 1
    
    def get_index(row, col):
        return row * m_divisions + col
    def generate_faces(m_divisions):
        for row in range(m_divisions - 1):
            for col in range(m_divisions - 1):
                i = get_index(row, col)
                i_right = get_index(row, col + 1)
                i_down = get_index(row + 1, col)
                i_down_right = get_index(row + 1, col + 1)
                
                yield (i, i_right, i_down)
                yield (i_right, i_down_right, i_down)
                
    return generate_faces(m_divisions)


class TriangularMesh(NamedTuple):
  """Data structure for triangular meshes.

  Attributes:
    vertices: spatial positions of the vertices of the mesh of shape
        [num_vertices, num_dims].
    faces: triangular faces of the mesh of shape [num_faces, 3]. Contains
        integer indices into `vertices`.

  """
  vertices: np.ndarray
  faces: np.ndarray


def merge_meshes(
    mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
  """Merges all meshes into one. Assumes the last mesh is the finest.

  Args:
     mesh_list: Sequence of meshes, from coarse to fine refinement levels. The
       vertices and faces may contain those from preceding, coarser levels.

  Returns:
     `TriangularMesh` for which the vertices correspond to the highest
     resolution mesh in the hierarchy, and the faces are the join set of the
     faces at all levels of the hierarchy.
  """
  for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
    num_nodes_mesh_i = mesh_i.vertices.shape[0]
    assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

  return TriangularMesh(
      vertices=mesh_list[-1].vertices,
      faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0))


def get_hierarchy_of_triangular_meshes_for_sphere(
    splits: int,
    lat_data: List[float],
    lon_data: List[float],
    divisions: int) -> List[TriangularMesh]:
  """Returns a sequence of meshes, each with triangularization sphere.

  Starting with a regular icosahedron (12 vertices, 20 faces, 30 edges) with
  circumscribed unit sphere. Then, each triangular face is iteratively
  subdivided into 4 triangular faces `splits` times. The new vertices are then
  projected back onto the unit sphere. All resulting meshes are returned in a
  list, from lowest to highest resolution.

  The vertices in each face are specified in counter-clockwise order as
  observed from the outside the icosahedron.

  Args:
     splits: How many times to split each triangle.
  Returns:
     Sequence of `TriangularMesh`s of length `splits + 1` each with:

       vertices: [num_vertices, 3] vertex positions in 3D, all with unit norm.
       faces: [num_faces, 3] with triangular faces joining sets of 3 vertices.
           Each row contains three indices into the vertices array, indicating
           the vertices adjacent to the face. Always with positive orientation
           (counterclock-wise when looking from the outside).
  """
  current_mesh = get_icosahedron(lat_data, lon_data, divisions)
  output_meshes = [current_mesh]
  for _ in range(splits):
    current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
    output_meshes.append(current_mesh)
  return output_meshes

# def get_icosahedron() -> TriangularMesh:
def get_icosahedron(lat: xarray.DataArray, lon: xarray.DataArray, divisions: int) -> TriangularMesh:
    # Resolution lat: 181 lon: 360
    # vertices = [(0.68121426, -0.46382228, 0.56640624),
    #             (0.82158568, -0.0646602, 0.56640624),
    #             (0.83961024, -0.29526172, 0.45593329),
    #             (0.77796645, -0.5296985, 0.33791672),
    #             (0.93827468, -0.07384382, 0.33791672),]
    # Resolution lat: 301 lon: 329
    # vertices = [(0.76908083, -0.29406798, 0.56748453),
    #             (0.82081757, -0.06496012, 0.56748453),
    #             (0.86854167, -0.19613258, 0.45515644),
    #             (0.8800642, -0.33650391, 0.33504049),
    #             (0.93926689, -0.07433429, 0.33504049),]
    gen_customesh_xyz = get_custommesh_nodes_xyz(lat, lon, divisions)
    vertices = list(gen_customesh_xyz)

# phi_list, theta_list = model_utils.lat_lon_deg_to_spherical(np.array(lat_node_list), np.array(lon_node_list))
# x_list, y_list, z_list = model_utils.spherical_to_cartesian(phi_list, theta_list)
                                                            
    vertices = np.array(vertices, dtype=np.float32)

    # I did this manually, checking the orientation one by one.
    # faces = [(0, 2, 1),
    #          (0, 3, 2),
    #          (2, 3, 4),
    #          (1, 2, 4),
    #          ]
    gen_faces = generate_triangles(lat, lon, divisions=divisions)
    faces = list(gen_faces)
    
    return TriangularMesh(vertices=vertices.astype(np.float32),
                          faces=np.array(faces, dtype=np.int32))


#def get_icosahedron2() -> TriangularMesh:
#  """Returns a regular icosahedral mesh with circumscribed unit sphere.
#
#  See https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
#  for details on the construction of the regular icosahedron.
#
#  The vertices in each face are specified in counter-clockwise order as observed
#  from the outside of the icosahedron.
#
#  Returns:
#     TriangularMesh with:
#
#     vertices: [num_vertices=12, 3] vertex positions in 3D, all with unit norm.
#     faces: [num_faces=20, 3] with triangular faces joining sets of 3 vertices.
#         Each row contains three indices into the vertices array, indicating
#         the vertices adjacent to the face. Always with positive orientation (
#         counterclock-wise when looking from the outside).
#
#  """
#  phi = (1 + np.sqrt(5)) / 2
#  vertices = []
#  for c1 in [1., -1.]:
#    for c2 in [phi, -phi]:
#      vertices.append((c1, c2, 0.))
#      vertices.append((0., c1, c2))
#      vertices.append((c2, 0., c1))
#
#  vertices = np.array(vertices, dtype=np.float32)
#  vertices /= np.linalg.norm([1., phi])
#
#  # I did this manually, checking the orientation one by one.
#  faces = [(0, 1, 2),
#           (0, 6, 1),
#           (8, 0, 2),
#           (8, 4, 0),
#           (3, 8, 2),
#           (3, 2, 7),
#           (7, 2, 1),
#           (0, 4, 6),
#           (4, 11, 6),
#           (6, 11, 5),
#           (1, 5, 7),
#           (4, 10, 11),
#           (4, 8, 10),
#           (10, 8, 3),
#           (10, 3, 9),
#           (11, 10, 9),
#           (11, 9, 5),
#           (5, 9, 7),
#           (9, 3, 7),
#           (1, 6, 5),
#           ]
#
#  # By default the top is an aris parallel to the Y axis.
#  # Need to rotate around the y axis by half the supplementary to the
#  # angle between faces divided by two to get the desired orientation.
#  #                          /O\  (top arist)
#  #                     /          \                           Z
#  # (adjacent face)/                    \  (adjacent face)     ^
#  #           /     angle_between_faces      \                 |
#  #      /                                        \            |
#  #  /                                                 \      YO-----> X
#  # This results in:
#  #  (adjacent faceis now top plane)
#  #  ----------------------O\  (top arist)
#  #                           \
#  #                             \
#  #                               \     (adjacent face)
#  #                                 \
#  #                                   \
#  #                                     \
#
#  angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
#  rotation_angle = (np.pi - angle_between_faces) / 2
#  rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
#  rotation_matrix = rotation.as_matrix()
#  vertices = np.dot(vertices, rotation_matrix)
#
#  return TriangularMesh(vertices=vertices.astype(np.float32),
#                        faces=np.array(faces, dtype=np.int32))


def _two_split_unit_sphere_triangle_faces(
    triangular_mesh: TriangularMesh) -> TriangularMesh:
  """Splits each triangular face into 4 triangles keeping the orientation."""

  # Every time we split a triangle into 4 we will be adding 3 extra vertices,
  # located at the edge centres.
  # This class handles the positioning of the new vertices, and avoids creating
  # duplicates.
  new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

  new_faces = []
  for ind1, ind2, ind3 in triangular_mesh.faces:
    # Transform each triangular face into 4 triangles,
    # preserving the orientation.
    #                    ind3
    #                   /    \
    #                /          \
    #              /      #3       \
    #            /                  \
    #         ind31 -------------- ind23
    #         /   \                /   \
    #       /       \     #4     /      \
    #     /    #1     \        /    #2    \
    #   /               \    /              \
    # ind1 ------------ ind12 ------------ ind2
    ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
    ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
    ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
    # Note how each of the 4 triangular new faces specifies the order of the
    # vertices to preserve the orientation of the original face. As the input
    # face should always be counter-clockwise as specified in the diagram,
    # this means child faces should also be counter-clockwise.
    new_faces.extend([[ind1, ind12, ind31],  # 1
                      [ind12, ind2, ind23],  # 2
                      [ind31, ind23, ind3],  # 3
                      [ind12, ind23, ind31],  # 4
                      ])
  return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(),
                        faces=np.array(new_faces, dtype=np.int32))


class _ChildVerticesBuilder(object):
  """Bookkeeping of new child vertices added to an existing set of vertices."""

  def __init__(self, parent_vertices):

    # Because the same new vertex will be required when splitting adjacent
    # triangles (which share an edge) we keep them in a hash table indexed by
    # sorted indices of the vertices adjacent to the edge, to avoid creating
    # duplicated child vertices.
    self._child_vertices_index_mapping = {}
    self._parent_vertices = parent_vertices
    # We start with all previous vertices.
    self._all_vertices_list = list(parent_vertices)

  def _get_child_vertex_key(self, parent_vertex_indices):
    return tuple(sorted(parent_vertex_indices))

  def _create_child_vertex(self, parent_vertex_indices):
    """Creates a new vertex."""
    # Position for new vertex is the middle point, between the parent points,
    # projected to unit sphere.
    # print(parent_vertex_indices) # to debug index 5 out bounds 
    child_vertex_position = self._parent_vertices[
        list(parent_vertex_indices)].mean(0)
    global var1
    var1 = (parent_vertex_indices, child_vertex_position)
    x, y, z = child_vertex_position
    # jax.debug.print("🤯 Antes {} 🤯", (x, y, z))
    # x, y = (np.sqrt(1 - z**2) / np.sqrt(x**2 + y**2)) * np.array([x, y])
    # jax.debug.print("🤯 Despues {} 🤯", (x, y, z))
    child_vertex_position = np.array([x, y, z])
    # child_vertex_position /= np.linalg.norm(child_vertex_position)

    # Add the vertex to the output list. The index for this new vertex will
    # match the length of the list before adding it.
    child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
    self._child_vertices_index_mapping[child_vertex_key] = len(
        self._all_vertices_list)
    self._all_vertices_list.append(child_vertex_position)

  def get_new_child_vertex_index(self, parent_vertex_indices):
    """Returns index for a child vertex, creating it if necessary."""
    # Get the key to see if we already have a new vertex in the middle.
    child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
    if child_vertex_key not in self._child_vertices_index_mapping:
      self._create_child_vertex(parent_vertex_indices)
    return self._child_vertices_index_mapping[child_vertex_key]

  def get_all_vertices(self):
    """Returns an array with old vertices."""
    return np.array(self._all_vertices_list)


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms polygonal faces to sender and receiver indices.

  It does so by transforming every face into N_i edges. Such if the triangular
  face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

  If all faces have consistent orientation, and the surface represented by the
  faces is closed, then every edge in a polygon with a certain orientation
  is also part of another polygon with the opposite orientation. In this
  situation, the edges returned by the method are always bidirectional.

  Args:
    faces: Integer array of shape [num_faces, 3]. Contains node indices
        adjacent to each face.
  Returns:
    Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

  """
  assert faces.ndim == 2
  assert faces.shape[-1] == 3
  senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
  receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
  return senders, receivers
