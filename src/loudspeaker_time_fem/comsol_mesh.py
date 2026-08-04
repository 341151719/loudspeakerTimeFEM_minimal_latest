from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ComsolMesh:
    """Linear topology and geometric-entity labels from a COMSOL MPHTXT mesh."""

    points_rz_m: np.ndarray
    cells: dict[str, np.ndarray]
    entities: dict[str, np.ndarray]

    @property
    def line_cells(self) -> np.ndarray:
        return self.cells.get("edg", np.empty((0, 2), dtype=int))

    @property
    def line_tags(self) -> np.ndarray:
        return self.entities.get("edg", np.empty(0, dtype=int))

    def domain_edges(self) -> dict[tuple[int, int], set[int]]:
        out: dict[tuple[int, int], set[int]] = {}
        for kind in ("tri", "quad"):
            cells = self.cells.get(kind, np.empty((0, 0), dtype=int))
            domains = self.entities.get(kind, np.empty(0, dtype=int))
            for cell, domain in zip(cells, domains):
                ordered = cell if kind == "tri" else cell[[0, 1, 3, 2]]
                pairs = zip(ordered, np.roll(ordered, -1))
                for a, b in pairs:
                    key = tuple(sorted((int(a), int(b))))
                    out.setdefault(key, set()).add(int(domain))
        return out

    def boundary_entities(self) -> dict[tuple[int, int], int]:
        return {
            tuple(sorted(map(int, edge))): int(entity)
            for edge, entity in zip(
                self.cells.get("edg", np.empty((0, 2), dtype=int)),
                self.entities.get("edg", np.empty(0, dtype=int)),
            )
        }

    def acoustic_interface_edges(
        self,
        acoustic_domains: set[int],
        structure_domains: set[int] | None = None,
    ) -> list[tuple[int, int, int, int, int]]:
        """Return a,b,acoustic_domain,solid_domain,boundary_entity."""
        labels = self.boundary_entities()
        out = []
        for edge, domains in self.domain_edges().items():
            acoustic = sorted(domains & acoustic_domains)
            other_domains = domains - acoustic_domains
            if structure_domains is not None:
                other_domains &= structure_domains
            other = sorted(other_domains)
            if acoustic and other:
                out.append(
                    (edge[0], edge[1], acoustic[0], other[0], labels.get(edge, -1))
                )
        return out

    def triangulated_domains(
        self, selected_domains: set[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        triangles: list[list[int]] = []
        domains: list[int] = []
        for cell, domain in zip(
            self.cells.get("tri", np.empty((0, 3), dtype=int)),
            self.entities.get("tri", np.empty(0, dtype=int)),
        ):
            if int(domain) in selected_domains:
                triangles.append(list(map(int, cell)))
                domains.append(int(domain))
        for cell, domain in zip(
            self.cells.get("quad", np.empty((0, 4), dtype=int)),
            self.entities.get("quad", np.empty(0, dtype=int)),
        ):
            if int(domain) in selected_domains:
                a, b, c, d = map(int, cell)
                # COMSOL quad topology is [lower-left, lower-right,
                # upper-left, upper-right], so the perimeter is 0-1-3-2.
                triangles.extend(([a, b, d], [a, d, c]))
                domains.extend((int(domain), int(domain)))
        return np.asarray(triangles, dtype=int), np.asarray(domains, dtype=int)


def _records(path: Path) -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def load_comsol_mphtxt_mesh(path: str | Path) -> ComsolMesh:
    """Read the single Mesh object written by MeshSequence.export()."""
    source = Path(path)
    rows = _records(source)
    i = 0

    def take() -> str:
        nonlocal i
        value = rows[i]
        i += 1
        return value

    if take() != "0 1":
        raise ValueError(f"{source}: unsupported MPHTXT header")
    ntags = int(take())
    for _ in range(ntags):
        take()
    ntypes = int(take())
    object_types = [take().split(maxsplit=1)[-1] for _ in range(ntypes)]
    if object_types != ["obj"]:
        raise ValueError(f"{source}: expected one obj, got {object_types}")
    take()  # object index, class version, reserved
    mesh_class = take().split(maxsplit=1)[-1]
    if mesh_class != "Mesh":
        raise ValueError(f"{source}: object is {mesh_class}, not Mesh")
    take()  # Mesh serialization version
    sdim = int(take())
    if sdim != 2:
        raise ValueError(f"{source}: only 2D axisymmetric meshes are supported")
    npoints = int(take())
    lowest_index = int(take())
    if lowest_index != 0:
        raise ValueError(f"{source}: nonzero lowest vertex index is unsupported")
    points = np.asarray(
        [[float(value) for value in take().split()] for _ in range(npoints)],
        dtype=float,
    )
    element_type_count = int(take())
    cells: dict[str, np.ndarray] = {}
    entities: dict[str, np.ndarray] = {}
    for _ in range(element_type_count):
        kind = take().split(maxsplit=1)[-1]
        vertices_per_element = int(take())
        count = int(take())
        connectivity = np.asarray(
            [[int(value) for value in take().split()] for _ in range(count)],
            dtype=int,
        ).reshape(count, vertices_per_element)
        entity_count = int(take())
        if entity_count != count:
            raise ValueError(
                f"{source}: {kind} entity count {entity_count} != element count {count}"
            )
        cells[kind] = connectivity
        entity_values = np.asarray([int(take()) for _ in range(count)], dtype=int)
        # MeshSequence.export() writes edge entity indices zero-based, whereas
        # COMSOL Java/API boundary selections are one-based. Domain labels in
        # the same file already retain their Java/API numbers. Normalize only
        # boundary elements so exported diagnostics can be selected verbatim.
        if kind == "edg":
            entity_values = entity_values + 1
        entities[kind] = entity_values
    if i != len(rows):
        raise ValueError(f"{source}: {len(rows) - i} unparsed records")
    return ComsolMesh(points, cells, entities)
