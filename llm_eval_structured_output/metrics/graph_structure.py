from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

import llm_client
from metrics.base import BaseMetric, MetricResult
from prompts import render_system_prompt


class GraphVertex(BaseModel):
    vertex_id: str = Field(
        ...,
        description="Unique vertex identifier (for example: v1, v2, v3).",
    )
    label: str = Field(
        ...,
        description="Short vertex label copied from the rationale.",
    )
    description: str = Field(
        ...,
        description="What this vertex represents in the rationale graph.",
    )
    text_correspondence: str = Field(
        ...,
        description="Exact quote from the rationale that grounds this vertex.",
    )


class GraphEdge(BaseModel):
    source_vertex_label: str = Field(
        ...,
        description="Label of the source vertex.",
    )
    target_vertex_label: str = Field(
        ...,
        description="Label of the target vertex.",
    )
    edge_label: str = Field(
        ...,
        description="Short label describing the directed relation.",
    )
    description: str = Field(
        ...,
        description="Explanation of this directed relation.",
    )
    text_correspondence: str = Field(
        ...,
        description="Exact quote from the rationale that grounds this edge.",
    )


class RationaleGraphResponse(BaseModel):
    general_reasoning: str = Field(
        ...,
        description="Brief explanation of how the rationale was mapped to the graph.",
    )
    vertices: list[GraphVertex] = Field(
        default_factory=list,
        description="All vertices extracted strictly from rationale text spans.",
    )
    edges: list[GraphEdge] = Field(
        default_factory=list,
        description="All directed edges extracted strictly from rationale text spans.",
    )


class RationaleGraphMetric(BaseMetric):
    metric_type = "graph"

    _MAX_SIMPLE_CYCLES = 10_000

    # Metrics where normalization to rationale length is informative.
    _NORMALIZABLE_KEYS = {
        "mean_out_degree",
        "max_out_degree",
        "std_out_degree",
        "average_depth_length",
        "maximum_depth",
        "std_depth",
        "mean_in_degree",
        "number_of_sink_nodes",
        "number_of_source_nodes",
        "cycle_count",
        "number_of_isolated_subgraphs",
    }

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.name = name
        self.description = description
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template

    @property
    def schema(self) -> type[RationaleGraphResponse]:
        return RationaleGraphResponse

    def evaluate(
        self,
        text: str,
        *,
        dataset: str = "unknown",
        problem_id: str = "N/A",
        checkpoint: str = "N/A",
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> MetricResult:
        if not isinstance(text, str) or not text.strip():
            return MetricResult(
                metric_name=self.name,
                error="Empty or invalid input text.",
                raw={"scalar_metrics": {}, "normalized_scalar_metrics": {}},
                scalar_metrics={},
                normalized_scalar_metrics={},
            )

        user_prompt = (
            self._user_prompt_template.replace("{text}", text).replace("{dataset}", dataset)
        )

        payload = llm_client.ask_llm(
            system_prompt=render_system_prompt(self._system_prompt, self.name, dataset),
            user_prompt=user_prompt,
            response_schema=RationaleGraphResponse,
            dataset=dataset,
            problem_id=problem_id,
            metric_type=self.name,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        tokens: dict[str, int] = payload.pop("__tokens__", {})
        general_reasoning = payload.get("general_reasoning", "")

        vertices = (
            payload.get("vertices", []) if isinstance(payload.get("vertices"), list) else []
        )
        edges = payload.get("edges", []) if isinstance(payload.get("edges"), list) else []

        graph, clean_vertices, clean_edges = self._build_graph(vertices, edges)
        scalar_metrics = self._compute_scalar_metrics(graph)

        word_count = max(1, len(text.split()))
        normalized_scalar_metrics = self._normalize_scalar_metrics(
            scalar_metrics,
            word_count,
        )

        examples: list[dict[str, Any]] = []
        for vertex in clean_vertices:
            examples.append(
                {
                    "kind": "vertex",
                    "label": vertex["label"],
                    "description": vertex["description"],
                    "text_correspondence": vertex["text_correspondence"],
                }
            )
        for edge in clean_edges:
            examples.append(
                {
                    "kind": "edge",
                    "source": edge["source_vertex_label"],
                    "target": edge["target_vertex_label"],
                    "edge_label": edge["edge_label"],
                    "description": edge["description"],
                    "text_correspondence": edge["text_correspondence"],
                }
            )

        primary_score = scalar_metrics.get("mean_out_degree", 0.0)

        raw_payload = dict(payload)
        raw_payload["scalar_metrics"] = scalar_metrics
        raw_payload["normalized_scalar_metrics"] = normalized_scalar_metrics
        raw_payload["vertices"] = clean_vertices
        raw_payload["edges"] = clean_edges

        return MetricResult(
            metric_name=self.name,
            detected=graph.number_of_nodes() > 0,
            reasoning=general_reasoning,
            examples=examples,
            score=primary_score,
            tokens=tokens,
            raw=raw_payload,
            scalar_metrics=scalar_metrics,
            normalized_scalar_metrics=normalized_scalar_metrics,
        )

    def _build_graph(
        self,
        vertices: list[Any],
        edges: list[Any],
    ) -> tuple[nx.DiGraph, list[dict[str, str]], list[dict[str, str]]]:
        graph = nx.DiGraph()

        clean_vertices: list[dict[str, str]] = []
        label_to_id: dict[str, str] = {}

        for item in vertices:
            if not isinstance(item, dict):
                continue
            vertex_id = str(item.get("vertex_id", "")).strip()
            label = str(item.get("label", "")).strip()
            if not vertex_id or not label:
                continue
            if vertex_id in graph:
                continue
            description = str(item.get("description", "")).strip()
            text_corr = str(item.get("text_correspondence", "")).strip()

            graph.add_node(vertex_id, label=label)
            if label not in label_to_id:
                label_to_id[label] = vertex_id
            clean_vertices.append(
                {
                    "vertex_id": vertex_id,
                    "label": label,
                    "description": description,
                    "text_correspondence": text_corr,
                }
            )

        clean_edges: list[dict[str, str]] = []
        for item in edges:
            if not isinstance(item, dict):
                continue
            source_label = str(item.get("source_vertex_label", "")).strip()
            target_label = str(item.get("target_vertex_label", "")).strip()
            edge_label = str(item.get("edge_label", "")).strip()
            if not source_label or not target_label:
                continue

            source_id = label_to_id.get(source_label)
            target_id = label_to_id.get(target_label)
            if source_id is None or target_id is None:
                continue

            if graph.has_edge(source_id, target_id):
                continue

            description = str(item.get("description", "")).strip()
            text_corr = str(item.get("text_correspondence", "")).strip()
            graph.add_edge(source_id, target_id, label=edge_label)
            clean_edges.append(
                {
                    "source_vertex_label": source_label,
                    "target_vertex_label": target_label,
                    "edge_label": edge_label,
                    "description": description,
                    "text_correspondence": text_corr,
                }
            )

        return graph, clean_vertices, clean_edges

    def _compute_scalar_metrics(self, graph: nx.DiGraph) -> dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {
                "mean_out_degree": 0.0,
                "max_out_degree": 0.0,
                "std_out_degree": 0.0,
                "average_depth_length": 0.0,
                "maximum_depth": 0.0,
                "std_depth": 0.0,
                "depth_to_width_ratio": 0.0,
                "mean_in_degree": 0.0,
                "in_degree_skewness": 0.0,
                "in_degree_centralization": 0.0,
                "number_of_sink_nodes": 0.0,
                "number_of_source_nodes": 0.0,
                "cycle_count": 0.0,
                "number_of_isolated_subgraphs": 0.0,
                "mean_betweenness_centrality": 0.0,
                "max_betweenness_centrality": 0.0,
            }

        in_degrees = [float(d) for _, d in graph.in_degree()]
        out_degrees = [float(d) for _, d in graph.out_degree()]

        mean_out = mean(out_degrees)
        mean_in = mean(in_degrees)
        std_out = pstdev(out_degrees) if len(out_degrees) > 1 else 0.0

        sink_nodes = float(sum(1 for _, d in graph.out_degree() if d == 0))
        source_nodes = float(sum(1 for _, d in graph.in_degree() if d == 0))
        isolated_subgraphs = float(nx.number_weakly_connected_components(graph))

        depth_lengths = self._root_to_leaf_depths(graph)
        avg_depth = mean(depth_lengths) if depth_lengths else 0.0
        max_depth = float(max(depth_lengths)) if depth_lengths else 0.0
        std_depth = pstdev(depth_lengths) if len(depth_lengths) > 1 else 0.0

        depth_to_width_ratio = avg_depth / mean_out if mean_out > 0 else 0.0

        in_deg_skew = self._skewness(in_degrees)
        in_deg_centralization = self._in_degree_centralization(graph, in_degrees)

        betweenness = nx.betweenness_centrality(graph, normalized=True)
        between_vals = [float(v) for v in betweenness.values()]
        mean_between = mean(between_vals) if between_vals else 0.0
        max_between = max(between_vals) if between_vals else 0.0

        cycle_count = float(self._cycle_count(graph))

        return {
            "mean_out_degree": round(mean_out, 6),
            "max_out_degree": round(float(max(out_degrees) if out_degrees else 0.0), 6),
            "std_out_degree": round(std_out, 6),
            "average_depth_length": round(float(avg_depth), 6),
            "maximum_depth": round(max_depth, 6),
            "std_depth": round(std_depth, 6),
            "depth_to_width_ratio": round(float(depth_to_width_ratio), 6),
            "mean_in_degree": round(mean_in, 6),
            "in_degree_skewness": round(float(in_deg_skew), 6),
            "in_degree_centralization": round(float(in_deg_centralization), 6),
            "number_of_sink_nodes": round(sink_nodes, 6),
            "number_of_source_nodes": round(source_nodes, 6),
            "cycle_count": round(cycle_count, 6),
            "number_of_isolated_subgraphs": round(isolated_subgraphs, 6),
            "mean_betweenness_centrality": round(float(mean_between), 6),
            "max_betweenness_centrality": round(float(max_between), 6),
        }

    def _root_to_leaf_depths(self, graph: nx.DiGraph) -> list[int]:
        condensation = nx.condensation(graph)
        if condensation.number_of_nodes() == 0:
            return []

        roots = [n for n, d in condensation.in_degree() if d == 0]
        leaves = [n for n, d in condensation.out_degree() if d == 0]
        if not roots or not leaves:
            return [0]

        topo_nodes = list(nx.topological_sort(condensation))
        depths: list[int] = []

        for root in roots:
            dist = {node: -1 for node in condensation.nodes()}
            dist[root] = 0
            for node in topo_nodes:
                if dist[node] < 0:
                    continue
                for succ in condensation.successors(node):
                    cand = dist[node] + 1
                    if cand > dist[succ]:
                        dist[succ] = cand
            for leaf in leaves:
                if dist[leaf] >= 0:
                    depths.append(int(dist[leaf]))

        return depths if depths else [0]

    def _skewness(self, values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        mu = mean(values)
        sigma = pstdev(values)
        if sigma == 0:
            return 0.0
        third = mean([(v - mu) ** 3 for v in values])
        return third / (sigma**3)

    def _in_degree_centralization(
        self,
        graph: nx.DiGraph,
        in_degrees: list[float],
    ) -> float:
        n = graph.number_of_nodes()
        if n <= 1:
            return 0.0
        max_in = max(in_degrees)
        numerator = sum(max_in - d for d in in_degrees)
        denominator = float((n - 1) ** 2)
        return numerator / denominator if denominator > 0 else 0.0

    def _cycle_count(self, graph: nx.DiGraph) -> int:
        cycle_total = 0
        for _ in nx.simple_cycles(graph):
            cycle_total += 1
            if cycle_total >= self._MAX_SIMPLE_CYCLES:
                break
        return cycle_total

    def _normalize_scalar_metrics(
        self,
        scalar_metrics: dict[str, float],
        word_count: int,
    ) -> dict[str, float]:
        norm_factor = 100.0 / word_count if word_count > 0 else 0.0
        normalized: dict[str, float] = {}
        for key, value in scalar_metrics.items():
            if key in self._NORMALIZABLE_KEYS:
                normalized[f"{key}_per_100_words"] = round(value * norm_factor, 6)
        return normalized
