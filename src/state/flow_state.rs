//! Central graph state container.
//!
//! [`FlowState`] owns the node and edge vectors, the internal lookup cache,
//! viewport, and animation state.  It is the single source of truth passed
//! to [`crate::render::canvas::FlowCanvas`] each frame.

use crate::types::position::CoordinateExtent;
use smallvec::SmallVec;
use std::collections::HashMap;

use crate::animation::viewport_animation::ViewportAnimation;
use crate::config::FlowConfig;
use crate::graph::utils::get_nodes_bounds;
use crate::types::changes::{EdgeChange, NodeChange};
use crate::types::connection::ConnectionState;
use crate::types::edge::{Edge, EdgeId};
use crate::types::handle::{Handle, HandleType, NodeHandle};
use crate::types::node::{InternalNode, Node, NodeHandleBounds, NodeId, NodeInternals};
use crate::types::position::Position;
use crate::types::viewport::Viewport;

/// Cached flow-space edge geometry.
///
/// Built lazily on first render and reused on every subsequent frame until
/// invalidated by a change to either endpoint node (move/resize) or to the
/// edge itself (replace, anchor change, structural). Pan/zoom never
/// invalidate the cache — the cached polyline is in flow space, so per-frame
/// work is reduced to one transform per vertex plus stroking.
#[derive(Debug, Clone)]
pub(crate) struct EdgePathCache {
    /// Polyline samples in flow space. For straight edges this is two points;
    /// for smoothstep / step, the full orthogonal path; for bezier, a 64-point
    /// sampling of the curve.
    pub points: SmallVec<[egui::Pos2; 16]>,
    /// Suggested label anchor in flow space.
    pub label_pos: egui::Pos2,
    /// Resolved source-side `Position` (for arrow marker orientation).
    pub source_pos: Position,
    /// Resolved target-side `Position` (for arrow marker orientation).
    pub target_pos: Position,
}

use super::changes::{apply_edge_changes, apply_node_changes};

/// Central state for the flow graph.
pub struct FlowState<ND = (), ED = ()> {
    /// User-facing nodes in insertion order. Mutated through
    /// [`add_node`](Self::add_node) / [`apply_node_changes`](Self::apply_node_changes).
    pub nodes: Vec<Node<ND>>,
    /// User-facing edges. Mutated through [`add_edge`](Self::add_edge) /
    /// [`apply_edge_changes`](Self::apply_edge_changes).
    pub edges: Vec<Edge<ED>>,
    /// Internal lookup cache keyed by [`NodeId`], holding per-node computed
    /// data (absolute position, z-index, handle bounds). Rebuilt by the state
    /// on structural changes; patched in place on position/dimensions/select.
    pub node_lookup: HashMap<NodeId, InternalNode<ND>>,
    /// Current viewport (pan offset + zoom). Mutated by the canvas each frame
    /// in response to pan/zoom input and by animation ticks.
    pub viewport: Viewport,
    /// Active drag-to-connect state (if any). Mutated by the canvas as the
    /// user drags from a handle.
    pub connection_state: ConnectionState,
    /// Screen-space rectangle of an in-progress box-selection. `None` when
    /// the user is not drawing a selection. Set by the canvas.
    pub selection_rect: Option<egui::Rect>,
    /// Configuration applied to rendering, interaction, and defaults. Owned
    /// by the state so builders and validators can read it without extra
    /// threading.
    pub config: FlowConfig,
    /// Currently running viewport animation, if any. Set by the `fit_*` /
    /// `zoom_*` / `set_viewport` helpers and cleared once complete.
    pub viewport_animation: Option<ViewportAnimation>,
    /// Tracks whether any edge is animated (for repaint requests).
    pub has_animated_edges: bool,
    /// Cached z-sorted node IDs. Invalidated on structural changes.
    sorted_ids_cache: Option<Vec<NodeId>>,
    /// Pending target for smoothed scroll/pinch zoom. `None` when no smoothing
    /// animation is in progress (or when `config.zoom_smoothing == 0.0`).
    /// Mutated by the pan/zoom interaction handler and the per-frame lerp in
    /// [`crate::render::canvas::FlowCanvas`].
    pub(crate) zoom_target: Option<ZoomTarget>,
    /// Per-edge cached flow-space polyline. Built lazily during render and
    /// invalidated by node move/resize and edge mutations. Pan / zoom do not
    /// invalidate it.
    pub(crate) edge_path_cache: HashMap<EdgeId, EdgePathCache>,
}

/// Pending scroll/pinch zoom target used by the smoothed-zoom path.
///
/// The pointer is stored in screen-space so each lerp step can re-anchor the
/// viewport — keeping the point under the cursor fixed while zoom converges.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ZoomTarget {
    pub(crate) pointer: egui::Pos2,
    pub(crate) zoom: f32,
}

impl<ND: Clone, ED: Clone> FlowState<ND, ED> {
    /// Create a new flow state with the given configuration.
    pub fn new(config: FlowConfig) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_lookup: HashMap::new(),
            viewport: Viewport::default(),
            connection_state: ConnectionState::None,
            selection_rect: None,
            config,
            viewport_animation: None,
            has_animated_edges: false,
            sorted_ids_cache: None,
            zoom_target: None,
            edge_path_cache: HashMap::new(),
        }
    }

    /// Drop cached path entries for every edge connected to `node_id`.
    /// Called when a node moves, resizes, or is replaced — its perimeter
    /// position changes the resolved endpoints of every connected edge.
    fn invalidate_edges_for_node(&mut self, node_id: &NodeId) {
        if self.edge_path_cache.is_empty() {
            return;
        }
        for edge in &self.edges {
            if &edge.source == node_id || &edge.target == node_id {
                self.edge_path_cache.remove(&edge.id);
            }
        }
    }

    /// Add a node to the graph and rebuild the internal lookup.
    pub fn add_node(&mut self, node: Node<ND>) {
        self.nodes.push(node);
        self.rebuild_lookup();
    }

    /// Add multiple nodes to the graph in a single batch.
    ///
    /// More efficient than calling [`add_node`](Self::add_node) in a loop
    /// because the internal lookup is rebuilt only once after all nodes are
    /// inserted.
    pub fn add_nodes(&mut self, nodes: impl IntoIterator<Item = Node<ND>>) {
        let before = self.nodes.len();
        self.nodes.extend(nodes);
        if self.nodes.len() > before {
            self.rebuild_lookup();
        }
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: Edge<ED>) {
        if edge.animated {
            self.has_animated_edges = true;
        }
        self.edges.push(edge);
    }

    /// Add multiple edges to the graph in a single batch.
    pub fn add_edges(&mut self, edges: impl IntoIterator<Item = Edge<ED>>) {
        for edge in edges {
            if edge.animated {
                self.has_animated_edges = true;
            }
            self.edges.push(edge);
        }
    }

    /// Apply a batch of node mutations.  Non-structural changes (position,
    /// dimensions, select) are applied incrementally; structural changes
    /// (add, remove, replace) trigger a full lookup rebuild.
    pub fn apply_node_changes(&mut self, changes: &[NodeChange<ND>]) {
        let has_structural = changes.iter().any(|c| {
            matches!(
                c,
                NodeChange::Add { .. } | NodeChange::Remove { .. } | NodeChange::Replace { .. }
            )
        });

        if has_structural {
            // Structural changes require a full rebuild.
            apply_node_changes(changes, &mut self.nodes);
            self.rebuild_lookup();
        } else {
            // Non-structural (Position, Dimensions, Select): update nodes vec
            // and patch the lookup in place — avoids cloning all N nodes.
            apply_node_changes(changes, &mut self.nodes);
            self.apply_incremental_lookup_updates(changes);
        }
    }

    /// Apply non-structural changes to the lookup without a full rebuild.
    fn apply_incremental_lookup_updates(&mut self, changes: &[NodeChange<ND>]) {
        let mut needs_parent_update = false;

        for change in changes {
            match change {
                NodeChange::Position { id, position, dragging } => {
                    if let Some(internal) = self.node_lookup.get_mut(id) {
                        if let Some(pos) = position {
                            internal.node.position = *pos;
                            internal.internals.position_absolute = *pos;
                            needs_parent_update = true;
                        }
                        if let Some(d) = dragging {
                            internal.node.dragging = *d;
                        }
                    }
                    if position.is_some() {
                        self.invalidate_edges_for_node(id);
                    }
                }
                NodeChange::Dimensions { id, dimensions } => {
                    if let Some(internal) = self.node_lookup.get_mut(id) {
                        internal.node.measured = *dimensions;
                        if let Some(d) = dimensions {
                            internal.node.width = Some(d.width);
                            internal.node.height = Some(d.height);
                        }
                        // Rebuild handle bounds since dimensions changed.
                        internal.internals.handle_bounds =
                            build_handle_bounds(&internal.node, &self.config);
                    }
                    self.invalidate_edges_for_node(id);
                }
                NodeChange::Select { id, selected } => {
                    if let Some(internal) = self.node_lookup.get_mut(id) {
                        internal.node.selected = *selected;
                    }
                }
                _ => {} // Structural changes handled by full rebuild path
            }
        }

        if needs_parent_update {
            self.update_absolute_positions();
        }
    }

    /// Apply a batch of edge mutations.
    pub fn apply_edge_changes(&mut self, changes: &[EdgeChange<ED>]) {
        // Invalidate cached paths for any edge whose geometry-affecting fields
        // are about to change. Style-only changes (`EdgeChange::Style`) and
        // selection-only changes leave the path intact.
        for change in changes {
            match change {
                EdgeChange::Remove { id }
                | EdgeChange::Replace { id, .. }
                | EdgeChange::Anchor { id, .. } => {
                    self.edge_path_cache.remove(id);
                }
                EdgeChange::Add { .. } | EdgeChange::Style { .. } | EdgeChange::Select { .. } => {}
            }
        }

        apply_edge_changes(changes, &mut self.edges);
        self.has_animated_edges = self.edges.iter().any(|e| e.animated);
    }

    /// Rebuild internal node lookup from user nodes.
    pub fn rebuild_lookup(&mut self) {
        self.sorted_ids_cache = None; // invalidate cache
        // Structural changes can shift any node — drop all cached edge paths.
        self.edge_path_cache.clear();
        self.node_lookup.clear();
        for node in &self.nodes {
            let handle_bounds = build_handle_bounds(node, &self.config);
            let internal = InternalNode {
                internals: NodeInternals {
                    position_absolute: node.position,
                    z: node.z_index.unwrap_or(0),
                    handle_bounds,
                    shape: crate::types::position::NodeShape::Rect,
                },
                node: node.clone(),
            };
            self.node_lookup.insert(node.id.clone(), internal);
        }
        self.update_absolute_positions();
    }

    /// Compute absolute positions for child nodes.
    fn update_absolute_positions(&mut self) {
        // Fast path: skip if no node has a parent.
        if !self.nodes.iter().any(|n| n.parent_id.is_some()) {
            return;
        }

        // Collect parent relationships.
        let parent_map: Vec<(NodeId, NodeId)> = self
            .nodes
            .iter()
            .filter_map(|n| n.parent_id.as_ref().map(|pid| (n.id.clone(), pid.clone())))
            .collect();

        // Pre-collect parent data (position_absolute + z) to avoid simultaneous
        // mutable + immutable borrows of node_lookup inside the loop.
        let parent_data: HashMap<&NodeId, (egui::Pos2, i32)> = parent_map
            .iter()
            .filter_map(|(_, pid)| {
                self.node_lookup
                    .get(pid)
                    .map(|p| (pid, (p.internals.position_absolute, p.internals.z)))
            })
            .collect();

        // Resolve absolute positions (simple single-level parent support).
        for (child_id, parent_id) in &parent_map {
            if let Some(&(parent_pos, parent_z)) = parent_data.get(parent_id)
                && let Some(child) = self.node_lookup.get_mut(child_id)
            {
                child.internals.position_absolute = egui::pos2(
                    parent_pos.x + child.node.position.x,
                    parent_pos.y + child.node.position.y,
                );
                if child.internals.z <= parent_z {
                    child.internals.z = parent_z + 1;
                }
            }
        }
    }

    /// Animate viewport to fit all nodes.
    pub fn fit_view(&mut self, canvas_rect: egui::Rect, padding: f32, current_time: f64) {
        let bounds = get_nodes_bounds(&self.node_lookup);
        if bounds == egui::Rect::NOTHING {
            return;
        }
        let target = self.viewport_to_fit(bounds, canvas_rect, padding);
        self.animate_viewport(target, current_time);
    }

    /// Viewport that frames `bounds` (flow space) inside `canvas_rect` (screen
    /// space). The viewport offset is a screen position, so the canvas origin
    /// is added: a canvas that does not start at the window's top-left corner
    /// (next to a side panel, below a toolbar) is framed correctly.
    fn viewport_to_fit(&self, bounds: egui::Rect, canvas_rect: egui::Rect, padding: f32) -> Viewport {
        let mut target = crate::graph::utils::get_viewport_for_bounds(
            bounds,
            canvas_rect.width(),
            canvas_rect.height(),
            self.config.min_zoom,
            self.config.max_zoom,
            padding,
        );
        target.x += canvas_rect.min.x;
        target.y += canvas_rect.min.y;
        target
    }

    /// Animate the viewport to fit an arbitrary **flow-space** bounding box
    /// into the canvas area.
    ///
    /// This is the Rust equivalent of xyflow's `fitBounds` helper.  Use it
    /// when you want to frame a specific region of the graph rather than all
    /// nodes (e.g. fitting the selected nodes, or an imported sub-graph).
    ///
    /// `padding` is extra space in canvas pixels added around the bounds.
    ///
    /// ```rust,ignore
    /// // Frame just the first two nodes:
    /// let bounds = CoordinateExtent {
    ///     min: egui::pos2(80.0, 100.0),
    ///     max: egui::pos2(480.0, 200.0),
    /// };
    /// state.fit_bounds(bounds, canvas_rect, 20.0, current_time);
    /// ```
    pub fn fit_bounds(
        &mut self,
        bounds: CoordinateExtent,
        canvas_rect: egui::Rect,
        padding: f32,
        current_time: f64,
    ) {
        let flow_rect = egui::Rect::from_min_max(bounds.min, bounds.max);

        if flow_rect.width() <= 0.0 || flow_rect.height() <= 0.0 {
            return;
        }

        let target = self.viewport_to_fit(flow_rect, canvas_rect, padding);
        self.animate_viewport(target, current_time);
    }

    /// Animate the viewport to fit only the currently **selected** nodes.
    ///
    /// Does nothing when no nodes are selected.
    pub fn fit_selected_nodes(&mut self, canvas_rect: egui::Rect, padding: f32, current_time: f64) {
        // Compute bounds directly without cloning into a temporary HashMap.
        let bounds = self
            .node_lookup
            .values()
            .filter(|n| n.node.selected && !n.node.hidden)
            .fold(egui::Rect::NOTHING, |acc, n| acc.union(n.rect()));

        if bounds == egui::Rect::NOTHING {
            return;
        }

        let target = self.viewport_to_fit(bounds, canvas_rect, padding);
        self.animate_viewport(target, current_time);
    }

    /// Animate viewport zoom in.
    pub fn zoom_in(&mut self, current_time: f64) {
        let target = Viewport {
            zoom: (self.viewport.zoom * 1.2).min(self.config.max_zoom),
            ..self.viewport
        };
        self.animate_viewport(target, current_time);
    }

    /// Animate viewport zoom out.
    pub fn zoom_out(&mut self, current_time: f64) {
        let target = Viewport {
            zoom: (self.viewport.zoom / 1.2).max(self.config.min_zoom),
            ..self.viewport
        };
        self.animate_viewport(target, current_time);
    }

    /// Animate to center the view on a specific flow-space point with an
    /// optional zoom level.  If `zoom` is `None` the current zoom is kept.
    ///
    /// ```rust,ignore
    /// state.set_center(400.0, 200.0, Some(1.0), current_time);
    /// ```
    pub fn set_center(
        &mut self,
        x: f32,
        y: f32,
        zoom: Option<f32>,
        canvas_rect: egui::Rect,
        current_time: f64,
    ) {
        let target_zoom = zoom
            .unwrap_or(self.viewport.zoom)
            .clamp(self.config.min_zoom, self.config.max_zoom);
        // viewport offset so that (x, y) appears at the canvas centre
        let target = Viewport {
            x: canvas_rect.center().x - x * target_zoom,
            y: canvas_rect.center().y - y * target_zoom,
            zoom: target_zoom,
        };
        self.animate_viewport(target, current_time);
    }

    /// Animate to an arbitrary viewport with a custom duration and easing
    /// function.
    ///
    /// ```rust,ignore
    /// use egui_xyflow::animation::easing::ease_linear;
    /// state.set_viewport(Viewport { x: 0.0, y: 0.0, zoom: 1.0 }, 0.5, ease_linear, current_time);
    /// ```
    pub fn set_viewport(
        &mut self,
        target: Viewport,
        duration: f32,
        easing: fn(f32) -> f32,
        current_time: f64,
    ) {
        self.viewport_animation = Some(ViewportAnimation::new(
            self.viewport,
            target,
            duration,
            current_time,
            easing,
        ));
    }

    /// Animate to a specific viewport using the default duration and easing.
    pub fn animate_viewport(&mut self, target: Viewport, current_time: f64) {
        self.viewport_animation = Some(ViewportAnimation::new(
            self.viewport,
            target,
            self.config.default_transition_duration,
            current_time,
            self.config.default_transition_easing,
        ));
    }

    /// Tick viewport animation. Returns true if animation is active.
    pub fn tick_animation(&mut self, current_time: f64) -> bool {
        if let Some(ref mut anim) = self.viewport_animation {
            self.viewport = anim.tick(current_time);
            if !anim.active {
                self.viewport_animation = None;
                return false;
            }
            return true;
        }
        false
    }

    /// Get sorted node IDs by z-index (lowest first).
    ///
    /// Uses an internal cache invalidated on structural changes
    /// (add/remove/replace). Non-structural changes (position, select)
    /// reuse the cached ordering.
    pub fn sorted_node_ids(&mut self) -> Vec<NodeId> {
        if let Some(ref cached) = self.sorted_ids_cache {
            return cached.clone();
        }
        let mut ids = Vec::with_capacity(self.node_lookup.len());
        ids.extend(self.node_lookup.keys().cloned());
        ids.sort_by_key(|id| self.node_lookup.get(id).map(|n| n.internals.z).unwrap_or(0));
        self.sorted_ids_cache = Some(ids.clone());
        ids
    }
}

/// Build handle bounds from user-specified handles.
///
/// A handle with a non-zero [`NodeHandle::x`] or [`NodeHandle::y`] is placed
/// at exactly that offset. All other handles are spread evenly along their
/// side, counting only the automatically placed handles on that side.
fn build_handle_bounds<D>(node: &Node<D>, config: &FlowConfig) -> NodeHandleBounds {
    let node_w = node.width.unwrap_or(config.default_node_width);
    let node_h = node.height.unwrap_or(config.default_node_height);

    let resolve = |handle_type: HandleType| -> Vec<Handle> {
        let is_manual = |h: &NodeHandle| h.x != 0.0 || h.y != 0.0;
        let of_type: Vec<&NodeHandle> = node
            .handles
            .iter()
            .filter(|h| h.handle_type == handle_type)
            .collect();
        of_type
            .iter()
            .map(|nh| {
                let width = nh.width.unwrap_or(config.handle_size);
                let height = nh.height.unwrap_or(config.handle_size);
                let (x, y) = if is_manual(nh) {
                    (nh.x, nh.y)
                } else {
                    let auto_on_side = || {
                        of_type
                            .iter()
                            .filter(|h| h.position == nh.position && !is_manual(h))
                    };
                    let count = auto_on_side().count();
                    let idx = auto_on_side()
                        .position(|h| std::ptr::eq(*h, *nh))
                        .unwrap_or(0);
                    compute_handle_offset(nh.position, node_w, node_h, width, height, count, idx)
                };
                Handle {
                    id: nh.id.clone(),
                    node_id: node.id.0.clone(), // Arc<str> clone — O(1)
                    x,
                    y,
                    position: nh.position,
                    handle_type,
                    width,
                    height,
                }
            })
            .collect()
    };

    NodeHandleBounds {
        source: resolve(HandleType::Source),
        target: resolve(HandleType::Target),
    }
}

/// Compute handle offset within node bounds.
fn compute_handle_offset(
    position: Position,
    node_w: f32,
    node_h: f32,
    handle_w: f32,
    handle_h: f32,
    count: usize,
    index: usize,
) -> (f32, f32) {
    let half_w = handle_w / 2.0;
    let half_h = handle_h / 2.0;
    match position {
        Position::Top => {
            let spacing = node_w / (count as f32 + 1.0);
            let x = spacing * (index as f32 + 1.0) - half_w;
            (x, -half_h)
        }
        Position::Bottom => {
            let spacing = node_w / (count as f32 + 1.0);
            let x = spacing * (index as f32 + 1.0) - half_w;
            (x, node_h - half_h)
        }
        Position::Left => {
            let spacing = node_h / (count as f32 + 1.0);
            let y = spacing * (index as f32 + 1.0) - half_h;
            (-half_w, y)
        }
        Position::Right => {
            let spacing = node_h / (count as f32 + 1.0);
            let y = spacing * (index as f32 + 1.0) - half_h;
            (node_w - half_w, y)
        }
        Position::Center => {
            // Handle sits at the center of the node
            (node_w / 2.0 - half_w, node_h / 2.0 - half_h)
        }
        Position::Closest => {
            // Resolved dynamically at render time; place at center for now
            (node_w / 2.0 - half_w, node_h / 2.0 - half_h)
        }
    }
}

#[cfg(test)]
mod handle_bounds_tests {
    use super::*;

    fn node_with(handles: Vec<NodeHandle>) -> Node<()> {
        let mut node: Node<()> = Node::builder("n").size(100.0, 60.0).build();
        node.handles = handles;
        node
    }

    #[test]
    fn manual_offset_is_used_as_given() {
        let node = node_with(vec![NodeHandle::source(Position::Right).at(90.0, 12.0)]);
        let bounds = build_handle_bounds(&node, &FlowConfig::default());
        assert_eq!((bounds.source[0].x, bounds.source[0].y), (90.0, 12.0));
        assert_eq!(bounds.source[0].position, Position::Right);
    }

    #[test]
    fn automatic_handles_are_spaced_without_counting_manual_ones() {
        let config = FlowConfig::default();
        let alone = node_with(vec![NodeHandle::source(Position::Right)]);
        let mixed = node_with(vec![
            NodeHandle::source(Position::Right).at(90.0, 12.0),
            NodeHandle::source(Position::Right),
        ]);
        let expected = build_handle_bounds(&alone, &config).source[0].y;
        assert_eq!(build_handle_bounds(&mixed, &config).source[1].y, expected);
    }

    #[test]
    fn size_override_is_applied_and_keeps_the_handle_centred_on_the_edge() {
        let node = node_with(vec![NodeHandle::target(Position::Left).size(20.0, 30.0)]);
        let h = &build_handle_bounds(&node, &FlowConfig::default()).target[0];
        assert_eq!((h.width, h.height), (20.0, 30.0));
        assert_eq!(h.center(), egui::pos2(0.0, 30.0));
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    #[test]
    fn fit_view_centres_nodes_in_a_canvas_that_is_offset_on_screen() {
        let mut state: FlowState<(), ()> = FlowState::new(FlowConfig::default());
        state.add_node(Node::builder("a").position(egui::pos2(0.0, 0.0)).size(100.0, 100.0).build());
        // A canvas below a 50 px toolbar and right of a 200 px side panel.
        let canvas = egui::Rect::from_min_size(egui::pos2(200.0, 50.0), egui::vec2(800.0, 600.0));

        let target = state.viewport_to_fit(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 100.0)), canvas, 0.0);

        // The node's centre (50, 50 in flow space) must land on the canvas centre.
        let screen = egui::pos2(50.0 * target.zoom + target.x, 50.0 * target.zoom + target.y);
        assert!((screen - canvas.center()).length() < 1e-3, "{screen:?} vs {:?}", canvas.center());
    }
}
