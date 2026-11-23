use crate::graph::node_position::snap_position;
use crate::types::changes::NodeChange;
use crate::types::node::{InternalNode, NodeId};
use crate::types::position::Transform;
use std::collections::HashMap;

/// Where a snapped drag started and how far the pointer has travelled since.
///
/// Snapping each frame's small movement on its own rounds slow drags back to
/// where they started, so the node sticks and then jumps. Snapping
/// `origin + total movement` instead lets the node follow the pointer.
#[derive(Clone, Default)]
pub(crate) struct DragSnapState {
    /// Flow-space position of each dragged node when the drag began.
    origins: HashMap<NodeId, egui::Pos2>,
    /// Pointer movement since the drag began, in flow units.
    travelled: egui::Vec2,
}

impl DragSnapState {
    /// Forget the previous drag. Call when a new drag starts.
    pub(crate) fn reset(&mut self) {
        self.origins.clear();
        self.travelled = egui::Vec2::ZERO;
    }

    /// Record one frame of pointer movement (screen pixels).
    pub(crate) fn advance(&mut self, drag_delta: egui::Vec2, transform: &Transform) {
        self.travelled += drag_delta / transform.scale;
    }

    /// Snapped position for `id`, whose position before this frame was `current`.
    fn position(&mut self, id: &NodeId, current: egui::Pos2, frame_delta: egui::Vec2, grid: &[f32; 2]) -> egui::Pos2 {
        // `travelled` already includes this frame's movement, so a node first
        // seen now started at `current + frame_delta - travelled`.
        let travelled = self.travelled;
        let origin = *self
            .origins
            .entry(id.clone())
            .or_insert(current + frame_delta - travelled);
        snap_position(origin + travelled, grid)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-node drag
// ─────────────────────────────────────────────────────────────────────────────

/// Process node drag for a single node. Returns position change if dragged.
pub(crate) fn handle_node_drag<D>(
    node_id: &NodeId,
    drag_delta: egui::Vec2,
    transform: &Transform,
    snap_to_grid: bool,
    snap_grid: &[f32; 2],
    current_pos: egui::Pos2,
    snap_state: &mut DragSnapState,
) -> Option<NodeChange<D>> {
    if drag_delta == egui::Vec2::ZERO {
        return None;
    }

    // Convert screen delta to flow delta
    let flow_delta = egui::vec2(
        drag_delta.x / transform.scale,
        drag_delta.y / transform.scale,
    );

    let new_pos = if snap_to_grid {
        snap_state.position(node_id, current_pos, flow_delta, snap_grid)
    } else {
        egui::pos2(current_pos.x + flow_delta.x, current_pos.y + flow_delta.y)
    };

    Some(NodeChange::Position {
        id: node_id.clone(),
        position: Some(new_pos),
        dragging: Some(true),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-node drag
// ─────────────────────────────────────────────────────────────────────────────

/// Compute position changes for **all** selected nodes when one selected node
/// is being dragged.
///
/// `primary_id` is the node the user is physically dragging.  If it is
/// selected we also move every other selected node by the same flow-space
/// delta, producing a unified multi-selection drag.
///
/// If `primary_id` is NOT selected (or there is only one selected node) the
/// function returns an empty `Vec` so the caller can fall back to the normal
/// single-node drag path.
pub(crate) fn handle_multi_node_drag<D>(
    primary_id: &NodeId,
    drag_delta: egui::Vec2,
    transform: &Transform,
    snap_to_grid: bool,
    snap_grid: &[f32; 2],
    node_lookup: &HashMap<NodeId, InternalNode<D>>,
    snap_state: &mut DragSnapState,
) -> Vec<NodeChange<D>> {
    if drag_delta == egui::Vec2::ZERO {
        return Vec::new();
    }

    // Only trigger multi-drag when the dragged node is itself selected.
    let primary_selected = node_lookup
        .get(primary_id)
        .map(|n| n.node.selected)
        .unwrap_or(false);

    if !primary_selected {
        return Vec::new();
    }

    // Collect all selected nodes (including the primary).
    let selected: Vec<(&NodeId, egui::Pos2)> = node_lookup
        .iter()
        .filter(|(_, n)| n.node.selected && !n.node.hidden)
        .map(|(id, n)| (id, n.internals.position_absolute))
        .collect();

    // A multi-drag is only meaningful when more than one node is selected.
    // With a single selection the normal single-node code path is fine, but
    // returning changes here is harmless (and slightly redundant — we do it
    // anyway so the caller can always use this result when multi-drag fires).
    if selected.is_empty() {
        return Vec::new();
    }

    let flow_delta = egui::vec2(
        drag_delta.x / transform.scale,
        drag_delta.y / transform.scale,
    );

    selected
        .into_iter()
        .filter_map(|(id, flow_pos)| {
            let can_drag = node_lookup
                .get(id)
                .and_then(|n| n.node.draggable)
                .unwrap_or(true);

            if !can_drag {
                return None;
            }

            let new_pos = if snap_to_grid {
                snap_state.position(id, flow_pos, flow_delta, snap_grid)
            } else {
                egui::pos2(flow_pos.x + flow_delta.x, flow_pos.y + flow_delta.y)
            };

            Some(NodeChange::Position {
                id: id.clone(),
                position: Some(new_pos),
                dragging: Some(true),
            })
        })
        .collect()
}

/// Generate drag-end changes for all currently-dragging nodes.
///
/// Call this when a drag gesture finishes so that every dragging node has its
/// `dragging` flag cleared.
pub(crate) fn handle_multi_node_drag_end<D>(
    node_lookup: &HashMap<NodeId, InternalNode<D>>,
) -> Vec<NodeChange<D>> {
    node_lookup
        .iter()
        .filter(|(_, n)| n.node.dragging)
        .map(|(id, _)| NodeChange::Position {
            id: id.clone(),
            position: None,
            dragging: Some(false),
        })
        .collect()
}


#[cfg(test)]
mod snap_tests {
    use super::*;

    #[test]
    fn slow_drag_with_snapping_still_moves_the_node() {
        let transform = Transform { x: 0.0, y: 0.0, scale: 1.0 };
        let grid = [10.0, 10.0];
        let id = NodeId::new("n");
        let mut state = DragSnapState::default();
        let mut pos = egui::pos2(20.0, 20.0);

        // 30 frames of 1 px each: every single step is under half a grid cell.
        for _ in 0..30 {
            let delta = egui::vec2(1.0, 0.0);
            state.advance(delta, &transform);
            let change: Option<NodeChange<()>> =
                handle_node_drag(&id, delta, &transform, true, &grid, pos, &mut state);
            if let Some(NodeChange::Position { position: Some(p), .. }) = change {
                pos = p;
            }
        }

        assert_eq!(pos, egui::pos2(50.0, 20.0));
    }
}
