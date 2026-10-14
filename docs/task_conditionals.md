# Task Conditionals

## Run a demo

To run the conditionals test suite:

```bash
python examples/test_conditionals.py --headless
```
The test will use the following test scene:

<img src="images/conditionals_scene.gif" alt="Conditionals scene" style="max-width:600px;">

## Conditionals:
See [`robolab/robolab/core/task/conditionals.py`](../robolab/core/task/conditionals.py) for implementation details.

## Frames in spatial conditions

Spatial conditions (`object_right_of`, `object_left_of`, `object_in_front_of`, `object_behind`) support different frame of reference modes:

- **`frame_of_reference="robot"`** (default): Uses the robot's egocentric perspective
  - X-axis: robot's forward direction
  - Y-axis: robot's left direction
- **`frame_of_reference="world"`**: Uses global world coordinates

The **`mirrored=False`** (default) uses the robot's natural perspective. Set **`mirrored=True`** for a flipped XY perspective, as if viewing the scene from across the robot.

<img src="images/conditionals_frame_overlay.png" alt="Frame of Reference Overlay" style="max-width:600px; width:100%;">

## Geometric Containment

### `object_in_container` / `object_inside` / `object_outside_of` / `object_enclosed` — Centroid-in-Convex-Hull Check

Containment is checked by transforming the **centroid of the inside-object's convex-hull vertices** into the **container's local frame** and testing it against the container's **convex-hull face planes**. The predicate is fully orientation-invariant — a flipped, tipped, or rotated container is handled correctly because the test happens entirely in the container's own coordinate system.

The container's convex hull is built once at scene-load from the prim's mesh points (via `scipy.spatial.ConvexHull`), cached on the `WorldState`, and reused on every per-step evaluation. For "open-top" semantics, the hull's top-facing faces (those with outward normal projecting ≥ 0.7 onto the container's local +z) are dropped, so the polytope is unbounded along the opening direction — an object lifted above the rim still reads as inside.

#### Mathematical formulation

Let $\mathbf{p}_c, \mathbf{q}_c$ be the container's world position and quaternion, $\mathbf{p}_o, \mathbf{q}_o$ the inside-object's world position and quaternion, $\bar{\mathbf{v}}$ the centroid of the object's hull vertices in the **object's own local frame**, and $\{(\mathbf{n}_i, d_i)\}_{i=1}^F$ the container's hull face planes (outward normal + offset) in the **container's local frame**.

The centroid is transformed object-local → world → container-local:

```math
\mathbf{x}_w = \mathbf{q}_o \cdot \bar{\mathbf{v}} + \mathbf{p}_o
```

```math
\mathbf{x}_c = \mathbf{q}_c^{-1} \cdot (\mathbf{x}_w - \mathbf{p}_c)
```

The predicate then evaluates a single boolean:

```math
\text{inside} \;=\; \max_i \big( \mathbf{n}_i \cdot \mathbf{x}_c + d_i \big) \;\le\; 0
```

i.e., the centroid satisfies every face's half-space constraint simultaneously.

| variant | face set used | semantics |
| -------- | -------------- | ----------- |
| `object_in_container` / `object_inside` | open-top (top faces dropped, $n_z \ge 0.7$ filter) | true iff the centroid is in the cavity, including the air column above the rim |
| `object_outside_of` | open-top (negation of in_opentop_container) | true iff the centroid is outside the cavity / column |
| `object_enclosed` | full closed hull | true iff the centroid is fully bounded (all faces, no open top) |

#### USD scale handling

Mesh points are extracted via the prim's full local-to-world transform (which absorbs any nested `xformOp:scale` and USD `metersPerUnit` conversions), then re-expressed in the prim's rotated frame **without undoing the scale** (`Gf.Matrix4d.RemoveScaleShear()` keeps only translation+rotation when inverting). This keeps the hull dimensions in world meters regardless of how the source USD was authored — a container in cm with `xformOp:scale = 0.01` produces the same hull as one authored directly in meters at scale 1.

#### Why centroid (not corners or fraction-of-vertices)

A single point at the object's hull centroid is the closest match to human intuition for "in" / "out" and gives clean boolean semantics with no thresholds. Two earlier attempts failed:

- **OBB corners:** elongated objects (e.g. a banana whose tips poke over the rim) have all 8 corners *outside* the cavity even when the body is clearly inside.
- **Fraction-of-hull-vertices** (e.g. ≥ 50% inside): introduces an asymmetry pathology — a banana with one tip dangling into the bin column lands in a marginal frac ≈ 0.3-0.6 range and fails both "mostly out" and strict "all out" thresholds.

The centroid-in-hull test is also ~30× cheaper per step than the per-vertex frac aggregation it replaced (one rotation + one matmul instead of $V$).

#### Performance

The hull data (vertices, full plane set, open-top plane set, hull centroid) is precomputed once per body in `LocalHull` (see `robolab/core/task/hull_check.py`) and cached on the `WorldState`. Per-step cost on the hot path: one `quat_apply`, one `quat_apply_inverse`, one $(F, 3) \cdot (3,) + (F,)$ matmul-and-max — fully vectorizable across envs.

## Contact Force Cone Detection

### `object_on_top` — Stable Support Detection

The `object_on_top` conditional uses physics-based contact force analysis to determine if an object is stably supported on a surface.

#### Mathematical Formulation

Let $\mathbf{f} = [f_x, f_y, f_z]^\top$ be the contact force from surface $B$ acting on object $A$, expressed in the **world frame** (Z-up).

For $A$ to be stably supported on $B$, the force must lie within an **upward cone**:

- **Cone axis**: $\hat{n} = [0, 0, 1]^\top$ (upward direction)
- **Cone half-angle**: $\theta_{\max}$ (default 45°)

**Conditions for stable support:**

```math
\begin{aligned}
\text{1. Meaningful contact:} \quad & \|\mathbf{f}\| > f_{\min} \\
\text{2. Upward force:} \quad & f_z > 0 \\
\text{3. Within cone:} \quad & f_z \geq \|\mathbf{f}\| \cdot \cos(\theta_{\max})
\end{aligned}
```

The cone constraint (3) can be derived from the dot product:

```math
\cos(\theta) = \frac{\mathbf{f} \cdot \hat{n}}{\|\mathbf{f}\|} = \frac{f_z}{\|\mathbf{f}\|} \geq \cos(\theta_{\max})
```

#### Comparison with Geometric Detection

| Function | Method | Use Case |
|----------|--------|----------|
| `object_on_top` | Contact force cone | Stable resting detection (terminations) |
| `object_above` | Bounding box geometry | Position-based checks (lifted above surface) |
| `object_in_container` / `object_inside` / `object_outside_of` / `object_enclosed` | Centroid of object's hull verts vs container's convex-hull face planes (orientation-invariant; open-top variant drops top faces so the air column above the rim counts as "in") | Containment detection (terminations, subtasks) |

#### Usage

```python
# Check if orange is stably resting on plate
object_on_top(env, object="orange", reference_object="plate", require_gripper_detached=True)

# Geometric check (e.g., for lifted detection)
object_above(env, object="orange", reference_object="table", z_margin=0.05)
```

---

## Details
### Logicals
For functions that support logicals, the available logicals are:
- `any`: if at least 1 object satisfies the condition
- `all`: All objects need to satisfy the condition
- `choose`: Given the set of `objects` with size `N`, exactly `K` objects must satisfy the condition.

### Function decorators

#### Atomic Functions
Base functions; can be used for task `Terminations` as well as `subtasks`.

#### Composite Functions
These expand into multiple atomic subtasks. These cannot be used for `Terminations`.
- `pick_and_place(object, container, logical)`: Picks up objects and places them in a container
  - Automatically creates the sequence: grab → move above → drop → verify in container
  - Supports multiple objects with "all" or "any" completion logic
