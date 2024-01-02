use anyhow;
use aoc23;
use derive_more::{Add, Constructor, From, Into, Sub};
use itertools::*;
use petgraph::algo::{has_path_connecting, DfsSpace};
use petgraph::dot::{Config as DotConfig, Dot};
use petgraph::visit::Reversed;
use petgraph::{data::Build, prelude::*};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
struct BrickId(u16);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into, Add, Sub, Constructor)]
struct Point3D {
    x: i32,
    y: i32,
    z: i32,
}

impl Point3D {
    fn set_x(mut self, x: i32) -> Self {
        self.x = x;
        self
    }
    fn set_y(mut self, y: i32) -> Self {
        self.y = y;
        self
    }
    fn set_z(mut self, z: i32) -> Self {
        self.z = z;
        self
    }
    fn add_axis(self, axis: Axis, offset: i32) -> Self {
        match axis {
            Axis::X => self.set_x(self.x + offset),
            Axis::Y => self.set_y(self.y + offset),
            Axis::Z => self.set_z(self.z + offset),
        }
    }
    fn to_tuple(self) -> (i32, i32, i32) {
        self.into()
    }
    fn xy(self) -> Point2D {
        Point2D {
            x: self.x,
            y: self.y,
        }
    }
}

impl std::fmt::Debug for Point3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_tuple())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into, Add, Sub, Constructor)]
struct Point2D {
    x: i32,
    y: i32,
}

impl Point2D {
    fn set_x(mut self, x: i32) -> Self {
        self.x = x;
        self
    }
    fn set_y(mut self, y: i32) -> Self {
        self.y = y;
        self
    }
    fn with_z(self, z: i32) -> Point3D {
        Point3D {
            x: self.x,
            y: self.y,
            z,
        }
    }
    fn to_tuple(self) -> (i32, i32) {
        self.into()
    }
}

impl std::fmt::Debug for Point2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_tuple())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BrickShape {
    start: Point3D,
    end_incl: Point3D,
}

impl BrickShape {
    fn is_single_cube(&self) -> bool {
        self.start == self.end_incl
    }
    fn is_x_long(&self) -> bool {
        self.start.x != self.end_incl.x
            && self.start.y == self.end_incl.y
            && self.start.z == self.end_incl.z
    }
    fn is_y_long(&self) -> bool {
        self.start.x == self.end_incl.x
            && self.start.y != self.end_incl.y
            && self.start.z == self.end_incl.z
    }
    fn is_z_long(&self) -> bool {
        self.start.x == self.end_incl.x
            && self.start.y == self.end_incl.y
            && self.start.z != self.end_incl.z
    }

    fn major_len(&self) -> i32 {
        let s = if self.is_single_cube() {
            1
        } else if self.is_x_long() {
            self.end_incl.x - self.start.x
        } else if self.is_y_long() {
            self.end_incl.y - self.start.y
        } else if self.is_z_long() {
            self.end_incl.z - self.start.z
        } else {
            panic!("Invalid shape: {:?}", self);
        };
        // correct for end_incl
        s + s.signum()
    }

    fn major_axis(&self) -> Axis {
        if self.is_single_cube() {
            Axis::X
        } else if self.is_x_long() {
            Axis::X
        } else if self.is_y_long() {
            Axis::Y
        } else if self.is_z_long() {
            Axis::Z
        } else {
            panic!("Invalid shape: {:?}", self);
        }
    }

    fn min_z(&self) -> i32 {
        self.start.z.min(self.end_incl.z)
    }

    fn to_aa_shape(&self) -> AAShape {
        AAShape {
            axis: self.major_axis(),
            len: self.major_len(),
            start: self.start,
        }
    }

    fn sub_z(&self, z_offset: i32) -> Self {
        BrickShape {
            start: self.start.set_z(self.start.z - z_offset),
            end_incl: self.end_incl.set_z(self.end_incl.z - z_offset),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Axis Aligned Shape
struct AAShape {
    axis: Axis,
    len: i32,
    start: Point3D,
}

impl AAShape {
    fn cubes(&self) -> impl Iterator<Item = Point3D> + '_ {
        let a = 0.min(self.len);
        let b = 0.max(self.len);
        (a..b).map(|offset| self.start.add_axis(self.axis, offset))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Brick {
    id: BrickId,
    shape: BrickShape,
}

fn parse(lines: impl Iterator<Item = String>) -> BTreeMap<BrickId, Brick> {
    let mut id = 0_u16;
    lines
        .map(|line| {
            id += 1;
            let (start_s, end_s) = line.split_once('~').unwrap();
            let (x, y, z) = start_s
                .split(',')
                .map(|s| s.parse::<i32>().unwrap())
                .collect_tuple()
                .unwrap();
            let start = Point3D::new(x, y, z);
            let (x, y, z) = end_s
                .split(',')
                .map(|s| s.parse::<i32>().unwrap())
                .collect_tuple()
                .unwrap();
            let end_incl = Point3D::new(x, y, z);
            let shape = BrickShape { start, end_incl };
            // dbg!(shape);
            assert!(shape.is_single_cube() || !(shape.is_x_long() && shape.is_y_long()));
            assert!(shape.is_single_cube() || !(shape.is_y_long() && shape.is_z_long()));
            assert!(shape.is_single_cube() || !(shape.is_x_long() && shape.is_z_long()));
            let brick = Brick {
                id: BrickId(id),
                shape,
            };
            (brick.id, brick)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct FilledCubes {
    // inner: HashMap<Point3D, BrickId>,
    // BTreeMap<(X, Y), BTreeMap<Z, BrickId>>
    inner: BTreeMap<Point2D, BTreeMap<i32, BrickId>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilledCube {
    Floor,
    Brick(BrickId),
}

impl FilledCube {
    fn as_brick_id(&self) -> Option<&BrickId> {
        match self {
            FilledCube::Brick(id) => Some(id),
            _ => None,
        }
    }
    fn is_floor(&self) -> bool {
        match self {
            FilledCube::Floor => true,
            _ => false,
        }
    }
}

impl FilledCubes {
    fn insert(&mut self, at: Point3D, id: BrickId) {
        if at.z > 0 {
            let old = self.inner.entry(at.xy()).or_default().insert(at.z, id);
            assert_eq!(old, None);
        }
    }

    fn extend(&mut self, iter: impl IntoIterator<Item = (Point3D, BrickId)>) {
        let iter = iter.into_iter();
        for (at, id) in iter {
            self.insert(at, id);
        }
    }

    fn insert_shape(&mut self, shape: AAShape, id: BrickId) {
        self.extend(shape.cubes().map(|c| (c, id)));
    }

    fn insert_brick(&mut self, brick: &Brick) {
        self.insert_shape(brick.shape.to_aa_shape(), brick.id);
    }

    fn get(&self, at: &Point3D) -> Option<FilledCube> {
        if at.z <= 0 {
            Some(FilledCube::Floor)
        } else {
            self.inner
                .get(&at.xy())
                .and_then(|m| m.get(&at.z))
                .map(|&id| FilledCube::Brick(id))
        }
    }

    fn get_max_z(&self, at: &Point2D) -> (i32, FilledCube) {
        self.inner
            .get(at)
            .and_then(|m| m.last_key_value())
            .map(|(&z, &id)| (z, FilledCube::Brick(id)))
            .unwrap_or((0, FilledCube::Floor))
    }

    fn brick_cubes(&self) -> impl Iterator<Item = (Point3D, BrickId)> + '_ {
        self.inner
            .iter()
            .flat_map(|(xy, m)| m.iter().map(|(&z, &id)| (xy.with_z(z), id)))
    }

    fn len(&self) -> usize {
        self.inner.values().map(|m| m.values().count()).sum()
    }
}

fn fall_brick(brick: &Brick, filled_cubes: &mut FilledCubes) -> Brick {
    let dropped_z_height = brick
        .shape
        .to_aa_shape()
        .cubes()
        .map(|cube| filled_cubes.get_max_z(&cube.xy()).0)
        .max()
        .unwrap()
        // one above the current max z at each location
        + 1;
    let z_offset = brick.shape.min_z() - dropped_z_height;
    let dropped_shape = brick.shape.sub_z(z_offset);
    let dropped_brick = Brick {
        id: brick.id,
        shape: dropped_shape,
    };
    filled_cubes.insert_brick(&dropped_brick);
    dropped_brick
}

fn fall_bricks(mut bricks: Vec<Brick>) -> (Vec<Brick>, FilledCubes) {
    bricks.sort_by_key(|b| b.shape.min_z());

    // TODO: maybe it is needed to build a snapshot dependency
    // list before letting the bricks fall

    let mut filled_cubes = FilledCubes::default();
    let new_bricks = bricks
        .into_iter()
        .map(|brick| fall_brick(&brick, &mut filled_cubes))
        .collect_vec();

    (new_bricks, filled_cubes)
}

struct SupportGraph {
    nodes: HashMap<BrickId, NodeIndex>,
    floor_node: NodeIndex,
    graph: StableDiGraph<FilledCube, ()>,
}

fn support_graph(filled_cubes: &FilledCubes) -> SupportGraph {
    let mut nodes: HashMap<BrickId, NodeIndex> = HashMap::new();
    let mut graph = StableDiGraph::new();

    let floor_node = graph.add_node(FilledCube::Floor);

    for (p, lower_id) in filled_cubes.brick_cubes() {
        let lower_node = *nodes
            .entry(lower_id)
            .or_insert_with(|| graph.add_node(FilledCube::Brick(lower_id)));
        if p.z == 1 {
            graph.add_edge(lower_node, floor_node, ());
        }
        if let Some(FilledCube::Brick(upper_id)) = filled_cubes.get(&p.set_z(p.z + 1)) {
            if upper_id != lower_id {
                let upper_node = *nodes
                    .entry(upper_id)
                    .or_insert_with(|| graph.add_node(FilledCube::Brick(upper_id)));
                graph.add_edge(upper_node, lower_node, ());
            }
        }
    }
    SupportGraph {
        nodes,
        floor_node,
        graph,
    }
}

fn removable_bricks(support_graph: &SupportGraph) -> u32 {
    let mut count = 0;
    for (_id, &node) in &support_graph.nodes {
        let mut graph_without_node = support_graph.graph.clone();
        let supported_by_node = graph_without_node
            .neighbors_directed(node, Direction::Incoming)
            .collect_vec();
        graph_without_node.remove_node(node);
        let mut dfs_space = DfsSpace::new(&graph_without_node);
        let all_supported_are_stable = supported_by_node.into_iter().all(|supported_node| {
            has_path_connecting(
                &graph_without_node,
                supported_node,
                support_graph.floor_node,
                Some(&mut dfs_space),
            )
        });
        if all_supported_are_stable {
            count += 1;
        }
    }
    count
}

fn additional_falling_bricks(bricks: &Vec<Brick>, remove_brick: BrickId) -> usize {
    let mut bricks_without_removed: HashMap<BrickId, &Brick> =
        bricks.iter().map(|b| (b.id, b)).collect();
    bricks_without_removed
        .remove(&remove_brick)
        .expect("removed brick should be present");
    let bricks_without_removed_vec = bricks_without_removed
        .values()
        .copied()
        .copied()
        .collect_vec();
    let (fallen_bricks, _) = fall_bricks(bricks_without_removed_vec);
    let count = fallen_bricks
        .into_iter()
        .filter(|b| bricks_without_removed.get(&b.id).unwrap() != &b)
        .count();

    count
}

fn total_falling_bricks(bricks: &Vec<Brick>) -> usize {
    bricks
        .iter()
        .map(|brick| additional_falling_bricks(bricks, brick.id))
        .sum()
}

fn main() -> anyhow::Result<()> {
    let lines = aoc23::read_input_lines();
    let snapshot = parse(lines);

    let bricks = snapshot.values().copied().collect_vec();
    println!("Snapshot ({} bricks)", bricks.len());
    let (fallen_bricks, filled_cubes) = fall_bricks(bricks);
    println!("Fallen ({} bricks)", fallen_bricks.len());
    println!("Filled cubes ({} cubes)", filled_cubes.len());

    let support_graph = support_graph(&filled_cubes);
    println!(
        "Support graph ({} nodes, {} edges):\n\n",
        support_graph.graph.node_count(),
        support_graph.graph.edge_count()
    );

    let sum = total_falling_bricks(&fallen_bricks);
    println!("{}", sum);
    Ok(())
}
