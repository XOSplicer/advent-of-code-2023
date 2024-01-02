use std::collections::{HashMap, HashSet};

use anyhow;
use aoc23::{self, Direction as LDirection, Location};
use itertools::*;
use petgraph::algo::all_simple_paths;
use petgraph::data::Build;
use petgraph::dot::{Config as DotConfig, Dot};
use petgraph::graph::EdgeReference;
use petgraph::prelude::*;
use petgraph::visit::Visitable;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Entry {
    Path,
    Forest,
    Slope(LDirection),
}

fn parse(lines: impl Iterator<Item = String>) -> HashMap<Location, Entry> {
    println!("Parsing input");
    lines
        .enumerate()
        .flat_map(|(row, line)| {
            line.chars()
                .enumerate()
                .map(move |(col, c)| {
                    (
                        Location::new_usize(row, col),
                        match c {
                            '.' => Entry::Path,
                            '#' => Entry::Forest,
                            '<' => Entry::Slope(LDirection::Left),
                            '>' => Entry::Slope(LDirection::Right),
                            '^' => Entry::Slope(LDirection::Up),
                            'v' => Entry::Slope(LDirection::Down),
                            _ => panic!("invalid entry {}", c),
                        },
                    )
                })
                .collect_vec()
        })
        .collect()
}

struct HikeGraph {
    graph: UnGraph<Location, u32>,
    nodes: HashMap<Location, NodeIndex>,
    entry_node: NodeIndex,
    exit_node: NodeIndex,
}

fn build_graph(pattern: HashMap<Location, Entry>) -> HikeGraph {
    use LDirection::*;
    println!("Building pattern graph");
    let max_row = pattern.keys().map(|loc| loc.row).max().unwrap();
    let max_col = pattern.keys().map(|loc| loc.col).max().unwrap();
    let path_count = pattern.values().filter(|&e| e != &Entry::Forest).count();

    let mut graph = UnGraph::with_capacity(path_count, path_count * 2);
    let mut nodes = HashMap::with_capacity(path_count);

    let mut entry_node = None;
    let mut exit_node = None;

    // find entry node and exit node
    for col in 0..=max_col {
        // entry node
        let loc = Location::new(0, col);
        if pattern.get(&loc) == Some(&Entry::Path) {
            let id = graph.add_node(loc);
            nodes.insert(loc, id);
            entry_node = Some(id);
        }
        // exit node
        let loc = Location::new(max_row, col);
        if pattern.get(&loc) == Some(&Entry::Path) {
            let id = graph.add_node(loc);
            nodes.insert(loc, id);
            exit_node = Some(id);
        }
    }

    // build graph

    for (&loc, entry) in pattern.iter() {
        let neighbors_dir = match entry {
            Entry::Forest => continue,
            Entry::Path | Entry::Slope(_) => vec![Up, Down, Left, Right],
        };

        let neighbors: Vec<Location> = neighbors_dir
            .iter()
            .filter_map(|&dir| {
                let n_loc = loc.apply(dir);
                match pattern.get(&n_loc) {
                    None => None,
                    Some(Entry::Forest) => None,
                    Some(Entry::Path) => Some(n_loc),
                    Some(Entry::Slope(_)) => Some(n_loc),
                }
            })
            .collect_vec();

        let id = *nodes.entry(loc).or_insert_with(|| graph.add_node(loc));
        for n_loc in neighbors {
            let n_id = *nodes.entry(n_loc).or_insert_with(|| graph.add_node(n_loc));
            // add_edge does not work, as duplicate edges will have incorrect traversal
            graph.update_edge(id, n_id, 1);
        }
    }

    HikeGraph {
        graph,
        nodes,
        entry_node: entry_node.unwrap(),
        exit_node: exit_node.unwrap(),
    }
}

fn edge_next_and_len(edge: EdgeReference<'_, u32>, curr: NodeIndex) -> Option<(NodeIndex, u32)> {
    if edge.source() == curr {
        Some((edge.target(), *edge.weight()))
    } else if edge.target() == curr {
        Some((edge.source(), *edge.weight()))
    } else {
        None
    }
}

fn next_condensed_nodes(graph: &UnGraph<Location, u32>, node: NodeIndex) -> Vec<(NodeIndex, u32)> {
    graph
        .edges(node)
        .map(|e| edge_next_and_len(e, node).unwrap())
        .map(|(neighbor_node, neighbor_edge_len)| {
            let mut last = node;
            let mut curr = neighbor_node;
            let mut total_edge_len = neighbor_edge_len;

            while graph.edges(curr).count() == 2 {
                let edges = graph
                    .edges(curr)
                    .filter(|e| e.source() != last && e.target() != last)
                    .collect_vec();
                assert_eq!(edges.len(), 1);
                let edge = edges.into_iter().next().unwrap();
                let (next, edge_len) = edge_next_and_len(edge, curr).unwrap();
                total_edge_len += edge_len;
                last = curr;
                curr = next;
            }

            (curr, total_edge_len)
        })
        .collect_vec()
}

fn condense_graph(graph: HikeGraph) -> HikeGraph {
    println!("Building condensed graph");

    let orig_graph = &graph.graph;
    let mut new_graph = UnGraph::new_undirected();
    let mut new_nodes = HashMap::new();

    let new_entry_node = new_graph.add_node(orig_graph[graph.entry_node]);
    new_nodes.insert(orig_graph[graph.entry_node], new_entry_node);
    let new_exit_node = new_graph.add_node(orig_graph[graph.exit_node]);
    new_nodes.insert(orig_graph[graph.exit_node], new_exit_node);

    let mut orig_dfs_stack = vec![graph.exit_node, graph.entry_node];
    let mut orig_dfs_visit_map = orig_graph.visit_map();

    while let Some(orig_node) = orig_dfs_stack.pop() {
        if !orig_dfs_visit_map.put(orig_node.index()) {
            let new_id = *new_nodes.get(&orig_graph[orig_node]).unwrap();
            for (orig_n_node, edge_len) in next_condensed_nodes(&orig_graph, orig_node) {
                let new_n_id = new_graph.add_node(orig_graph[orig_n_node]);
                new_nodes.insert(orig_graph[orig_n_node], new_n_id);
                new_graph.update_edge(new_id, new_n_id, edge_len);
                orig_dfs_stack.push(orig_n_node);
                println!("{:?} -({})- {:?}", orig_graph[orig_n_node], edge_len, orig_graph[orig_n_node]);
            }
        }
    }

    HikeGraph {
        graph: new_graph,
        nodes: new_nodes,
        entry_node: new_entry_node,
        exit_node: new_entry_node,
    }
}

fn find_longest_hike(graph: HikeGraph) -> u32 {
    println!("Finding longest path in graph (bruteforce)");
    let mut longest = 0;
    all_simple_paths::<Vec<_>, _>(&graph.graph, graph.entry_node, graph.exit_node, 0, None)
        .map(|path| {
            let mut len = 0;
            for (&a, &b) in path.iter().tuple_windows() {
                let edge = graph.graph.find_edge(a, b).unwrap();
                let edge_len = graph.graph[edge];
                len += edge_len;
            }
            len
        })
        .map(|len| {
            longest = longest.max(len);
            longest
        })
        .enumerate()
        .inspect(|(i, len)| {
            if i % 1_000 == 0 {
                println!("{} -> {}", i, len);
            }
        })
        .map(|(_, len)| len)
        .max()
        .unwrap()
}

fn main() -> anyhow::Result<()> {
    // let lines = aoc23::read_input_lines();
    let lines = aoc23::read_file_lines("input/23-example.txt");
    let pattern = parse(lines);
    println!("Parsed pattern ({} entries)", pattern.len());
    let graph = build_graph(pattern);
    println!(
        "Built pattern graph ({} nodes, {} edges)",
        graph.graph.node_count(),
        graph.graph.edge_count()
    );
    let graph = condense_graph(graph);
    println!(
        "Built condensed graph ({} nodes, {} edges)",
        graph.graph.node_count(),
        graph.graph.edge_count()
    );
    println!("Condensed graph: \n\n{:?}\n\n", Dot::new(&graph.graph));
    let sum: u32 = find_longest_hike(graph);
    println!("{}", sum);
    Ok(())
}
