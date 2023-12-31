use anyhow;
use aoc23;
use aoc23::Location;
use itertools::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Symbol {
    symbol: char,
    row: usize,
    col: usize,
}

impl Symbol {
    fn new(symbol: char, row: usize, col: usize) -> Symbol {
        Symbol { symbol, row, col }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Number {
    value: u32,
    row: usize,
    col_start: usize,
    col_end_incl: usize,
}

impl Number {
    fn adjacent_locations(&self) -> impl Iterator<Item = Location> {
        let mut locations = Vec::with_capacity(8);

        // line above
        if self.row > 0 {
            for col in self.col_start.saturating_sub(1)..=self.col_end_incl.saturating_add(1) {
                locations.push(Location::new_usize(self.row - 1, col));
            }
        }

        // line below
        for col in self.col_start.saturating_sub(1)..=self.col_end_incl.saturating_add(1) {
            locations.push(Location::new_usize(self.row + 1, col));
        }

        // same line
        if self.col_start > 0 {
            locations.push(Location::new_usize(self.row, self.col_start - 1));
        }
        locations.push(Location::new_usize(self.row, self.col_end_incl + 1));

        locations.into_iter()
    }
}

#[derive(Debug, Clone)]
struct Symbols {
    inner: HashMap<usize, HashMap<usize, Symbol>>,
}

impl Symbols {
    fn from_lines<'a>(lines: impl IntoIterator<Item = &'a str>) -> Symbols {
        let symbols: HashMap<usize, HashMap<usize, Symbol>> = lines
            .into_iter()
            .enumerate()
            .map(|(row, s)| {
                let row_symbols = s
                    .chars()
                    .enumerate()
                    .filter(|(_, c)| !c.is_ascii_digit() && c != &'.')
                    .map(|(col, c)| (col, Symbol::new(c, row, col)))
                    .collect();
                (row, row_symbols)
            })
            .collect();
        Symbols { inner: symbols }
    }
}

#[derive(Debug, Clone)]
struct Numbers {
    inner: HashMap<usize, HashMap<usize, Number>>,
}

impl Numbers {
    fn from_lines<'a>(lines: impl IntoIterator<Item = &'a str>) -> Numbers {
        let numbers: HashMap<usize, HashMap<usize, Number>> = lines
            .into_iter()
            .enumerate()
            .map(|(row, s)| {
                let groups = s
                    .chars()
                    .enumerate()
                    .group_by(|(_col, c)| c.is_ascii_digit());
                let mut row_numbers = HashMap::new();
                for group in groups
                    .into_iter()
                    .filter(|(key, _)| *key)
                    .map(|(_, group)| group)
                {
                    let group_v = group.collect_vec();
                    let number_val = group_v
                        .iter()
                        .map(|(_, c)| c.to_digit(10).unwrap())
                        .reduce(|acc, e| acc * 10 + e)
                        .unwrap();
                    let col_start = group_v.first().unwrap().0;
                    let col_end_incl = group_v.last().unwrap().0;
                    let number = Number {
                        value: number_val,
                        col_start,
                        col_end_incl,
                        row,
                    };
                    row_numbers.insert(col_start, number);
                }

                (row, row_numbers)
            })
            .collect();
        Numbers { inner: numbers }
    }
}

impl Symbols {
    fn iter(&self) -> impl Iterator<Item = &Symbol> {
        self.inner.values().flat_map(|m| m.values())
    }
}

impl Numbers {
    fn iter(&self) -> impl Iterator<Item = &Number> {
        self.inner.values().flat_map(|m| m.values())
    }
}

impl Symbol {
    fn is_gear(&self, numbers: &Numbers) -> bool {
        self.symbol == '*' && self.get_adjacent_numbers(numbers).count() == 2
    }

    fn gear_ratio(&self, numbers: &Numbers) -> Option<u32> {
        if self.is_gear(numbers) {
            Some(
                self.get_adjacent_numbers(numbers)
                    .map(|n| n.value)
                    .product(),
            )
        } else {
            None
        }
    }

    fn get_adjacent_numbers<'s, 'a: 's>(
        &'s self,
        numbers: &'a Numbers,
    ) -> impl Iterator<Item = &'a Number> + 's {
        numbers.iter().filter(|n| {
            n.adjacent_locations()
                .any(|loc| loc.row as usize == self.row && loc.col as usize == self.col)
        })
    }
}

fn main() -> anyhow::Result<()> {
    let lines = aoc23::read_input_lines().collect_vec();
    let symbols = Symbols::from_lines(lines.iter().map(|s| s.as_str()));
    let numbers = Numbers::from_lines(lines.iter().map(|s| s.as_str()));

    let sum: u32 = symbols.iter().filter_map(|s| s.gear_ratio(&numbers)).sum();

    println!("{}", sum);
    Ok(())
}
