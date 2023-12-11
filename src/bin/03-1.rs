use anyhow;
use aoc23;
use itertools::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Symbol {
    symbol: char,
    line: usize,
    col: usize,
}

impl Symbol {
    fn new(symbol: char, line: usize, col: usize) -> Symbol {
        Symbol { symbol, line, col }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Number {
    value: u32,
    line: usize,
    col_start: usize,
    col_end_incl: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Location {
    line: usize,
    col: usize,
}

impl Number {
    fn adjacent_locations(&self) -> impl Iterator<Item = Location> {
        let mut locations = Vec::with_capacity(8);

        // line above
        if self.line > 0 {
            for col in self.col_start.saturating_sub(1)..=self.col_end_incl.saturating_add(1) {
                locations.push(Location {
                    line: self.line - 1,
                    col,
                });
            }
        }

        // line below
        for col in self.col_start.saturating_sub(1)..=self.col_end_incl.saturating_add(1) {
            locations.push(Location {
                line: self.line + 1,
                col,
            });
        }

        // same line
        if self.col_start > 0 {
            locations.push(Location {
                line: self.line,
                col: self.col_start - 1,
            });
        }
        locations.push(Location {
            line: self.line,
            col: self.col_end_incl + 1,
        });

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
            .map(|(line, s)| {
                let line_symbols = s
                    .chars()
                    .enumerate()
                    .filter(|(_, c)| !c.is_ascii_digit() && c != &'.')
                    .map(|(col, c)| (col, Symbol::new(c, line, col)))
                    .collect();
                (line, line_symbols)
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
            .map(|(line, s)| {
                let groups = s
                    .chars()
                    .enumerate()
                    .group_by(|(col, c)| c.is_ascii_digit());
                let mut line_numbers = HashMap::new();
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
                        line,
                    };
                    line_numbers.insert(col_start, number);
                }

                (line, line_numbers)
            })
            .collect();
        Numbers { inner: numbers }
    }
}

impl Symbols {
    fn get(&self, line: usize, col: usize) -> Option<&Symbol> {
        self.inner.get(&line).and_then(|m| m.get(&col))
    }
}

fn main() -> anyhow::Result<()> {
    let lines = aoc23::read_input_lines().collect_vec();
    let symbols = Symbols::from_lines(lines.iter().map(|s| s.as_str()));
    let numbers = Numbers::from_lines(lines.iter().map(|s| s.as_str()));

    let sum: u32 = numbers
        .inner
        .values()
        .flat_map(|m| m.values())
        .filter_map(|number| {
            if number
                .adjacent_locations()
                .any(|loc| symbols.get(loc.line, loc.col).is_some())
            {
                Some(number.value)
            } else {
                None
            }
        })
        .sum();

    println!("{}", sum);
    Ok(())
}