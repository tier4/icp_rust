// Linear time median search
use num_traits::Num;
use std::cmp::PartialOrd;
use std::marker::Copy;

pub fn median<T: Num + PartialOrd + Copy>(input: &Vec<T>) -> Option<f64>
where
    T: Into<f64>,
{
    let n = input.len();
    if n == 0 {
        return None;
    }
    if n % 2 == 1 {
        let a: f64 = find_separated(input, n / 2).into();
        return Some(a);
    }
    let b: f64 = find_separated(input, n / 2 - 1).into();
    let c: f64 = find_separated(input, n / 2 - 0).into();
    Some((b + c) / 2.)
}

fn find_separated<T: Num + PartialOrd + Copy>(input: &Vec<T>, index: usize) -> T
where
    T: Into<f64>,
{
    if input.len() == 1 {
        return input[0];
    }

    let pivot = input[0];
    let mut lower = vec![];
    let mut pivots = vec![];
    let mut upper = vec![];
    for e in input {
        if *e < pivot {
            lower.push(*e);
            continue;
        }
        if *e > pivot {
            upper.push(*e);
            continue;
        }
        pivots.push(*e);
    }

    let nl = lower.len();
    let pl = pivots.len();

    if index < nl {
        return find_separated(&lower, index);
    }
    if index >= nl + pl {
        return find_separated(&upper, index - nl - pl);
    }
    return pivots[0];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        let input = vec![-9, -6, -4, -1, -6,  5,  8,  5,  5,  4];
        assert_eq!(median(&input), Some(1.5));

        let input = vec![
            15, 34, 26, -76, -19, 25, 93, -99, -52, 12, 6, -70, 59, 78, 69, -6, -33, 2, -27
        ];
        assert_eq!(median(&input), Some(6.0));

        let input = vec![-19., 38., -45., 35., 36., 68., 26., -27., 52., 41.];
        assert_eq!(median(&input), Some(35.5));

        let input: Vec<f64> = vec![];
        assert_eq!(median(&input), None);

        let input = vec![50];
        assert_eq!(median(&input), Some(50.));

        let input = vec![10, 11];
        assert_eq!(median(&input), Some(10.5));
    }
}
