// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::brent::BrentOpt;

/// Test function: `f(x) = exp(-x) - exp(5-x/2)`
/// xmin == 2 log(2 exp(-5))
/// xmin ~= -8.6137056388801093812
/// f(xmin) == -exp(10)/4
/// f(xmin) ~= -5506.6164487016791292
// struct TestFunc {}

/// Nelson-Siegel (1987) model parameters.
pub struct NelsonSiegel {
    beta0: f64,
    beta1: f64,
    beta2: f64,
    lambda: f64,
}

impl CostFunction for NelsonSiegel {
    // one dimensional problem, no vector needed
    type Param = f64;
    type Output = f64;

    fn cost(&self, tau: &Self::Param) -> Result<Self::Output, Error> {
        // assert!(
        //     date > OffsetDateTime::now_utc(),
        //     "Date must be in the future."
        // );

        // let tau = DayCounter::day_count_factor(
        //     OffsetDateTime::now_utc(),
        //     date,
        //     &DayCountConvention::Actual365,
        // );

        let term1 = self.lambda * (1. - f64::exp(-tau / self.lambda)) / tau;
        let term2 = term1 - f64::exp(-tau / self.lambda);

        Ok(self.beta0 + self.beta1 * term1 + self.beta2 * term2)
    }
}

fn main() {
    let cost = NelsonSiegel {
        beta0: 0.5,
        beta1: 0.5,
        beta2: 0.5,
        lambda: 0.5,
    };

    let solver = BrentOpt::new(0., 30.);

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(100))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    println!("Result of brent:\n{res}");
}
