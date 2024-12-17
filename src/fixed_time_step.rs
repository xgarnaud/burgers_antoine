use fv::time_step::{TimeStep, TIME_STEP_TOL};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

pub struct FixedTimeStep {
    dt: f64,
    n_verts: usize,
    dt_min: f64,
}

impl FixedTimeStep {
    pub fn new(dt: f64, n_verts: usize) -> Self {
        Self {
            dt,
            n_verts,
            dt_min: f64::MAX,
        }
    }
}

impl std::fmt::Display for FixedTimeStep {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:.2e} (cfl={:.2e})", self.dt, self.dt / self.dt_min)
    }
}

impl TimeStep for FixedTimeStep {
    fn is_constant(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.dt_min = f64::MAX;
    }

    fn get_mut(&mut self, _i: usize) -> &mut f64 {
        &mut self.dt_min
    }

    fn update_iter<I: IndexedParallelIterator<Item = f64>>(&mut self, vals: I) {
        self.dt_min = vals
            .fold(|| self.dt_min, f64::min)
            .reduce(|| self.dt_min, f64::min);
    }

    fn set(&mut self, val: f64) {
        if (self.dt - val).abs() > TIME_STEP_TOL {
            panic!("fixed time step {:.2e} cannot be set to {val:.2e}", self.dt);
        }
    }

    fn finalize(&mut self) {}

    fn min(&self) -> f64 {
        self.dt
    }

    fn max(&self) -> f64 {
        self.dt
    }

    fn par_iter(&self) -> impl IndexedParallelIterator<Item = f64> {
        (0..self.n_verts).into_par_iter().map(|_| self.dt)
    }

    fn seq_iter(&self) -> impl ExactSizeIterator<Item = f64> {
        (0..self.n_verts).map(|_| self.dt)
    }
}
