mod burgers;
mod fixed_time_step;

use burgers::BurgersTestCase;
use fixed_time_step::FixedTimeStep;
use fv::{
    dual_mesh::DualType,
    mesh::Mesh,
    mesh_2d::Mesh2d,
    pde_solver::{PDESolver, SpatialScheme, TemporalScheme},
    time_step::TimeStep,
};
use log::info;
use serde::ser::Serialize;
use serde_json::json;
use std::{fs::File, io::Write};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let lx = 100.0;
    let ly = 100.0;
    let nx = 250;
    let ny = 250;

    let msh = Mesh2d::rect_uniform(lx, nx, ly, ny);

    msh.to_meshb("mesh.meshb").unwrap();

    let test = BurgersTestCase::new(&msh, [4.3, 0.021]);

    // (f(r) - f(l)) / (r - l) = 0.5*(4.5^2 - 1) / (4.5 - 1) = 0.5*5.5
    // shock at t = 25 at x = 25*.5*5.5 = 68.75
    let mut x = test.initial();

    Mesh2d::sol_to_meshb("sol_000.solb", &x).unwrap();
    let solver = test.solver(DualType::Barth);

    let tf = 25.0;
    let n_time_steps = 500;
    let fixed_dt = tf / n_time_steps as f64;
    let mut dt = FixedTimeStep::new(fixed_dt, msh.n_verts());
    let mut t = 0.0;

    let mut steps = Vec::new();
    for i in 0..n_time_steps {
        solver.update_time_step(&x, &mut dt);
        solver
            .explicit_step(
                &mut dt,
                &mut x,
                SpatialScheme::SecondOrderUpwind(fv::limiter::LimiterType::MinMod),
                TemporalScheme::RK3,
            )
            .unwrap();
        t += dt.min();
        info!("Iteration {}: t = {t:.2e}, time_step = {dt}", i + 1);
        let fname = format!("sol_{:03}.solb", i + 1);
        Mesh2d::sol_to_meshb(&fname, &x).unwrap();
        steps.push(json!({
            "time":t,
            "vertex_fields": {
                "u": fname
            }
        }));
    }

    let config = json!({
        "names": {
            "ymin":1,
            "xmax":2,
            "ymax":3,
            "xmin":4,
        },
        "steps":steps
    });

    let mut file = File::create("mesh.json").unwrap();

    let mut buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    config.serialize(&mut ser).unwrap();

    file.write_all(&buf).expect("Cannot write mesh.json");
}
